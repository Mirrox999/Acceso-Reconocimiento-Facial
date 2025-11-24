import os
import cv2
import numpy as np
from datetime import datetime, timedelta
import pickle
import tkinter as tk
from tkinter import messagebox, simpledialog, ttk
from PIL import Image, ImageTk
import threading
import time

class ImprovedFacialRecognizer:
    def __init__(self):
        self.known_face_features = []
        self.known_face_names = []
        self.known_face_timestamps = []
        self.face_locations = []
        
        # Detector de caras con mejores parámetros
        self.face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # CONFIGURACIÓN MEJORADA PARA LBPH
        self.face_recognizer = cv2.face.LBPHFaceRecognizer_create(
            radius=2,        # Aumentado para capturar más patrones
            neighbors=16,    
            grid_x=8,
            grid_y=8,
            threshold=120.0  
        )
        
        self.min_size = (60, 60)
        self.confidence_threshold = 120  
        self.recognition_count = 1
        self.recognized_frames = {}
        self.last_detected_faces = {}
        self.detection_persistence = 10
        self.has_trained_model = False
        
    def enhance_image(self, image):
        # Convertir a escala de grises
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        denoised = cv2.fastNlMeansDenoising(gray)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(denoised)
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        sharpened = cv2.filter2D(enhanced, -1, kernel)
        normalized = cv2.normalize(sharpened, None, 0, 255, cv2.NORM_MINMAX)
        
        return normalized
        
    def preprocess_face(self, face_image):
        face = cv2.resize(face_image, (200, 200))
        face = self.enhance_image(face)
        face = cv2.bilateralFilter(face, 9, 75, 75)
        
        return face
    
    def detect_faces_multiple_scales(self, gray):
        params_sets = [
            {'scaleFactor': 1.1, 'minNeighbors': 6, 'minSize': (60, 60), 'maxSize': (300, 300)},
            {'scaleFactor': 1.15, 'minNeighbors': 5, 'minSize': (50, 50), 'maxSize': (250, 250)},
        ]
        
        all_faces = []
        for params in params_sets:
            faces = self.face_detector.detectMultiScale(
                gray, 
                scaleFactor=params['scaleFactor'],
                minNeighbors=params['minNeighbors'],
                minSize=params['minSize'],
                maxSize=params['maxSize'],
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            all_faces.extend(faces)
        
        if len(all_faces) > 1:
            all_faces = self.remove_duplicate_faces(all_faces, overlap_threshold=0.2)  # Más estricto
        
        filtered_faces = []
        for (x, y, w, h) in all_faces:
            area = w * h
            if 3000 <= area <= 50000:  # Ajusta estos valores según tu cámara
                filtered_faces.append((x, y, w, h))
        
        return filtered_faces
    
    def remove_duplicate_faces(self, faces, overlap_threshold=0.2):
        if len(faces) <= 1:
            return faces
        
        # Convertir a formato con área y calcular centros
        faces_with_info = []
        for (x, y, w, h) in faces:
            area = w * h
            center_x = x + w // 2
            center_y = y + h // 2
            faces_with_info.append({
                'x': x, 'y': y, 'w': w, 'h': h,
                'area': area,
                'center': (center_x, center_y)
            })
        faces_with_info.sort(key=lambda f: f['area'], reverse=True)
        
        keep = []
        for current_face in faces_with_info:
            is_duplicate = False
            
            for kept_face in keep:
                # Calcular IoU (Intersection over Union)
                x1 = max(current_face['x'], kept_face['x'])
                y1 = max(current_face['y'], kept_face['y'])
                x2 = min(current_face['x'] + current_face['w'], kept_face['x'] + kept_face['w'])
                y2 = min(current_face['y'] + current_face['h'], kept_face['y'] + kept_face['h'])
                
                if x2 > x1 and y2 > y1:
                    intersection = (x2 - x1) * (y2 - y1)
                    union = current_face['area'] + kept_face['area'] - intersection
                    iou = intersection / union if union > 0 else 0
                    
                    center_dist = np.sqrt(
                        (current_face['center'][0] - kept_face['center'][0])**2 + 
                        (current_face['center'][1] - kept_face['center'][1])**2
                    )
                    
                    if iou > overlap_threshold or center_dist < min(current_face['w'], current_face['h']) * 0.5:
                        is_duplicate = True
                        break
            
            if not is_duplicate:
                keep.append(current_face)
        
        # Convertir de vuelta al formato original
        return [(f['x'], f['y'], f['w'], f['h']) for f in keep]
    
    def validate_face_quality(self, face_region, min_variance=100):
        """Valida que la región detectada sea realmente una cara"""
        try:
            # Calcular varianza para detectar regiones uniformes
            variance = np.var(face_region)
            if variance < min_variance:
                return False, "Región muy uniforme"
            
            # Verificar que tenga suficiente detalle
            edges = cv2.Canny(face_region, 50, 150)
            edge_density = np.sum(edges > 0) / (face_region.shape[0] * face_region.shape[1])
            
            if edge_density < 0.02:
                return False, "Pocos detalles faciales"
            
            # Verificar proporciones típicas de una cara
            height, width = face_region.shape
            aspect_ratio = width / height
            if not (0.7 <= aspect_ratio <= 1.4): 
                return False, f"Proporción incorrecta: {aspect_ratio:.2f}"
            
            return True, "Cara válida"
            
        except Exception as e:
            return False, f"Error en validación: {e}"
        
    def generate_face_variations(self, face):
        """Genera variaciones de una cara para mejorar el entrenamiento"""
        variations = [face]  # Imagen original
        
        try:
            # Rotación ligera
            angles = [-5, 5]
            for angle in angles:
                center = (face.shape[1]//2, face.shape[0]//2)
                rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
                rotated = cv2.warpAffine(face, rotation_matrix, (face.shape[1], face.shape[0]))
                variations.append(rotated)
            
            # Ajuste de brillo
            brightness_values = [0.9, 1.1]
            for brightness in brightness_values:
                bright_face = cv2.convertScaleAbs(face, alpha=brightness, beta=0)
                variations.append(bright_face)
            
            # Filtro gaussiano suave
            blurred = cv2.GaussianBlur(face, (3, 3), 0)
            variations.append(blurred)
            
        except Exception as e:
            print(f"Error generando variaciones: {e}")
        
        return variations
        
    def train_from_images(self, image_paths, labels):
        faces = []
        label_ids = []
        label_map = {}
        next_id = 0
        
        print(f"Entrenando con {len(image_paths)} imágenes para {len(set(labels))} usuarios")
        
        # Crear mapeo único de etiquetas
        unique_labels = list(set(labels))
        for label in unique_labels:
            label_map[label] = next_id
            next_id += 1
        
        successful_faces = 0
        processed_images = 0
        
        for i, img_path in enumerate(image_paths):
            try:
                print(f"Procesando imagen {i+1}/{len(image_paths)}: {os.path.basename(img_path)}")
                
                img = cv2.imread(img_path)
                if img is None:
                    print(f"No se pudo cargar: {img_path}")
                    continue
                    
                processed_images += 1
                enhanced_img = self.enhance_image(img)
                
                face_rects = self.detect_faces_multiple_scales(enhanced_img)
                
                if len(face_rects) > 0:
                    # Tomar la cara más grande si hay múltiples
                    areas = [w * h for (x, y, w, h) in face_rects]
                    max_area_idx = np.argmax(areas)
                    (x, y, w, h) = face_rects[max_area_idx]
                    
                    # Extraer región de la cara
                    margin = max(15, min(w, h) // 4)
                    x_start = max(0, x - margin)
                    y_start = max(0, y - margin)
                    x_end = min(enhanced_img.shape[1], x + w + margin)
                    y_end = min(enhanced_img.shape[0], y + h + margin)
                    
                    face = enhanced_img[y_start:y_end, x_start:x_end]
                    
                    # Verificar tamaño mínimo
                    if face.shape[0] < 40 or face.shape[1] < 40:
                        print(f"Cara muy pequeña en {img_path}")
                        continue
                    
                    # Preprocesar la cara
                    face = self.preprocess_face(face)
                    
                    face_variations = self.generate_face_variations(face)
                    
                    for variation in face_variations:
                        faces.append(variation)
                        label_ids.append(label_map[labels[i]])
                        successful_faces += 1
                    
                    print(f"Cara extraída exitosamente para {labels[i]} ({len(face_variations)} variaciones)")
                else:
                    print(f"No se detectó cara en {img_path}")
                    
            except Exception as e:
                print(f"Error procesando imagen {img_path}: {e}")
        
        # Guardar mapeos
        self.known_face_names = unique_labels
        self.label_to_id = label_map
        self.id_to_label = {v: k for k, v in label_map.items()}
        
        print(f"\nResultado del entrenamiento:")
        print(f"- Imágenes procesadas: {processed_images}")
        print(f"- Caras extraídas (con variaciones): {successful_faces}")
        print(f"- Personas únicas: {len(unique_labels)}")
        
        if len(faces) >= 2:
            print(f"Entrenando modelo LBPH con {len(faces)} muestras...")
            
            # Convertir a numpy arrays
            faces_array = np.array(faces)
            labels_array = np.array(label_ids)
            
            # Entrenar el modelo
            self.face_recognizer.train(faces_array, labels_array)
            self.has_trained_model = True
            print("Entrenamiento completado exitosamente")
            
            # Mostrar distribución
            for person in unique_labels:
                original_count = labels.count(person)
                total_samples = label_ids.count(label_map[person])
                print(f"  - {person}: {original_count} imágenes → {total_samples} muestras")
        else:
            print("No se encontraron suficientes caras para entrenar")
            
        return successful_faces
    
    def recognize_face(self, frame, timestamp=None, measure_facial_features=False):
        if timestamp is None:
            timestamp = datetime.now()
        
        # Mejorar imagen de entrada
        enhanced_frame = self.enhance_image(frame)
        
        # Detectar caras con parámetros más conservadores
        face_rects = self.detect_faces_multiple_scales(enhanced_frame)
        
        validated_faces = []
        for (x, y, w, h) in face_rects:
            # Extraer región para validación
            face_region = enhanced_frame[y:y+h, x:x+w]
            is_valid, reason = self.validate_face_quality(face_region)
            
            if is_valid:
                validated_faces.append((x, y, w, h))
            else:
                print(f"Cara rechazada en ({x},{y}): {reason}")
        
        face_rects = validated_faces
        
        face_names = []
        face_locations = []
        confidence_scores = []
        face_measures = []
        
        print(f"Detectadas y validadas {len(face_rects)} caras en el frame")
        
        for i, (x, y, w, h) in enumerate(face_rects):
            # Convertir a formato (top, right, bottom, left)
            face_location = (y, x+w, y+h, x)
            face_locations.append(face_location)
            
            # Extraer región de la cara con margen
            margin = max(5, min(w, h) // 10)
            x_start = max(0, x - margin)
            y_start = max(0, y - margin)
            x_end = min(enhanced_frame.shape[1], x + w + margin)
            y_end = min(enhanced_frame.shape[0], y + h + margin)
            
            face_roi = enhanced_frame[y_start:y_end, x_start:x_end]
            
            # Verificar tamaño mínimo
            if face_roi.shape[0] < 40 or face_roi.shape[1] < 40:
                face_names.append("Cara muy pequeña")
                confidence_scores.append(0)
                face_measures.append(None)
                continue
            
            # Preprocesar la cara
            face_roi = self.preprocess_face(face_roi)
            
            name = "Desconocido"
            confidence = 0
            
            # Intentar reconocer
            if self.has_trained_model:
                try:
                    label_id, distance = self.face_recognizer.predict(face_roi)
                    
                    print(f"Reconocimiento - ID: {label_id}, Distancia: {distance:.2f}")
                    
                    # Umbrales ajustados
                    if distance <= 120:  # Más estricto que antes (era 150)
                        if label_id in self.id_to_label:
                            name = self.id_to_label[label_id]
                            # Calcular confianza
                            max_distance = 200.0
                            confidence = max(0, min(100, ((max_distance - distance) / max_distance) * 100))
                            print(f"✓ Reconocido como: {name} (distancia: {distance:.2f}, confianza: {confidence:.1f}%)")
                            
                            self.last_detected_faces[name] = timestamp
                        else:
                            print(f"ID {label_id} no encontrado en mapeo")
                    else:
                        print(f"✗ Distancia muy alta: {distance:.2f} > 120")
                        
                except Exception as e:
                    print(f"Error en reconocimiento: {e}")
            else:
                print("Modelo no entrenado")
            
            face_names.append(name)
            confidence_scores.append(confidence)
            face_measures.append(None)
        
        return face_locations, face_names, confidence_scores, face_measures
    
    def set_confidence_threshold(self, threshold_percent):
        # Convertir porcentaje a distancia
        max_distance = 200.0  
        distance_threshold = max_distance * (1 - threshold_percent/100)
        self.confidence_threshold = distance_threshold
        print(f"Umbral establecido: {threshold_percent}% (distancia LBPH: {distance_threshold:.1f})")
        
    def save_model(self, filename):
        """Guarda el modelo entrenado"""
        if not self.has_trained_model:
            print("No hay modelo entrenado para guardar")
            return False
            
        try:
            # Guardar el modelo LBPH
            model_file = filename + "_improved_model.xml"
            self.face_recognizer.write(model_file)
            
            # Guardar mapeo de etiquetas y otros datos
            data = {
                'id_to_label': self.id_to_label,
                'label_to_id': self.label_to_id,
                'names': self.known_face_names,
                'confidence_threshold': self.confidence_threshold
            }
            
            data_file = filename + "_improved_data.pkl"
            with open(data_file, 'wb') as f:
                pickle.dump(data, f)
                
            print(f"Modelo guardado en {model_file} y {data_file}")
            return True
        except Exception as e:
            print(f"Error al guardar modelo: {e}")
            return False
        
    def load_model(self, filename):
        """Carga el modelo desde archivo"""
        try:
            # Intentar cargar modelo primero
            model_file = filename + "_improved_model.xml"
            data_file = filename + "_improved_data.pkl"
            
            # Si no existe, intentar cargar modelo anterior
            if not os.path.exists(model_file) or not os.path.exists(data_file):
                model_file = filename + "_model.xml"
                data_file = filename + "_data.pkl"
                
                if not os.path.exists(model_file) or not os.path.exists(data_file):
                    print(f"Archivos del modelo no encontrados")
                    return False
            
            # Cargar el modelo LBPH
            self.face_recognizer.read(model_file)
            
            # Cargar mapeo de etiquetas y otros datos
            with open(data_file, 'rb') as f:
                data = pickle.load(f)
                self.id_to_label = data['id_to_label']
                self.label_to_id = data['label_to_id']
                self.known_face_names = data['names']
                if 'confidence_threshold' in data:
                    self.confidence_threshold = data['confidence_threshold']
            
            self.has_trained_model = True
            print(f"Modelo cargado exitosamente: {len(self.known_face_names)} usuarios")
            for name in self.known_face_names:
                print(f"  - {name}")
            return True
        except Exception as e:
            print(f"Error al cargar modelo: {e}")
            return False

class FacialAccessControlGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Sistema Avanzado de Control de Acceso Facial - VERSIÓN MEJORADA")
        self.root.geometry("1200x800")
        
        # Lista de eventos recientes
        self.recent_events = []
        self.max_events = 15
        
        # Configurar estilo visual moderno
        self.style = ttk.Style()
        self.style.theme_use('clam')
        
        # Crear directorios
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.data_dir = os.path.join(self.base_dir, "face_data")
        self.log_dir = os.path.join(self.base_dir, "access_logs")
        self.model_file = os.path.join(self.data_dir, "face_model")
        
        for directory in [self.data_dir, self.log_dir]:
            if not os.path.exists(directory):
                os.makedirs(directory)
        
        # Inicializar reconocedor facial
        self.face_recognizer = ImprovedFacialRecognizer()
        
        # Configuración de la cámara
        self.camera = None
        self.is_camera_running = False
        self.processing_frame = False
        self.camera_thread = None
        self.frame_rate = 0
        self.last_frame_time = time.time()
        self.fps_update_interval = 0.5
        self.last_fps_update = time.time()
        
        # Variables de control
        self.verification_mode = False
        self.measure_faces = False  
        self.current_frame = None
        self.current_frame_rgb = None
        
        # Cargar modelo si existe
        model_loaded = False
        if os.path.exists(self.model_file + "_improved_model.xml") and os.path.exists(self.model_file + "_improved_data.pkl"):
            success = self.face_recognizer.load_model(self.model_file)
            if success:
                self.log_event("Modelo cargado exitosamente")
                model_loaded = True
        
        # Si no se cargó el mejorado, intentar el anterior
        if not model_loaded and os.path.exists(self.model_file + "_model.xml") and os.path.exists(self.model_file + "_data.pkl"):
            success = self.face_recognizer.load_model(self.model_file)
            if success:
                self.log_event("Modelo anterior cargado - Se recomienda volver a entrenar")
                model_loaded = True
        
        if not model_loaded:
            self.log_event("No se encontró modelo preentrenado - Entrenar con imágenes")
            
        # Crear la interfaz gráfica
        self.create_widgets()
        
        # Temporizador para actualizar estado
        self.root.after(1000, self.update_status)
    
    def log_event(self, message, event_type='INFO', user=None):
        event = {
            'time': datetime.now(),
            'message': message,
            'type': event_type,
            'user': user
        }
        
        self.recent_events.append(event)
        
        if len(self.recent_events) > self.max_events:
            self.recent_events.pop(0)
        
        # Solo actualizar la interfaz si ya existe
        if hasattr(self, 'events_text'):
            self.update_event_display()
            
        print(f"[{event_type}] {message}")
    
    def create_widgets(self):
        # Panel principal con grid
        self.main_frame = ttk.Frame(self.root, padding="10")
        self.main_frame.grid(row=0, column=0, sticky="nsew")
        
        # Configurar expansión de filas y columnas
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        self.main_frame.columnconfigure(0, weight=3)  # Cámara
        self.main_frame.columnconfigure(1, weight=1)  # Controles
        self.main_frame.rowconfigure(0, weight=1)
        
        # Panel de la cámara
        self.camera_frame = ttk.LabelFrame(self.main_frame, text="Visualización de Cámara - VERSIÓN MEJORADA", padding="5")
        self.camera_frame.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        
        self.camera_view = ttk.Label(self.camera_frame)
        self.camera_view.pack(fill=tk.BOTH, expand=True)
        
        # Panel de información y controles
        self.control_frame = ttk.Frame(self.main_frame)
        self.control_frame.grid(row=0, column=1, sticky="nsew", padx=5, pady=5)
        
        # Panel de controles
        self.buttons_frame = ttk.LabelFrame(self.control_frame, text="Controles", padding="5")
        self.buttons_frame.pack(fill=tk.X, expand=False, pady=5)
        
        # Botones principales
        self.btn_start_camera = ttk.Button(
            self.buttons_frame, text="Iniciar Cámara", 
            command=self.toggle_camera)
        self.btn_start_camera.pack(fill=tk.X, pady=3)
        
        self.btn_verify = ttk.Button(
            self.buttons_frame, text="Verificar Acceso", 
            command=self.toggle_verification_mode)
        self.btn_verify.pack(fill=tk.X, pady=3)
        
        # Botón para entrenar con nuevas imágenes
        self.btn_train = ttk.Button(
            self.buttons_frame, text="Entrenar Modelo", 
            command=self.train_with_folder)
        self.btn_train.pack(fill=tk.X, pady=3)
        
        # Botón para verificar base de datos
        self.btn_check_db = ttk.Button(
            self.buttons_frame, text="Verificar Base de Datos", 
            command=self.check_database)
        self.btn_check_db.pack(fill=tk.X, pady=3)
        
        self.btn_exit = ttk.Button(
            self.buttons_frame, text="Salir", 
            command=self.exit_application)
        self.btn_exit.pack(fill=tk.X, pady=3)
        
        # Opciones de configuración
        self.options_frame = ttk.LabelFrame(self.control_frame, text="Configuración Mejorada", padding="5")
        self.options_frame.pack(fill=tk.X, expand=False, pady=5)
        
        # Control deslizante para umbral de confianza
        ttk.Label(self.options_frame, text="Umbral de Confianza (%):").pack(anchor="w", pady=(5,0))
        self.threshold_var = tk.DoubleVar(value=40)  # Valor más permisivo
        self.threshold_slider = ttk.Scale(
            self.options_frame, from_=20, to_=70, 
            variable=self.threshold_var, 
            command=self.update_threshold)
        self.threshold_slider.pack(fill=tk.X)
        
        self.threshold_label = ttk.Label(self.options_frame, text="40%")
        self.threshold_label.pack(anchor="w")
        
        # Botón para aplicar cambios
        ttk.Button(
            self.options_frame, text="Aplicar Configuración", 
            command=self.apply_settings).pack(fill=tk.X, pady=5)
        
        # Panel de estado
        self.status_frame = ttk.LabelFrame(self.control_frame, text="Estado del Sistema Mejorado", padding="5")
        self.status_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Información del modelo
        ttk.Label(self.status_frame, text="Modelo:").pack(anchor="w")
        self.model_status_label = ttk.Label(self.status_frame, text="No entrenado")
        self.model_status_label.pack(anchor="w")
        
        # Información técnica
        ttk.Label(self.status_frame, text="Rendimiento:").pack(anchor="w", pady=(5,0))
        self.perf_label = ttk.Label(self.status_frame, text="FPS: --")
        self.perf_label.pack(anchor="w")
        
        # Estado actual
        ttk.Label(self.status_frame, text="Estado:").pack(anchor="w", pady=(5,0))
        self.status_label = ttk.Label(self.status_frame, text="Inactivo")
        self.status_label.pack(anchor="w")
        
        # Lista de eventos recientes
        ttk.Label(self.status_frame, text="Eventos Recientes:").pack(anchor="w", pady=(5,0))
        
        # Frame con scrollbar para eventos
        events_frame = ttk.Frame(self.status_frame)
        events_frame.pack(fill=tk.BOTH, expand=True)
        
        self.events_text = tk.Text(events_frame, height=10, width=35, wrap="word")
        scrollbar = ttk.Scrollbar(events_frame, orient="vertical", command=self.events_text.yview)
        self.events_text.configure(yscrollcommand=scrollbar.set)
        
        self.events_text.pack(side="left", fill=tk.BOTH, expand=True)
        scrollbar.pack(side="right", fill="y")
        
        self.events_text.config(state="disabled")
        
        # Configurar tags para colores en eventos
        self.events_text.tag_configure("access", foreground="green")
        self.events_text.tag_configure("error", foreground="red")
        self.events_text.tag_configure("info", foreground="blue")
        
        # Actualizar estado del modelo
        self.update_model_status()
        
        # Actualizar eventos si ya existen algunos
        if self.recent_events:
            self.update_event_display()
    
    def check_database(self):
        """Verifica la estructura de la base de datos de imágenes"""
        base_datos_path = os.path.join(self.base_dir, "base_Datos")
        
        if not os.path.exists(base_datos_path):
            messagebox.showerror("Error", f"No se encuentra la carpeta 'base_Datos' en {self.base_dir}")
            return
        
        # Analizar estructura
        personas = []
        total_imagenes = 0
        
        for item in os.listdir(base_datos_path):
            item_path = os.path.join(base_datos_path, item)
            if os.path.isdir(item_path):
                imagenes = [f for f in os.listdir(item_path) 
                           if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
                personas.append((item, len(imagenes)))
                total_imagenes += len(imagenes)
        
        # Crear reporte
        report = f"REPORTE DE BASE DE DATOS - VERSIÓN MEJORADA\n"
        report += f"="*40 + "\n\n"
        report += f"Ubicación: {base_datos_path}\n"
        report += f"Personas encontradas: {len(personas)}\n"
        report += f"Total de imágenes: {total_imagenes}\n\n"
        
        if personas:
            report += "Detalle por persona:\n"
            for nombre, count in personas:
                if count >= 10:
                    status = "✓ Excelente"
                elif count >= 5:
                    status = "✓ Bueno"
                else:
                    status = "Pocas imágenes"
                report += f"  • {nombre}: {count} imágenes {status}\n"
                
            # Verificar si hay suficientes imágenes
            report += "\nEVALUACIÓN PARA MODELO:\n"
            personas_excelentes = [p for p, c in personas if c >= 10]
            personas_buenas = [p for p, c in personas if 5 <= c < 10]
            personas_pocas = [p for p, c in personas if c < 5]
            
            if personas_excelentes:
                report += f"✓ Personas con excelente cantidad (≥10): {len(personas_excelentes)}\n"
            if personas_buenas:
                report += f"✓ Personas con buena cantidad (5-9): {len(personas_buenas)}\n"
            if personas_pocas:
                report += f"⚠ Personas con pocas imágenes (<5): {len(personas_pocas)}\n"
                report += "  Recomendación: Agregar más imágenes para mejor precisión\n"
                
            # Estimación de muestras totales con variaciones
            estimated_samples = sum(count * 6 for _, count in personas)  # 6 variaciones por imagen
            report += f"\nMuestras estimadas con variaciones: {estimated_samples}\n"
            
        else:
            report += "No se encontraron carpetas de personas\n"
        
        # Mostrar reporte
        messagebox.showinfo("Estado de la Base de Datos", report)
        self.log_event(f"Verificación BD: {len(personas)} personas, {total_imagenes} imágenes")
    
    def update_model_status(self):
        """Actualiza el estado del modelo en la interfaz"""
        if self.face_recognizer.has_trained_model:
            names = ", ".join(self.face_recognizer.known_face_names[:3]) 
            if len(self.face_recognizer.known_face_names) > 3:
                names += f" y +{len(self.face_recognizer.known_face_names) - 3} más"
            status_text = f"Modelo Entrenado\nPersonas: {len(self.face_recognizer.known_face_names)}\nUsuarios: {names}"
        else:
            status_text = "No entrenado - Usar 'Entrenar Modelo'"
        
        self.model_status_label.config(text=status_text)
    
    def toggle_camera(self):
        if self.is_camera_running:
            self.stop_camera()
            self.btn_start_camera.config(text="Iniciar Cámara")
        else:
            if self.start_camera():
                self.btn_start_camera.config(text="Detener Cámara")

    def start_camera(self):
        try:
            self.camera = cv2.VideoCapture(0)
            if not self.camera.isOpened():
                messagebox.showerror("Error", "No se pudo abrir la cámara")
                return False
            
            # Configurar resolución
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            
            self.is_camera_running = True
            self.log_event("Cámara iniciada con procesamiento mejorado")
            
            # Iniciar thread de captura
            self.camera_thread = threading.Thread(target=self.camera_loop)
            self.camera_thread.daemon = True
            self.camera_thread.start()
            
            return True
        except Exception as e:
            messagebox.showerror("Error", f"Error al iniciar cámara: {e}")
            return False
    
    def stop_camera(self):
        if self.is_camera_running:
            self.is_camera_running = False
            if self.camera:
                self.camera.release()
                self.camera = None
            
            if self.camera_thread:
                self.camera_thread.join(timeout=1.0)
                self.camera_thread = None
            
            self.camera_view.config(image='')
            self.log_event("Cámara detenida")
    
    def camera_loop(self):
        """Thread principal para captura y procesamiento de cámara"""
        frame_count = 0
        start_time = time.time()
        
        while self.is_camera_running and self.camera:
            try:
                ret, frame = self.camera.read()
                if not ret:
                    self.log_event("Error: No se pudo leer frame de cámara")
                    break
                
                # Actualizar contador FPS
                frame_count += 1
                current_time = time.time()
                
                elapsed = current_time - start_time
                if elapsed >= 1.0:
                    self.frame_rate = frame_count / elapsed
                    frame_count = 0
                    start_time = current_time
                
                # Almacenar el frame actual
                self.current_frame = frame.copy()
                
                # Procesar frame si estamos en modo verificación
                if self.verification_mode:
                    self.process_and_display_frame(frame.copy())
                else:
                    self.show_frame(frame)
                
                # Control de velocidad
                time.sleep(0.05)  # ~20 FPS
                    
            except Exception as e:
                self.log_event(f"Error en bucle de cámara: {e}")
        
        self.log_event("Bucle de cámara finalizado")

    def process_and_display_frame(self, frame):
        """Procesa un frame con detección mejorada para evitar múltiples caras falsas"""
        try:
            # Realizar reconocimiento facial mejorado
            result = self.face_recognizer.recognize_face(frame)
            face_locations, face_names, confidences, measures = result
            
            # FILTRO ADICIONAL: Si hay muchas detecciones, tomar solo la más confiable
            if len(face_locations) > 1:
                # Encontrar la cara con mayor confianza
                max_confidence_idx = np.argmax(confidences)
                
                # Si la diferencia de confianza es significativa, tomar solo la mejor
                max_conf = confidences[max_confidence_idx]
                if max_conf > 20:  # Solo si tiene confianza razonable
                    face_locations = [face_locations[max_confidence_idx]]
                    face_names = [face_names[max_confidence_idx]]
                    confidences = [confidences[max_confidence_idx]]
                    print(f"Múltiples detecciones filtradas. Seleccionada la de mayor confianza: {max_conf:.1f}%")
            
            print(f"Procesando {len(face_locations)} cara(s) después del filtrado")
            
            # Crear copia del frame para dibujar
            display_frame = frame.copy()
            
            # Si no hay caras válidas, mostrar mensaje
            if len(face_locations) == 0:
                cv2.putText(display_frame, "Buscando caras...", (50, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            
            # Dibujar resultados
            for i, ((top, right, bottom, left), name, confidence) in enumerate(zip(face_locations, face_names, confidences)):
                
                print(f"Cara detectada: {name}, Confianza: {confidence:.1f}%")
                
                # Determinar color y estado
                if name != "Desconocido" and confidence >= 60:
                    color = (0, 255, 0)  # Verde brillante
                    thickness = 3
                    status = "ACCESO AUTORIZADO"
                    status_color = (0, 255, 0)
                elif name != "Desconocido" and confidence >= 40:
                    color = (0, 255, 255)  # Amarillo
                    thickness = 2
                    status = "VERIFICANDO..."
                    status_color = (0, 255, 255)
                elif name != "Desconocido" and confidence >= 20:
                    color = (0, 165, 255)  # Naranja
                    thickness = 2
                    status = "CONFIANZA BAJA"
                    status_color = (0, 165, 255)
                else:
                    color = (0, 0, 255)  # Rojo
                    thickness = 2
                    status = "ACCESO DENEGADO"
                    status_color = (0, 0, 255)
                
                # Dibujar rectángulo principal
                cv2.rectangle(display_frame, (left, top), (right, bottom), color, thickness)
                
                # Dibujar esquinas decorativas
                corner_length = 25
                corner_thickness = thickness + 1
                
                # Esquinas superiores
                cv2.line(display_frame, (left, top), (left + corner_length, top), color, corner_thickness)
                cv2.line(display_frame, (left, top), (left, top + corner_length), color, corner_thickness)
                cv2.line(display_frame, (right, top), (right - corner_length, top), color, corner_thickness)
                cv2.line(display_frame, (right, top), (right, top + corner_length), color, corner_thickness)
                
                # Esquinas inferiores
                cv2.line(display_frame, (left, bottom), (left + corner_length, bottom), color, corner_thickness)
                cv2.line(display_frame, (left, bottom), (left, bottom - corner_length), color, corner_thickness)
                cv2.line(display_frame, (right, bottom), (right - corner_length, bottom), color, corner_thickness)
                cv2.line(display_frame, (right, bottom), (right, bottom - corner_length), color, corner_thickness)
                
                # Texto del nombre
                if name != "Desconocido":
                    label = f"{name}"
                    confidence_text = f"{confidence:.1f}%"
                else:
                    label = "Usuario Desconocido"
                    confidence_text = "---"
                
                # Fondo para el nombre
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, 0.8, 2)[0]
                cv2.rectangle(display_frame, (left, bottom - 40), (left + label_size[0] + 20, bottom), color, cv2.FILLED)
                cv2.putText(display_frame, label, (left + 10, bottom - 12), 
                        cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 2)
                
                # Texto de confianza
                conf_size = cv2.getTextSize(confidence_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)[0]
                cv2.rectangle(display_frame, (left, bottom - 70), (left + conf_size[0] + 15, bottom - 40), (0, 0, 0), cv2.FILLED)
                cv2.putText(display_frame, confidence_text, (left + 7, bottom - 48), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                
                # Estado arriba
                status_size = cv2.getTextSize(status, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                cv2.rectangle(display_frame, (left, top - 35), (left + status_size[0] + 20, top), status_color, cv2.FILLED)
                cv2.putText(display_frame, status, (left + 10, top - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # Registrar acceso
                if name != "Desconocido" and confidence >= 40:
                    self.register_access(name, confidence)
            
            # Mostrar frame procesado
            self.show_frame(display_frame)
            
        except Exception as e:
            print(f"Error en procesamiento: {e}")
            import traceback
            traceback.print_exc()
            self.log_event(f"Error en procesamiento: {e}")
            self.show_frame(frame)
    
    def show_frame(self, frame):
        try:
            # Verificar que el frame no esté vacío
            if frame is None or frame.size == 0:
                print("Frame vacío recibido")
                return
                
            # Convertir a formato para Tkinter
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb_frame)
            
            # Obtener dimensiones del contenedor
            frame_width = self.camera_frame.winfo_width()
            frame_height = self.camera_frame.winfo_height()
            
            if frame_width > 10 and frame_height > 10:
                # Calcular proporción de aspecto
                img_width, img_height = img.size
                img_aspect = img_width / img_height
                container_aspect = frame_width / frame_height
                
                # Redimensionar manteniendo proporción
                if container_aspect > img_aspect:
                    new_height = frame_height - 20  # Margen
                    new_width = int(new_height * img_aspect)
                else:
                    new_width = frame_width - 20   # Margen
                    new_height = int(new_width / img_aspect)
                    
                img = img.resize((new_width, new_height), Image.LANCZOS)
                
            img_tk = ImageTk.PhotoImage(image=img)
            self.root.after(0, lambda: self._update_camera_view(img_tk))
            
        except Exception as e:
            print(f"Error detallado al mostrar frame: {e}")
            import traceback
            traceback.print_exc()
    
    def _update_camera_view(self, img_tk):
        """Actualiza la vista de cámara desde el thread principal"""
        self.camera_view.config(image=img_tk)
        self.camera_view.image = img_tk
    
    def toggle_verification_mode(self):
        """Activa/desactiva el modo de verificación continua"""
        if not self.face_recognizer.has_trained_model:
            messagebox.showwarning("Advertencia", 
                                 "Primero debe entrenar el modelo.\n\n" +
                                 "Use el botón 'Entrenar Modelo' con su carpeta base_Datos.")
            return
            
        self.verification_mode = not self.verification_mode
        
        if self.verification_mode:
            self.btn_verify.config(text="Detener Verificación")
            if not self.is_camera_running:
                self.toggle_camera()
            self.log_event("Modo verificación mejorado activado")
            self.status_label.config(text="✓ Verificando accesos...")
        else:
            self.btn_verify.config(text="Verificar Acceso")
            self.log_event("Modo verificación desactivado")
            self.status_label.config(text="Inactivo")
    
    def register_access(self, name, confidence):
        """Registra un acceso exitoso"""
        current_time = datetime.now()
        
        # Evitar registros duplicados cercanos en el tiempo
        for event in self.recent_events:
            if event['type'] == 'ACCESS' and event['user'] == name:
                if current_time - event['time'] < timedelta(seconds=5):  # Aumentado a 5 segundos
                    return
        
        # Registrar acceso en log
        timestamp = current_time.strftime("%Y-%m-%d %H:%M:%S")
        log_file = os.path.join(self.log_dir, f"accesos_{current_time.strftime('%Y%m%d')}.log")
        
        with open(log_file, 'a') as f:
            f.write(f"{timestamp},{name},{confidence:.2f},ACCESO_MEJORADO\n")
        
        # Notificar en la interfaz
        access_message = f"✓ ACCESO: {name} ({confidence:.1f}%)"
        self.log_event(access_message, event_type='ACCESS', user=name)
    
    def train_with_folder(self):
        """Entrena el reconocedor con imágenes de la carpeta base_Datos"""
        try:
            folder_path = os.path.join(self.base_dir, "base_Datos")
            
            if not os.path.isdir(folder_path):
                messagebox.showerror("Error", 
                                   f"No se encontró la carpeta 'base_Datos' en {self.base_dir}\n\n" +
                                   "Estructura requerida:\n" +
                                   "base_Datos/\n" +
                                   "├── Persona1/\n" +
                                   "│   ├── foto1.jpg\n" +
                                   "│   └── foto2.jpg\n" +
                                   "├── Persona2/\n" +
                                   "│   └── fotos...\n")
                return
            
            # Recopilar imágenes y etiquetas
            image_paths = []
            labels = []
            
            # Buscar en subcarpetas
            for person_name in os.listdir(folder_path):
                person_dir = os.path.join(folder_path, person_name)
                if os.path.isdir(person_dir):
                    for image_file in os.listdir(person_dir):
                        if image_file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                            image_path = os.path.join(person_dir, image_file)
                            image_paths.append(image_path)
                            labels.append(person_name)
            
            if len(image_paths) == 0:
                messagebox.showwarning("Advertencia", 
                                     "No se encontraron imágenes válidas en la carpeta base_Datos\n\n" +
                                     "Formatos soportados: JPG, JPEG, PNG, BMP")
                return
            
            # Informar al usuario sobre las mejoras
            unique_people = len(set(labels))
            estimated_samples = len(image_paths) * 6  # 6 variaciones por imagen
            
            info_message = f"ENTRENAMIENTO DEL MODELO\n\n" + \
                          f"Estadísticas:\n" + \
                          f"• Imágenes encontradas: {len(image_paths)}\n" + \
                          f"• Personas únicas: {unique_people}\n" + \
                          f"• Muestras estimadas (con variaciones): {estimated_samples}\n\n" + \
                          f"¿Continuar con el entrenamiento?"
            
            result = messagebox.askyesno("Entrenamiento Mejorado", info_message)
            if not result:
                return
            
            # Mostrar progreso mejorado
            progress_window = tk.Toplevel(self.root)
            progress_window.title("Entrenando Modelo...")
            progress_window.geometry("500x200")
            progress_window.transient(self.root)
            progress_window.grab_set()
            
            # Centrar ventana
            progress_window.update_idletasks()
            x = (progress_window.winfo_screenwidth() // 2) - (500 // 2)
            y = (progress_window.winfo_screenheight() // 2) - (200 // 2)
            progress_window.geometry(f"+{x}+{y}")
            
            ttk.Label(progress_window, text="Entrenando modelo de reconocimiento facial mejorado...", 
                     font=('Arial', 12, 'bold')).pack(pady=20)
            
            progress_bar = ttk.Progressbar(progress_window, mode='indeterminate')
            progress_bar.pack(pady=10, padx=20, fill='x')
            progress_bar.start()
            
            status_label = ttk.Label(progress_window, text="Inicializando procesamiento avanzado...")
            status_label.pack(pady=10)
            
            progress_window.update()
            
            # Entrenar en thread separado
            def train_thread():
                try:
                    self.log_event(f"🚀 Iniciando entrenamiento mejorado: {len(image_paths)} imágenes...")
                    
                    # Actualizar estado
                    progress_window.after(0, lambda: status_label.config(text="Aplicando mejoras de imagen..."))
                    
                    num_faces = self.face_recognizer.train_from_images(image_paths, labels)
                    
                    # Cerrar ventana de progreso
                    progress_window.after(0, progress_window.destroy)
                    
                    if num_faces > 0:
                        # Guardar el modelo
                        if self.face_recognizer.save_model(self.model_file):
                            success_message = f"ENTRENAMIENTO COMPLETADO EXITOSAMENTE\n\n" + \
                                            f"Resultados:\n" + \
                                            f"• Muestras procesadas: {num_faces}\n" + \
                                            f"• Personas entrenadas: {unique_people}\n" + \
                                            f"• Variaciones generadas automáticamente\n" + \
                                            f"• Modelo guardado\n\n" + \
                                            f"El sistema está listo para reconocimiento de alta precisión!"
                            
                            self.root.after(0, lambda: messagebox.showinfo("Entrenamiento Completado", success_message))
                            self.log_event(f"Entrenamiento exitoso: {num_faces} muestras, {unique_people} personas")
                        else:
                            self.root.after(0, lambda: messagebox.showerror("Error", 
                                                          "Entrenamiento completado pero no se pudo guardar el modelo"))
                    else:
                        error_message = f"PROBLEMA EN EL ENTRENAMIENTO\n\n" + \
                                      f"No se detectaron caras en las imágenes.\n\n" + \
                                      f"Recomendaciones:\n" + \
                                      f"• Verificar que las imágenes contengan caras visibles\n" + \
                                      f"• Usar imágenes con buena iluminación\n" + \
                                      f"• Asegurar que las caras sean de tamaño adecuado\n" + \
                                      f"• Usar formatos JPG, PNG, BMP"
                        
                        self.root.after(0, lambda: messagebox.showwarning("Advertencia", error_message))
                    
                    # Actualizar estado del modelo en UI
                    self.root.after(0, self.update_model_status)
                    
                except Exception as e:
                    progress_window.after(0, progress_window.destroy)
                    error_msg = f"ERROR EN EL ENTRENAMIENTO\n\n{str(e)}\n\n" + \
                               f"Posibles causas:\n" + \
                               f"• Imágenes corruptas\n" + \
                               f"• Falta de permisos de escritura\n" + \
                               f"• Problemas con OpenCV"
                    self.root.after(0, lambda: messagebox.showerror("Error", error_msg))
                    self.log_event(f"Error de entrenamiento: {e}", event_type='ERROR')
            
            # Iniciar entrenamiento
            train_thread_obj = threading.Thread(target=train_thread)
            train_thread_obj.daemon = True
            train_thread_obj.start()
                
        except Exception as e:
            messagebox.showerror("Error", f"Error durante el entrenamiento: {e}")
            self.log_event(f"Error de entrenamiento: {e}", event_type='ERROR')

    def update_threshold(self, value):
        """Actualiza el valor de umbral mostrado"""
        threshold_value = float(value)
        self.threshold_label.config(text=f"{threshold_value:.0f}%")
    
    def apply_settings(self):
        """Aplica la configuración actual"""
        new_threshold = self.threshold_var.get()
        self.face_recognizer.set_confidence_threshold(new_threshold)
        self.log_event(f"⚙️ Configuración actualizada: Umbral={new_threshold:.1f}%")
        messagebox.showinfo("Configuración", 
                           f"Configuración aplicada\n\n" +
                           f"Umbral de confianza: {new_threshold:.1f}%\n" +
                           f"Algoritmo: LBPH Mejorado")
    
    def update_event_display(self):
        """Actualiza la visualización de eventos recientes"""
        self.events_text.config(state="normal")
        self.events_text.delete(1.0, tk.END)
        
        for event in self.recent_events:
            time_str = event['time'].strftime("%H:%M:%S")
            
            if event['type'] == 'ACCESS':
                self.events_text.insert(tk.END, f"{time_str} {event['message']}\n", "access")
            elif event['type'] == 'ERROR':
                self.events_text.insert(tk.END, f"{time_str} {event['message']}\n", "error")
            else:
                self.events_text.insert(tk.END, f"{time_str} {event['message']}\n", "info")
        
        self.events_text.config(state="disabled")
        # Auto-scroll al final
        self.events_text.see(tk.END)
    
    def update_status(self):
        """Actualiza información de estado periódicamente"""
        current_time = time.time()
        
        # Actualizar FPS
        if current_time - self.last_fps_update >= self.fps_update_interval:
            self.perf_label.config(text=f"FPS: {self.frame_rate:.1f} | Mejorado: ✓")
            self.last_fps_update = current_time
        
        # Actualizar estado general
        if self.verification_mode:
            if self.face_recognizer.has_trained_model:
                self.status_label.config(text="🔍 Verificando con IA mejorada...")
            else:
                self.status_label.config(text="⚠ Modelo no entrenado")
        elif self.is_camera_running:
            self.status_label.config(text="📹 Cámara activa")
        else:
            self.status_label.config(text="💤 Sistema inactivo")
        
        # Programar próxima actualización
        self.root.after(500, self.update_status)
    
    def exit_application(self):
        """Cierra la aplicación"""
        if messagebox.askyesno("Confirmar salida", 
                              "¿Está seguro que desea salir del sistema?\n\n" +
                              "Se detendrán todos los procesos de reconocimiento."):
            self.stop_camera()
            self.log_event("Sistema cerrado por el usuario")
            self.root.destroy()

def main():
    
    root = tk.Tk()
    root.title("Sistema de Control de Acceso Facial")
    
    app = FacialAccessControlGUI(root)
    root.protocol("WM_DELETE_WINDOW", app.exit_application)
    
    try:
        root.mainloop()
    except KeyboardInterrupt:
        print("\nSistema interrumpido por el usuario")
        app.exit_application()

if __name__ == "__main__":
    main()