# Sistema de Control de Acceso Facial

Sistema avanzado de reconocimiento facial para control de acceso basado en OpenCV y Python, con interfaz gráfica intuitiva y múltiples niveles de seguridad.

##  Características Principales

### Reconocimiento Facial Avanzado
- **Algoritmo LBPH Mejorado**: Reconocimiento facial de alta precisión con configuración optimizada
- **Detección Multi-escala**: Detecta caras en diferentes tamaños y distancias
- **Preprocesamiento Inteligente**: 
  - Reducción de ruido con fastNlMeansDenoising
  - Mejora de contraste adaptativa (CLAHE)
  - Filtros de nitidez y normalización
  - Validación de calidad de imagen
- **Eliminación de Duplicados**: Algoritmo IoU para filtrar detecciones falsas

### Sistema de Control de Acceso
- **Gestión de Usuarios**: Registro y administración de personas
- **Roles y Permisos**: Sistema de 3 niveles
  -  Administrador
  -  Usuario
  -  Visitante
- **Registro de Eventos**: Historial completo de accesos con timestamps
- **Modo de Verificación**: Sistema de verificación continua con alertas

### Interfaz Gráfica Moderna
- **Dashboard Intuitivo**: Panel de control visual con Tkinter
- **Cámara en Tiempo Real**: Vista previa con overlays de detección
- **Panel de Configuración**: Ajuste de parámetros de reconocimiento
- **Métricas de Rendimiento**: Monitoreo de FPS y estado del sistema
- **Log de Eventos**: Visualización en tiempo real de accesos

## Instalación

### Requisitos Previos
- Python 3.7 o superior
- Cámara web (webcam)
- Sistema operativo: Windows, Linux o macOS

### Instalación de Dependencias

```bash
# Clonar el repositorio
git clone https://github.com/tu-usuario/sistema-acceso-facial.git
cd sistema-acceso-facial

# Instalar dependencias
pip install -r requirements.txt
```

### Configuración de OpenCV

Si encuentras problemas con `opencv-contrib-python`, puedes instalar manualmente:

```bash
pip install opencv-python==4.8.0.74
pip install opencv-contrib-python==4.8.0.74
```

##  Uso

### Iniciar el Sistema

```bash
python Acceso_facial.py
```

### Flujo de Trabajo Básico

1. **Registrar Caras**
   - Click en "📷 Registrar Nueva Cara"
   - Ingresa el nombre de la persona
   - Selecciona el rol de acceso
   - Captura múltiples fotos con diferentes expresiones

2. **Entrenar el Modelo**
   - Click en "🧠 Entrenar Modelo"
   - El sistema procesará automáticamente todas las caras registradas
   - Espera a que se complete el entrenamiento

3. **Verificar Acceso**
   - Click en "🔍 Verificar Persona"
   - El sistema reconocerá automáticamente las caras registradas
   - Se mostrarán alertas visuales según el nivel de acceso

4. **Gestionar Usuarios**
   - Click en "👥 Gestionar Caras"
   - Ver lista de personas registradas
   - Eliminar usuarios si es necesario

##  Controles y Funcionalidades

### Panel Principal
- ** Registrar Nueva Cara**: Agregar nuevas personas al sistema
- ** Entrenar Modelo**: Entrenar el algoritmo de reconocimiento
- ** Verificar Persona**: Activar modo de verificación continua
- ** Gestionar Caras**: Administrar usuarios registrados
- ** Configuración**: Ajustar parámetros del sistema

### Panel de Configuración
- **Umbral de Confianza**: Control deslizante (0-100%)
  - Valores bajos: Más estricto
  - Valores altos: Más permisivo
- **Aplicar Configuración**: Guardar cambios

### Atajos de Teclado
- `Esc`: Salir del modo de captura/verificación
- `Espacio`: Capturar foto durante el registro

## Estructura del Proyecto

```
sistema-acceso-facial/
│
├── Acceso_facial.py          # Archivo principal
├── requirements.txt           # Dependencias
├── README.md                  # Documentación
├── .gitignore                # Archivos ignorados
│
├── known_faces/              # Carpeta de caras registradas
│   └── [nombre]/            # Una carpeta por persona
│       ├── photo_1.jpg
│       └── photo_2.jpg
│
├── face_model.pkl           # Modelo entrenado
└── access_log.txt           # Registro de eventos
```

## Parámetros Técnicos

### Reconocimiento LBPH
- **Radius**: 2 (área de análisis)
- **Neighbors**: 16 (puntos de comparación)
- **Grid**: 8x8 (división de la imagen)
- **Threshold**: 120.0 (umbral de confianza)

### Detección de Caras
- **Tamaño mínimo**: 60x60 píxeles
- **Tamaño máximo**: 300x300 píxeles
- **Escalas de detección**: Múltiples (1.1, 1.15)
- **Vecinos mínimos**: 5-6 (anti-falsos positivos)

### Procesamiento de Imagen
- **Resolución de preprocesamiento**: 200x200 píxeles
- **Varianza mínima**: 100 (validación de calidad)
- **Densidad de bordes**: 2% (detección de falsos positivos)

## Casos de Uso

- **Control de Acceso Residencial**: Edificios y condominios
- **Seguridad Empresarial**: Oficinas y áreas restringidas
- **Sistemas de Asistencia**: Registro de entrada/salida
- **Espacios Educativos**: Control de acceso en instituciones
- **Eventos**: Verificación de invitados y participantes

## Limitaciones y Consideraciones

- Requiere buena iluminación para óptimo rendimiento
- Las caras deben estar claramente visibles y frontales
- Se recomienda registrar múltiples fotos por persona
- El rendimiento depende de la calidad de la cámara
- No está diseñado para ambientes de alta seguridad crítica

## Privacidad y Seguridad

- Los datos faciales se almacenan **localmente**
- No se envía información a servidores externos
- Las imágenes se guardan en carpetas del sistema local
- El modelo de reconocimiento es privado
- Se recomienda cumplir con regulaciones de protección de datos (ej. GDPR, Ley 1581 de Colombia)


## Problemas Conocidos

- En algunas configuraciones, OpenCV puede requerir instalación manual
- El rendimiento puede variar según el hardware
- Webcams de baja calidad pueden afectar la precisión


## Autor

**Luis Estupiñan Morales**
- Ingeniero Multimedia - Universidad Militar Nueva Granada
- GitHub: [@Mirrox999](https://github.com/mirrox999)
