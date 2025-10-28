# 🎥 Sistema de Captura de Video con Filtros en Tiempo Real

**Cameo** es una aplicación desarrollada en **Python** que utiliza **OpenCV** para capturar video en tiempo real desde una cámara, aplicar diversos **filtros visuales**, realizar **grabaciones** y **tomar capturas de pantalla**.  
El proyecto está estructurado en módulos independientes que gestionan la captura, el procesamiento y la visualización del video, facilitando su mantenimiento y extensión.

---

## 🚀 Características principales

- Captura de video en tiempo real desde webcam o dispositivo conectado.  
- Aplicación de filtros de color, curvas y efectos especiales.  
- Grabación de video y toma de imágenes instantáneas.  
- Interfaz visual sencilla basada en ventanas de OpenCV.  
- Arquitectura modular con clases reutilizables y documentación completa.

---

## 🧩 Estructura del proyecto

- **cameo.py** → Archivo principal. Controla la aplicación, cámara y filtros.  
- **filters.py** → Contiene las clases y funciones de filtros de imagen.  
- **managers.py** → Maneja la captura, grabación y visualización del video.  
- **utils.py** → Funciones auxiliares: curvas, interpolación y lookup tables.

---

## 🧠 Descripción de los módulos

### 🎬 `cameo.py`
Archivo principal que inicializa la cámara, la ventana y administra los filtros.  
Permite recorrer diferentes efectos presionando teclas definidas.

### 🎨 `filters.py`
Contiene los filtros disponibles:
- Recoloraciones por canales (`recolorRC`, `recolorRGV`, `recolorCMV`).
- Efectos de película (`BGRPortraCurveFilter`, `BGRProviaCurveFilter`, `BGRVelviaCurveFilter`, `BGRCrossProcessCurveFilter`).
- Filtros de convolución (`BlurFilter`, `SharpenFilter`, `EmbossFilter`).
- Efectos especiales como `strokeEdges`.

### 🪟 `managers.py`
Incluye dos clases principales:
- **CaptureManager:** controla la entrada de video, FPS, grabación y captura de imágenes.
- **WindowManager:** administra la ventana de previsualización y eventos del teclado.

### ⚙️ `utils.py`
Proporciona funciones auxiliares para interpolación y procesamiento de curvas:
- `createCurveFunc()` → genera funciones de curvas suaves.
- `createLookupArray()` → crea tablas de búsqueda (lookup tables).
- `applyLookupArray()` → aplica transformaciones rápidas a imágenes.
- `flatView()` → genera vistas planas de arreglos sin copiar datos.

---

## 🖥️ Requisitos

Antes de ejecutar el programa, asegúrate de tener instalado:

- Python 3.8 o superior  
- OpenCV 
- NumPy  
- SciPy

Puedes instalarlos fácilmente con:


pip install opencv-python numpy scipy

---

## Entorno virtual

- Creacion del entorno → python -m venv venv  
- Activar entorno → venv\Scripts\activate 
- Instalar dependencias → pip install opencv-python numpy scipy 
- Para iniciar la aplicación → python cameo.py

---

## ⌨️ Controles del teclado

Durante la ejecución del programa, puedes usar las siguientes teclas:

| Tecla   | Acción                                 |
| ------- | -------------------------------------- |
| Espacio | Captura una imagen (`screenshot.png`)  |
| F       | Cambia al siguiente filtro disponible  |
| Tap     | Inicia o detiene la grabación de video |
| q/Esc   | Cierra la aplicación                   |

---

## 💡 Ejemplo de uso

1. Ejecuta `cameo.py`.  
2. Verás la imagen de tu cámara en la ventana de OpenCV.  
3. Presiona **F** para aplicar diferentes filtros visuales.  
4. Presiona **Espacio** para guardar una captura.  
5. Presiona **Tap** para grabar un video (se guarda como `output.avi`).  
6. Presiona **Esc/q** para salir del programa.

---

### 👤 Autor

**Roberto Carlos Hernández**  
Proyecto basado en ejercicios y estructura del libro  
*"OpenCV Computer Vision Projects with Python – Learning Path"* (Joseph Howse, 2016).

*"Del código a la cámara: una herramienta visual interactiva para aplicar procesamiento de imagen en tiempo real."*

---

