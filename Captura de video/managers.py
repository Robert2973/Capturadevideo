import cv2
import numpy as np
import time


class CaptureManager(object):
    """
    Clase para gestionar la captura de video desde una fuente.

    Maneja la captura de frames, cálculo de FPS, previsualización en ventana,
    guardado de imágenes y grabación de video. Soporta mirroring de previsualización
    y es compatible con versiones de OpenCV 3.x y 4.x.

    Atributos:
        previewWindowManager (WindowManager, opcional): Gestor de la ventana de previsualización.
        shouldMirrorPreview (bool): Si True, refleja horizontalmente la previsualización.
        _capture (cv2.VideoCapture): Objeto de captura de video.
        _channel (int): Canal de captura (usado en algunos dispositivos).
        _enteredFrame (bool): Indica si se ha entrado en un frame sin salir.
        _frame (numpy.ndarray): Frame actual capturado.
        _imageFilename (str): Nombre del archivo para guardar imagen (None si no se guarda).
        _videoFilename (str): Nombre del archivo para grabar video (None si no se graba).
        _videoEncoding: Código de codificación para el video.
        _videoWriter (cv2.VideoWriter): Escritor de video.
        _startTime (float): Tiempo de inicio para cálculo de FPS.
        _framesElapsed (int): Número de frames procesados.
        _fpsEstimate (float): Estimación de FPS.
    """

    def __init__(self, capture, previewWindowManager=None, shouldMirrorPreview=False):
        """
        Inicializa el gestor de captura.

        Args:
            capture (cv2.VideoCapture): Objeto de captura de video.
            previewWindowManager (WindowManager, opcional): Gestor para mostrar previsualización.
            shouldMirrorPreview (bool): Si True, refleja la previsualización (útil para webcams frontales).
        """
        # Asignar el gestor de ventana y opción de mirroring
        self.previewWindowManager = previewWindowManager
        self.shouldMirrorPreview = shouldMirrorPreview

        # Inicializar atributos internos
        self._capture = capture
        self._channel = 0  # Canal predeterminado
        self._enteredFrame = False  # Bandera para controlar entrada/salida de frames
        self._frame = None  # Frame actual
        self._imageFilename = None  # Archivo para imagen (None = no guardar)
        self._videoFilename = None  # Archivo para video (None = no grabar)
        self._videoEncoding = None  # Codificación de video
        self._videoWriter = None  # Escritor de video

        # Variables para estimación de FPS
        self._startTime = None
        self._framesElapsed = 0
        self._fpsEstimate = None

    @property
    def channel(self):
        """
        Propiedad para obtener el canal de captura actual.

        Returns:
            int: Número del canal.
        """
        return self._channel

    @channel.setter
    def channel(self, value):
        """
        Establece el canal de captura y resetea el frame si cambia.

        Args:
            value (int): Nuevo canal.
        """
        if self._channel != value:
            self._channel = value
            self._frame = None  # Resetear frame al cambiar canal

    @property
    def frame(self):
        """
        Propiedad para obtener el frame actual.

        Si no se ha capturado aún en este ciclo, lo recupera de la fuente.
        Compatible con OpenCV 3.x (usando read) y 4.x (usando retrieve).

        Returns:
            numpy.ndarray or None: Frame capturado o None si falla.
        """
        # Solo capturar si se ha entrado en el frame y no se ha hecho aún
        if self._enteredFrame and self._frame is None:
            # En OpenCV 4.x, retrieve() no acepta 'channel'; usar read() como fallback
            if hasattr(self._capture, 'retrieve'):
                res, self._frame = self._capture.retrieve()
            else:
                res, self._frame = self._capture.read()
            # Si falla la captura, establecer a None
            if not res:
                self._frame = None
        return self._frame

    @property
    def isWritingImage(self):
        """
        Propiedad que indica si se está guardando una imagen.

        Returns:
            bool: True si hay un archivo de imagen pendiente.
        """
        return self._imageFilename is not None

    @property
    def isWritingVideo(self):
        """
        Propiedad que indica si se está grabando video.

        Returns:
            bool: True si hay un archivo de video activo.
        """
        return self._videoFilename is not None

    def enterFrame(self):
        """
        Entra en el ciclo de captura de un nuevo frame.

        Captura el siguiente frame de la fuente sin procesarlo aún.
        Debe llamarse antes de acceder a 'frame' y emparejarse con exitFrame().

        Raises:
            AssertionError: Si se llama sin un exitFrame() previo.
        """
        # Asegurar que no hay un frame pendiente
        assert not self._enteredFrame, 'previous enterFrame() had no matching exitFrame()'
        if self._capture is not None:
            # Capturar (grab) el siguiente frame
            self._enteredFrame = self._capture.grab()

    def exitFrame(self):
        """
        Sale del ciclo de frame, procesando y liberando el frame actual.

        Calcula FPS, muestra en ventana, guarda imagen/video si corresponde,
        y libera el frame para el siguiente ciclo.
        """
        # Si no hay frame, resetear y salir
        if self.frame is None:
            self._enteredFrame = False
            self._frame = None
            return

        # Calcular estimación de FPS
        if self._framesElapsed == 0:
            self._startTime = time.time()  # Iniciar temporizador en el primer frame
        else:
            timeElapsed = time.time() - self._startTime
            if timeElapsed > 0:
                self._fpsEstimate = self._framesElapsed / timeElapsed  # FPS = frames / tiempo
        self._framesElapsed += 1  # Incrementar contador de frames

        # Mostrar el frame en la ventana de previsualización si existe
        if self.previewWindowManager is not None:
            if self.shouldMirrorPreview:
                # Reflejar horizontalmente el frame para previsualización (ej. webcam)
                mirrored = np.fliplr(self._frame).copy()
                self.previewWindowManager.show(mirrored)
            else:
                # Mostrar el frame sin mirroring
                self.previewWindowManager.show(self._frame)

        # Guardar imagen si se solicitó
        if self.isWritingImage:
            cv2.imwrite(self._imageFilename, self._frame)
            self._imageFilename = None  # Resetear después de guardar

        # Escribir frame en video si corresponde
        self._writeVideoFrame()

        # Liberar el frame y resetear banderas
        self._frame = None
        self._enteredFrame = False

    def writeImage(self, filename):
        """
        Solicita guardar el próximo frame como imagen.

        Args:
            filename (str): Nombre del archivo de imagen (ej. 'screenshot.png').
        """
        self._imageFilename = filename

    def startWritingVideo(self, filename, encoding=None):
        """
        Inicia la grabación de video en el archivo especificado.

        Args:
            filename (str): Nombre del archivo de video (ej. 'video.avi').
            encoding (opcional): Código de codificación (por defecto XVID).
        """
        self._videoFilename = filename
        if encoding is None:
            # Codificación predeterminada para compatibilidad
            self._videoEncoding = cv2.VideoWriter_fourcc(*'XVID')
        else:
            self._videoEncoding = encoding

    def stopWritingVideo(self):
        """
        Detiene la grabación de video y libera el escritor.
        """
        self._videoFilename = None
        self._videoEncoding = None
        if self._videoWriter is not None:
            self._videoWriter.release()  # Liberar recursos
        self._videoWriter = None

    def _writeVideoFrame(self):
        """
        Escribe el frame actual en el archivo de video si está grabando.

        Inicializa el VideoWriter si es necesario, usando FPS estimados si no se obtienen de la captura.
        """
        # Solo proceder si se está grabando video
        if not self.isWritingVideo:
            return

        # Inicializar VideoWriter si no existe
        if self._videoWriter is None:
            # Obtener FPS de la captura
            fps = self._capture.get(cv2.CAP_PROP_FPS)
            # Si FPS es inválido, usar estimación o valor por defecto
            if fps == 0 or fps is None or np.isnan(fps):
                if self._framesElapsed < 20:
                    return  # Esperar más frames para estimar FPS
                else:
                    fps = self._fpsEstimate if self._fpsEstimate is not None else 30.0

            # Obtener dimensiones del frame
            width = int(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            size = (width, height)

            # Crear el VideoWriter
            self._videoWriter = cv2.VideoWriter(self._videoFilename, self._videoEncoding, fps, size)

        # Escribir el frame si el escritor está listo
        if self._videoWriter is not None:
            self._videoWriter.write(self._frame)


class WindowManager(object):
    """
    Clase para gestionar una ventana de OpenCV.

    Maneja la creación, visualización y destrucción de la ventana,
    así como el procesamiento de eventos de teclado.

    Atributos:
        keypressCallback (callable, opcional): Función a llamar en eventos de teclado.
        _windowName (str): Nombre de la ventana.
        _isWindowCreated (bool): Indica si la ventana está creada.
    """

    def __init__(self, windowName, keypressCallback=None):
        """
        Inicializa el gestor de ventana.

        Args:
            windowName (str): Nombre de la ventana.
            keypressCallback (callable, opcional): Callback para teclas presionadas.
        """
        self.keypressCallback = keypressCallback
        self._windowName = windowName
        self._isWindowCreated = False

    @property
    def isWindowCreated(self):
        """
        Propiedad que indica si la ventana está creada.

        Returns:
            bool: True si la ventana existe.
        """
        return self._isWindowCreated

    def createWindow(self):
        """
        Crea la ventana con el nombre especificado.
        """
        cv2.namedWindow(self._windowName)
        self._isWindowCreated = True

    def show(self, frame):
        """
        Muestra un frame en la ventana.

        Args:
            frame (numpy.ndarray): Imagen a mostrar.
        """
        cv2.imshow(self._windowName, frame)

    def destroyWindow(self):
        """
        Destruye la ventana y resetea el estado.
        """
        cv2.destroyWindow(self._windowName)
        self._isWindowCreated = False

    def processEvents(self):
        """
        Procesa eventos de la ventana, incluyendo teclas presionadas.

        Llama al callback si hay una tecla y este está definido.
        """
        # Esperar 1 ms por eventos y obtener código de tecla
        keycode = cv2.waitKey(1)
        if self.keypressCallback is not None and keycode != -1:
            # Máscara para compatibilidad (obtener solo 8 bits bajos)
            keycode &= 0xFF
            self.keypressCallback(keycode)
