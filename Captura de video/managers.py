import cv2
import numpy as np
import time

class CaptureManager(object):
    def __init__(self, capture, previewWindowManager=None, shouldMirrorPreview=False):
        self.previewWindowManager = previewWindowManager
        self.shouldMirrorPreview = shouldMirrorPreview

        self._capture = capture
        self._channel = 0
        self._enteredFrame = False
        self._frame = None
        self._imageFilename = None
        self._videoFilename = None
        self._videoEncoding = None
        self._videoWriter = None

        self._startTime = None
        self._framesElapsed = 0
        self._fpsEstimate = None

    @property
    def channel(self):
        return self._channel

    @channel.setter
    def channel(self, value):
        if self._channel != value:
            self._channel = value
            self._frame = None

    @property
    def frame(self):
        if self._enteredFrame and self._frame is None:
            # En OpenCV 4.x, retrieve() ya no acepta 'channel'
            if hasattr(self._capture, 'retrieve'):
                res, self._frame = self._capture.retrieve()
            else:
                res, self._frame = self._capture.read()

            if not res:
                self._frame = None
        return self._frame

    @property
    def isWritingImage(self):
        return self._imageFilename is not None

    @property
    def isWritingVideo(self):
        return self._videoFilename is not None

    def enterFrame(self):
        """Captura el siguiente frame, si existe."""
        assert not self._enteredFrame, 'previous enterFrame() had no matching exitFrame()'
        if self._capture is not None:
            # Captura el siguiente frame
            self._enteredFrame = self._capture.grab()

    def exitFrame(self):
        """Muestra, guarda y libera el frame actual."""
        if self.frame is None:
            self._enteredFrame = False
            self._frame = None
            return

        # Calcular FPS estimado
        if self._framesElapsed == 0:
            self._startTime = time.time()
        else:
            timeElapsed = time.time() - self._startTime
            if timeElapsed > 0:
                self._fpsEstimate = self._framesElapsed / timeElapsed
        self._framesElapsed += 1

        # Mostrar el frame
        if self.previewWindowManager is not None:
            if self.shouldMirrorPreview:
                mirrored = np.fliplr(self._frame).copy()
                self.previewWindowManager.show(mirrored)
            else:
                self.previewWindowManager.show(self._frame)

        # Guardar imagen si se solicitó
        if self.isWritingImage:
            cv2.imwrite(self._imageFilename, self._frame)
            self._imageFilename = None

        # Escribir video si corresponde
        self._writeVideoFrame()

        # Liberar frame
        self._frame = None
        self._enteredFrame = False

    def writeImage(self, filename):
        """Guardar una imagen (screenshot)."""
        self._imageFilename = filename

    def startWritingVideo(self, filename, encoding=None):
        """Iniciar grabación de video."""
        self._videoFilename = filename
        if encoding is None:
            self._videoEncoding = cv2.VideoWriter_fourcc(*'XVID')
        else:
            self._videoEncoding = encoding

    def stopWritingVideo(self):
        """Detener grabación de video."""
        self._videoFilename = None
        self._videoEncoding = None
        if self._videoWriter is not None:
            self._videoWriter.release()
        self._videoWriter = None

    def _writeVideoFrame(self):
        """Escribir frame actual en el archivo de video."""
        if not self.isWritingVideo:
            return

        if self._videoWriter is None:
            fps = self._capture.get(cv2.CAP_PROP_FPS)
            if fps == 0 or fps is None or np.isnan(fps):
                if self._framesElapsed < 20:
                    return  # Esperar estimación de FPS
                else:
                    fps = self._fpsEstimate if self._fpsEstimate is not None else 30.0

            width = int(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            size = (width, height)

            self._videoWriter = cv2.VideoWriter(self._videoFilename, self._videoEncoding, fps, size)

        if self._videoWriter is not None:
            self._videoWriter.write(self._frame)


class WindowManager(object):
    def __init__(self, windowName, keypressCallback=None):
        self.keypressCallback = keypressCallback
        self._windowName = windowName
        self._isWindowCreated = False

    @property
    def isWindowCreated(self):
        return self._isWindowCreated

    def createWindow(self):
        cv2.namedWindow(self._windowName)
        self._isWindowCreated = True

    def show(self, frame):
        cv2.imshow(self._windowName, frame)

    def destroyWindow(self):
        cv2.destroyWindow(self._windowName)
        self._isWindowCreated = False

    def processEvents(self):
        keycode = cv2.waitKey(1)
        if self.keypressCallback is not None and keycode != -1:
            keycode &= 0xFF
            self.keypressCallback(keycode)
