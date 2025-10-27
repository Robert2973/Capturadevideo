import cv2
import filters
from managers import WindowManager, CaptureManager

class Cameo(object):
    def __init__(self):
        self._windowManager = WindowManager('Cameo', self.onKeypress)
        self._captureManager = CaptureManager(cv2.VideoCapture(0), self._windowManager, True)

        # lista de filtros para ciclar
        self._filters = [
            None,  # sin filtro
            ('stroke', filters.strokeEdges),
            ('portra', filters.BGRPortraCurveFilter()),
            ('provia', filters.BGRProviaCurveFilter()),
            ('velvia', filters.BGRVelviaCurveFilter()),
            ('cross', filters.BGRCrossProcessCurveFilter()),
            ('sharpen', filters.SharpenFilter()),
            ('blur', filters.BlurFilter()),
            ('emboss', filters.EmbossFilter()),
        ]
        self._filter_index = 0

    def run(self):
        self._windowManager.createWindow()
        print("Controles: SPACE=screenshot, TAB=start/stop video, f=next filter, q/Esc=salir")

        while self._windowManager.isWindowCreated:
            self._captureManager.enterFrame()
            frame = self._captureManager.frame
            if frame is None:
                self._captureManager.exitFrame()
                self._windowManager.processEvents()
                continue

            # === Aplicar el filtro seleccionado (corregido) ===
            current_filter = self._filters[self._filter_index]

            if current_filter is None:
                pass  # sin filtro

            # Filtro tipo función (como strokeEdges)
            elif isinstance(current_filter, tuple) and current_filter[0] == 'stroke':
                filters.strokeEdges(frame, frame)

            # Filtro tipo clase dentro de tupla (('portra', objeto))
            elif isinstance(current_filter, tuple) and hasattr(current_filter[1], 'apply'):
                current_filter[1].apply(frame, frame)

            # Filtro tipo clase directa (con método .apply)
            elif hasattr(current_filter, 'apply'):
                current_filter.apply(frame, frame)

            # Función directa (por compatibilidad)
            elif callable(current_filter):
                current_filter(frame, frame)

            # === Mostrar frame en ventana ===
            self._captureManager.exitFrame()
            self._windowManager.processEvents()

    def onKeypress(self, keycode):
        # SPACE -> guardar imagen
        if keycode == 32:
            self._captureManager.writeImage('screenshot.png')
            print("Screenshot guardado: screenshot.png")

        # TAB -> iniciar/detener grabación
        elif keycode == 9:
            if not self._captureManager.isWritingVideo:
                self._captureManager.startWritingVideo('screencast.avi')
                print("Grabando video: screencast.avi")
            else:
                self._captureManager.stopWritingVideo()
                print("Grabación detenida")

        # f -> siguiente filtro
        elif keycode == ord('f'):
            self._filter_index = (self._filter_index + 1) % len(self._filters)
            entry = self._filters[self._filter_index]
            name = (
                "none"
                if entry is None
                else (
                    entry[0]
                    if isinstance(entry, tuple)
                    else (
                        entry.__class__.__name__
                        if hasattr(entry, "apply")
                        else str(entry)
                    )
                )
            )
            print(f"Filtro actual: {name}")

        # q o ESC -> salir
        elif keycode == ord('q') or keycode == 27:
            print("Cerrando ventana...")
            self._windowManager.destroyWindow()


if __name__ == "__main__":
    Cameo().run()
