import cv2
import filters
from managers import WindowManager, CaptureManager


class Cameo(object):
    """
    Clase principal para la aplicación Cameo.

    Esta clase gestiona una aplicación de captura de video en tiempo real desde la webcam,
    aplicando filtros visuales seleccionables, y permitiendo capturas de pantalla y grabación de video.

    Utiliza las siguientes dependencias:
    - cv2: Para captura y procesamiento de video.
    - filters: Módulo personalizado con funciones y clases de filtros de imagen.
    - managers.WindowManager: Para manejar la ventana de visualización y eventos de teclado.
    - managers.CaptureManager: Para gestionar la captura de video y su procesamiento.

    Atributos:
        _windowManager (WindowManager): Gestor de la ventana principal.
        _captureManager (CaptureManager): Gestor de la captura de video.
        _filters (list): Lista de filtros disponibles para aplicar al video.
        _filter_index (int): Índice del filtro actualmente seleccionado.
    """

    def __init__(self):
        """
        Inicializa una nueva instancia de Cameo.

        Configura el gestor de ventana con un título y un callback para eventos de teclado,
        inicializa el gestor de captura con la webcam predeterminada (dispositivo 0),
        y define una lista de filtros que se pueden aplicar al video en tiempo real.

        La lista de filtros incluye opciones como curvas de color, efectos de borde,
        nitidez, desenfoque y relieve, además de la opción de no aplicar filtro.
        """
        # Crear el gestor de ventana con título 'Cameo' y asignar el método onKeypress como callback
        # para manejar eventos de teclado
        self._windowManager = WindowManager('Cameo', self.onKeypress)

        # Crear el gestor de captura usando la webcam (cv2.VideoCapture(0)) y el gestor de ventana
        # El parámetro True indica que el video debe mostrarse en la ventana automáticamente
        self._captureManager = CaptureManager(cv2.VideoCapture(0), self._windowManager, True)

        # Definir la lista de filtros disponibles. Cada elemento puede ser:
        # - None: Representa la ausencia de filtro (video original).
        # - Tupla ('nombre', función): Para filtros implementados como funciones, ej. strokeEdges.
        # - Tupla ('nombre', objeto): Para filtros implementados como clases con método apply.
        # - Objeto filtro directo: Clases con método apply que se aplican directamente.
        # Esta estructura permite flexibilidad en la implementación de filtros.
        self._filters = [
            None,  # Sin filtro: Muestra el video sin modificaciones
            ('stroke', filters.strokeEdges),  # Filtro de bordes con efecto de trazo (stroke)
            ('portra', filters.BGRPortraCurveFilter()),  # Filtro de curva de color estilo Portra (calido y suave)
            ('provia', filters.BGRProviaCurveFilter()),  # Filtro de curva de color estilo Provia (vibrante)
            ('velvia', filters.BGRVelviaCurveFilter()),  # Filtro de curva de color estilo Velvia (contraste alto)
            ('cross', filters.BGRCrossProcessCurveFilter()),  # Filtro de proceso cruzado (efecto vintage)
            ('sharpen', filters.SharpenFilter()),  # Filtro de nitidez para resaltar detalles
            ('blur', filters.BlurFilter()),  # Filtro de desenfoque para suavizar la imagen
            ('emboss', filters.EmbossFilter()),  # Filtro de relieve para dar efecto 3D a los bordes
        ]

        # Índice inicial del filtro seleccionado (0 = sin filtro)
        self._filter_index = 0

    def run(self):
        """
        Ejecuta el bucle principal de la aplicación Cameo.

        Este metodo crea la ventana de visualización, informa al usuario sobre los controles disponibles,
        y entra en un bucle infinito que captura frames de video, aplica el filtro seleccionado,
        muestra el resultado en la ventana y procesa eventos de entrada (como teclas).

        El bucle continúa hasta que la ventana sea destruida (por ejemplo, al presionar 'q' o ESC).

        Raises:
            Puede lanzar excepciones relacionadas con OpenCV si hay problemas con la captura de video,
            pero estas no se manejan explícitamente en este código.
        """
        # Crear la ventana para mostrar el video procesado
        self._windowManager.createWindow()

        # Imprimir en consola las instrucciones de uso para el usuario
        print("Controles: SPACE=screenshot, TAB=start/stop video, f=next filter, q/Esc=salir")

        # Bucle principal: Se ejecuta mientras la ventana esté abierta
        while self._windowManager.isWindowCreated:
            # Notificar al gestor de captura que se va a procesar un nuevo frame
            self._captureManager.enterFrame()

            # Obtener el frame actual capturado por la webcam
            frame = self._captureManager.frame

            # Si no se pudo capturar un frame (ej. error en la webcam), saltar al siguiente ciclo
            # Esto evita procesar frames nulos y permite continuar el bucle
            if frame is None:
                self._captureManager.exitFrame()  # Finalizar el procesamiento del frame actual
                self._windowManager.processEvents()  # Procesar eventos de la ventana (teclas, etc.)
                continue  # Pasar al siguiente frame

            # === Sección de aplicación de filtros ===
            # Obtener el filtro actual basado en el índice seleccionado
            current_filter = self._filters[self._filter_index]

            # Caso 1: Sin filtro (None) - No modificar el frame
            if current_filter is None:
                pass  # El frame permanece sin cambios

            # Caso 2: Filtro de tipo tupla con 'stroke' y función strokeEdges
            # strokeEdges es una función que aplica un efecto de bordes con trazo
            elif isinstance(current_filter, tuple) and current_filter[0] == 'stroke':
                filters.strokeEdges(frame, frame)  # Aplicar el filtro directamente al frame (in-place)

            # Caso 3: Filtro de tipo tupla con un objeto que tiene metodo apply
            # Ej. curvas de color como Portra, que son clases con apply(frame, frame)
            elif isinstance(current_filter, tuple) and hasattr(current_filter[1], 'apply'):
                current_filter[1].apply(frame, frame)  # Llamar al metodo apply del objeto filtro

            # Caso 4: Filtro directo como objeto con metodo apply
            # Ej. SharpenFilter, BlurFilter, etc.
            elif hasattr(current_filter, 'apply'):
                current_filter.apply(frame, frame)  # Aplicar el filtro al frame

            # Caso 5: Filtro como función callable (para compatibilidad futura)
            elif callable(current_filter):
                current_filter(frame, frame)  # Llamar a la función con el frame como argumento

            # Nota: La lógica de filtros usa 'frame' como entrada y salida para modificar in-place,
            # lo que es eficiente para procesamiento en tiempo real.

            # === Sección de visualización y eventos ===
            # Finalizar el procesamiento del frame actual y mostrarlo en la ventana
            self._captureManager.exitFrame()

            # Procesar cualquier evento pendiente en la ventana (como pulsaciones de teclas)
            self._windowManager.processEvents()

    def onKeypress(self, keycode):
        """
        Maneja los eventos de teclado durante la ejecución de la aplicación.

        Este metodo es un callback que se llama automáticamente cuando se presiona una tecla
        en la ventana de visualización. Dependiendo del código de la tecla, realiza acciones
        como guardar capturas, iniciar/detener grabaciones, cambiar filtros o salir.

        Args:
            keycode (int): El código ASCII de la tecla presionada (obtenido de cv2.waitKey).
                           Ejemplos: 32 para SPACE, 9 para TAB, ord('f') para 'f'.

        Returns:
            None: Este metodo no retorna valores; solo ejecuta acciones basadas en la tecla.

        Notas:
            - Los códigos de tecla siguen la convención de OpenCV (cv2.waitKey).
            - Las acciones se imprimen en consola para feedback al usuario.
        """
        # Si se presiona SPACE (código 32): Guardar una captura de pantalla del frame actual
        if keycode == 32:
            self._captureManager.writeImage('screenshot.png')  # Guardar imagen como 'screenshot.png'
            print("Screenshot guardado: screenshot.png")  # Confirmación en consola

        # Si se presiona TAB (código 9): Alternar entre iniciar y detener la grabación de video
        elif keycode == 9:
            if not self._captureManager.isWritingVideo:  # Si no se está grabando actualmente
                self._captureManager.startWritingVideo('screencast.avi')  # Iniciar grabación en 'screencast.avi'
                print("Grabando video: screencast.avi")  # Mensaje de inicio
            else:  # Si ya se está grabando
                self._captureManager.stopWritingVideo()  # Detener la grabación
                print("Grabación detenida")  # Mensaje de detención

        # Si se presiona 'f': Cambiar al siguiente filtro en la lista de filtros
        elif keycode == ord('f'):
            # Avanzar el índice al siguiente (cíclico, vuelve a 0 al final de la lista)
            self._filter_index = (self._filter_index + 1) % len(self._filters)

            # Determinar el nombre del filtro actual para mostrarlo al usuario
            entry = self._filters[self._filter_index]
            name = (
                "none"  # Si es None, mostrar "none"
                if entry is None
                else (
                    entry[0]  # Si es tupla, usar el primer elemento como nombre
                    if isinstance(entry, tuple)
                    else (
                        entry.__class__.__name__  # Si es objeto con apply, usar el nombre de la clase
                        if hasattr(entry, "apply")
                        else str(entry)  # Caso por defecto: convertir a string
                    )
                )
            )
            print(f"Filtro actual: {name}")  # Imprimir el nombre del filtro en consola

        # Si se presiona 'q' o ESC (código 27): Salir de la aplicación
        elif keycode == ord('q') or keycode == 27:
            print("Cerrando ventana...")  # Mensaje de salida
            self._windowManager.destroyWindow()  # Destruir la ventana para terminar el bucle


# Punto de entrada del script: Si se ejecuta directamente (no importado como módulo),
# crear una instancia de Cameo y ejecutar su metodo run()
if __name__ == "__main__":
    Cameo().run()
