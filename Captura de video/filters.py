import cv2
import numpy as np
import utils


# --- Funciones de recolor (mezcla de canales) ---
def recolorRC(src, dst):
    """
    Aplica un filtro de recolor basado en la mezcla de canales Rojo y Cian.

    Mezcla los canales Azul (B) y Verde (G) para crear un efecto de cian,
    y combina con el canal Rojo (R) original. Esto produce un tono cian-rojo.

    Args:
        src (numpy.ndarray): Imagen de entrada en formato BGR.
        dst (numpy.ndarray): Imagen de salida donde se almacenará el resultado.

    Nota: Modifica dst in-place.
    """
    # Separar la imagen en sus canales B, G, R
    b, g, r = cv2.split(src)
    # Mezclar B y G con pesos iguales (0.5) para crear un canal de cian
    cv2.addWeighted(b, 0.5, g, 0.5, 0, b)
    # Combinar el nuevo canal B (cian) con sí mismo y R para el resultado
    cv2.merge((b, b, r), dst)


def recolorRGV(src, dst):
    """
    Aplica un filtro de recolor que toma el mínimo de los canales Azul, Verde y Rojo.

    Crea un efecto donde el canal Azul se establece al mínimo de B, G y R,
    resultando en tonos oscuros y un aspecto monocromático con énfasis en sombras.

    Args:
        src (numpy.ndarray): Imagen de entrada en formato BGR.
        dst (numpy.ndarray): Imagen de salida donde se almacenará el resultado.

    Nota: Modifica dst in-place.
    """
    # Separar la imagen en sus canales B, G, R
    b, g, r = cv2.split(src)
    # Establecer B al mínimo entre B y G
    cv2.min(b, g, b)
    # Luego, establecer B al mínimo entre el resultado anterior y R
    cv2.min(b, r, b)
    # Combinar el nuevo B con G y R originales
    cv2.merge((b, g, r), dst)


def recolorCMV(src, dst):
    """
    Aplica un filtro de recolor que toma el máximo de los canales Azul, Verde y Rojo.

    Crea un efecto donde el canal Azul se establece al máximo de B, G y R,
    resultando en tonos brillantes y un aspecto luminoso.

    Args:
        src (numpy.ndarray): Imagen de entrada en formato BGR.
        dst (numpy.ndarray): Imagen de salida donde se almacenará el resultado.

    Nota: Modifica dst in-place.
    """
    # Separar la imagen en sus canales B, G, R
    b, g, r = cv2.split(src)
    # Establecer B al máximo entre B y G
    cv2.max(b, g, b)
    # Luego, establecer B al máximo entre el resultado anterior y R
    cv2.max(b, r, b)
    # Combinar el nuevo B con G y R originales
    cv2.merge((b, g, r), dst)


# --- Clases para curvas y funciones (filtros basados en lookup arrays) ---
class VFuncFilter(object):
    """
    Filtro base que aplica una función de valor (V) a todos los canales de la imagen.

    Utiliza un array de lookup (precomputado) para mapear valores de intensidad
    usando una función personalizada. Es eficiente para transformaciones no lineales.

    Atributos:
        _vLookupArray (numpy.ndarray): Array de lookup para la función de valor.
    """

    def __init__(self, vFunc=None, dtype=np.uint8):
        """
        Inicializa el filtro con una función de valor opcional.

        Args:
            vFunc (callable, opcional): Función que toma un valor y retorna otro (ej. lambda x: x**2).
            dtype (numpy.dtype): Tipo de dato para el array (por defecto np.uint8).
        """
        # Calcular la longitud del array de lookup basada en el rango del tipo de dato
        length = np.iinfo(dtype).max + 1
        # Crear el array de lookup usando la función proporcionada
        self._vLookupArray = utils.createLookupArray(vFunc, length)

    def apply(self, src, dst):
        """
        Aplica el filtro a la imagen de entrada.

        Args:
            src (numpy.ndarray): Imagen de entrada.
            dst (numpy.ndarray): Imagen de salida donde se almacenará el resultado.

        Nota: Usa vistas planas de la imagen para aplicar el lookup eficientemente.
        """
        # Obtener vistas planas (1D) de src y dst para procesamiento
        srcFlat = utils.flatView(src)
        dstFlat = utils.flatView(dst)
        # Aplicar el array de lookup a los valores planos
        utils.applyLookupArray(self._vLookupArray, srcFlat, dstFlat)


class VCurveFilter(VFuncFilter):
    """
    Filtro que aplica una curva de valor (V) definida por puntos de control.

    Extiende VFuncFilter para crear una función de curva a partir de puntos,
    permitiendo ajustes no lineales como curvas de tono en fotografía.
    """

    def __init__(self, vPoints=None, dtype=np.uint8):
        """
        Inicializa el filtro con puntos de curva para el valor.

        Args:
            vPoints (list of tuples, opcional): Lista de puntos (x, y) para definir la curva.
            dtype (numpy.dtype): Tipo de dato para el array (por defecto np.uint8).
        """
        # Llamar al constructor padre con una función de curva creada a partir de los puntos
        super().__init__(utils.createCurveFunc(vPoints), dtype)


class BGRFuncFilter(object):
    """
    Filtro que aplica funciones separadas a cada canal BGR, con una función de valor global.

    Permite ajustes independientes por canal (Azul, Verde, Rojo) y una transformación global de valor.
    Utiliza arrays de lookup para eficiencia.

    Atributos:
        _bLookupArray, _gLookupArray, _rLookupArray (numpy.ndarray): Arrays de lookup por canal.
    """

    def __init__(self, vFunc=None, bFunc=None, gFunc=None, rFunc=None, dtype=np.uint8):
        """
        Inicializa el filtro con funciones para valor y cada canal.

        Args:
            vFunc (callable, opcional): Función global de valor.
            bFunc, gFunc, rFunc (callable, opcional): Funciones para canales B, G, R.
            dtype (numpy.dtype): Tipo de dato para los arrays (por defecto np.uint8).
        """
        # Calcular la longitud del array de lookup
        length = np.iinfo(dtype).max + 1

        # Función auxiliar para componer funciones (si ambas existen, aplicar una después de la otra)
        def comp(a, b): return (lambda x: a(b(x))) if (a is not None and b is not None) else (a if b is None else b)

        # Crear arrays de lookup compuestos: primero vFunc, luego la función del canal
        # Nota: Verifica si utils tiene createCompositeFunc; si no, usa None (se maneja abajo)
        self._bLookupArray = utils.createLookupArray(
            utils.createCompositeFunc(bFunc, vFunc) if hasattr(utils, 'createCompositeFunc') else None, length)
        self._gLookupArray = utils.createLookupArray(
            utils.createCompositeFunc(gFunc, vFunc) if hasattr(utils, 'createCompositeFunc') else None, length)
        self._rLookupArray = utils.createLookupArray(
            utils.createCompositeFunc(rFunc, vFunc) if hasattr(utils, 'createCompositeFunc') else None, length)

    def apply(self, src, dst):
        """
        Aplica el filtro a la imagen, procesando cada canal por separado.

        Args:
            src (numpy.ndarray): Imagen de entrada en BGR.
            dst (numpy.ndarray): Imagen de salida.

        Nota: Solo aplica lookup si el array correspondiente no es None.
        """
        # Separar la imagen en canales B, G, R
        b, g, r = cv2.split(src)
        # Aplicar lookup a cada canal si el array existe
        if self._bLookupArray is not None: utils.applyLookupArray(self._bLookupArray, b, b)
        if self._gLookupArray is not None: utils.applyLookupArray(self._gLookupArray, g, g)
        if self._rLookupArray is not None: utils.applyLookupArray(self._rLookupArray, r, r)
        # Combinar los canales modificados en dst
        cv2.merge([b, g, r], dst)


# Función fallback para createCompositeFunc si no está en utils (basado en el libro original)
def createCompositeFunc(func0, func1):
    """
    Crea una función compuesta aplicando func0 después de func1.

    Si una de las funciones es None, retorna la otra. Si ambas son None, retorna None.

    Args:
        func0, func1 (callable or None): Funciones a componer.

    Returns:
        callable or None: Función compuesta o None.
    """
    if func0 is None:
        return func1
    if func1 is None:
        return func0
    return lambda x: func0(func1(x))


# Monkey-patch: Agregar createCompositeFunc a utils si no existe
if not hasattr(utils, 'createCompositeFunc'):
    utils.createCompositeFunc = createCompositeFunc


class BGRCurveFilter(BGRFuncFilter):
    """
    Filtro que aplica curvas definidas por puntos a cada canal BGR y al valor global.

    Extiende BGRFuncFilter para usar curvas en lugar de funciones arbitrarias,
    ideal para emulación de películas fotográficas.
    """

    def __init__(self, vPoints=None, bPoints=None, gPoints=None, rPoints=None, dtype=np.uint8):
        """
        Inicializa el filtro con puntos de curva para valor y canales.

        Args:
            vPoints, bPoints, gPoints, rPoints (list of tuples, opcional): Puntos para curvas.
            dtype (numpy.dtype): Tipo de dato (por defecto np.uint8).
        """
        # Llamar al constructor padre con funciones de curva creadas a partir de puntos
        super().__init__(utils.createCurveFunc(vPoints), utils.createCurveFunc(bPoints),
                         utils.createCurveFunc(gPoints), utils.createCurveFunc(rPoints), dtype)


# --- Filtros de emulación de película (basados en curvas de películas Kodak) ---
class BGRPortraCurveFilter(BGRCurveFilter):
    """
    Filtro que emula la curva de color de la película Kodak Portra.

    Produce tonos cálidos y suaves, típicos de retratos.
    """

    def __init__(self, dtype=np.uint8):
        """
        Inicializa con curvas predefinidas para Portra.

        Args:
            dtype (numpy.dtype): Tipo de dato (por defecto np.uint8).
        """
        # Curvas basadas en datos de la película Portra
        super().__init__(vPoints=[(0, 0), (23, 20), (157, 173), (255, 255)],
                         bPoints=[(0, 0), (41, 46), (231, 228), (255, 255)],
                         gPoints=[(0, 0), (52, 47), (189, 196), (255, 255)],
                         rPoints=[(0, 0), (69, 69), (213, 218), (255, 255)],
                         dtype=dtype)


class BGRProviaCurveFilter(BGRCurveFilter):
    """
    Filtro que emula la curva de color de la película Kodak Provia.

    Produce colores vibrantes y naturales, ideal para paisajes.
    """

    def __init__(self, dtype=np.uint8):
        """
        Inicializa con curvas predefinidas para Provia.

        Args:
            dtype (numpy.dtype): Tipo de dato (por defecto np.uint8).
        """
        # Curvas basadas en datos de la película Provia
        super().__init__(bPoints=[(0, 0), (35, 25), (205, 227), (255, 255)],
                         gPoints=[(0, 0), (27, 21), (196, 207), (255, 255)],
                         rPoints=[(0, 0), (59, 54), (202, 210), (255, 255)],
                         dtype=dtype)


class BGRVelviaCurveFilter(BGRCurveFilter):
    """
    Filtro que emula la curva de color de la película Kodak Velvia.

    Produce alto contraste y colores saturados, típico de diapositivas.
    """

    def __init__(self, dtype=np.uint8):
        """
        Inicializa con curvas predefinidas para Velvia.

        Args:
            dtype (numpy.dtype): Tipo de dato (por defecto np.uint8).
        """
        # Curvas basadas en datos de la película Velvia
        super().__init__(vPoints=[(0, 0), (128, 118), (221, 215), (255, 255)],
                         bPoints=[(0, 0), (25, 21), (122, 153), (165, 206), (255, 255)],
                         gPoints=[(0, 0), (25, 21), (95, 102), (181, 208), (255, 255)],
                         rPoints=[(0, 0), (41, 28), (183, 209), (255, 255)],
                         dtype=dtype)


class BGRCrossProcessCurveFilter(BGRCurveFilter):
    """
    Filtro que emula el efecto de proceso cruzado (cross-processing).

    Produce tonos vintage con dominantes de color inusuales.
    """

    def __init__(self, dtype=np.uint8):
        """
        Inicializa con curvas predefinidas para cross-processing.

        Args:
            dtype (numpy.dtype): Tipo de dato (por defecto np.uint8).
        """
        # Curvas para simular cross-processing
        super().__init__(bPoints=[(0, 20), (255, 235)],
                         gPoints=[(0, 0), (56, 39), (208, 226), (255, 255)],
                         rPoints=[(0, 0), (56, 22), (211, 255), (255, 255)],
                         dtype=dtype)


# --- Función para bordes con efecto de trazo (estilo cómic) ---
def strokeEdges(src, dst, blurKsize=7, edgeKsize=5):
    """
    Aplica un filtro de bordes con efecto de trazo, simulando un estilo de cómic.

    Detecta bordes usando Laplaciano, los invierte y los multiplica con la imagen original
    para crear un efecto de "trazo" negro sobre los bordes.

    Args:
        src (numpy.ndarray): Imagen de entrada en BGR.
        dst (numpy.ndarray): Imagen de salida.
        blurKsize (int): Tamaño del kernel de desenfoque (por defecto 7, mínimo 3).
        edgeKsize (int): Tamaño del kernel para detección de bordes (por defecto 5).

    Nota: Si blurKsize < 3, no se aplica desenfoque.
    """
    # Aplicar desenfoque mediano si el kernel es >= 3 para reducir ruido
    if blurKsize >= 3:
        blurred = cv2.medianBlur(src, blurKsize)
        gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
    else:
        # Convertir directamente a gris sin desenfoque
        gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    # Aplicar Laplaciano para detectar bordes (valores altos en bordes)
    cv2.Laplacian(gray, cv2.CV_8U, gray, ksize=edgeKsize)
    # Invertir y normalizar para crear una máscara alfa (bordes oscuros)
    normalizedInverseAlpha = (1.0 / 255) * (255 - gray)
    # Separar canales de la imagen original
    channels = cv2.split(src)
    # Multiplicar cada canal por la máscara para oscurecer los bordes
    for ch in channels:
        ch[:] = ch * normalizedInverseAlpha
    # Combinar los canales modificados en dst
    cv2.merge(channels, dst)


# --- Filtros de convolución (basados en kernels) ---
class VConvolutionFilter(object):
    """
    Filtro base que aplica una convolución 2D usando un kernel personalizado.

    Utiliza cv2.filter2D para aplicar el kernel a la imagen.

    Atributos:
        _kernel (numpy.ndarray): Kernel de convolución.
    """

    def __init__(self, kernel):
        """
        Inicializa el filtro con un kernel.

        Args:
            kernel (numpy.ndarray): Matriz del kernel (ej. 3x3).
        """
        self._kernel = kernel

    def apply(self, src, dst):
        """
        Aplica la convolución a la imagen.

        Args:
            src (numpy.ndarray): Imagen de entrada.
            dst (numpy.ndarray): Imagen de salida.
        """
        # Aplicar filtro 2D con el kernel
        cv2.filter2D(src, -1, self._kernel, dst)


class SharpenFilter(VConvolutionFilter):
    """
    Filtro de nitidez que resalta detalles y bordes.

    Usa un kernel que aumenta el centro y reduce los alrededores.
    """

    def __init__(self):
        # Kernel de nitidez: resalta el píxel central
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        super().__init__(kernel)


class FindEdgesFilter(VConvolutionFilter):
    """
    Filtro para detectar bordes (similar a Sobel o Laplacian).

    Usa un kernel que resalta diferencias con los vecinos.
    """

    def __init__(self):
        # Kernel de detección de bordes
        kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
        super().__init__(kernel)


class BlurFilter(VConvolutionFilter):
    """
    Filtro de desenfoque que suaviza la imagen.

    Usa un kernel promedio (mean filter).
    """

    def __init__(self):
        # Kernel de desenfoque: promedio de 5x5
        kernel = np.ones((5, 5), dtype=np.float32) / 25.0
        super().__init__(kernel)


class EmbossFilter(VConvolutionFilter):
    """
    Filtro de relieve que crea un efecto 3D en los bordes.

    Usa un kernel que simula iluminación diagonal.
    """
    def __init__(self):
        # Kernel de relieve: resalta bordes con gradiente
        kernel = np.array([[-2,-1,0],[-1,1,1],[0,1,2]])
        super().__init__(kernel)
