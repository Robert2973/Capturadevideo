import numpy as np
import scipy.interpolate


def createCurveFunc(points):
    """
    Crea una función de interpolación a partir de puntos de control.

    Utiliza scipy.interpolate.interp1d para generar una función que interpola
    entre los puntos dados. Si hay menos de 2 puntos, retorna None.
    Usa interpolación cúbica si hay 4 o más puntos, lineal en caso contrario.

    Args:
        points (list of tuples or None): Lista de puntos (x, y) para la curva.
                                         Ej. [(0, 0), (128, 128), (255, 255)].

    Returns:
        callable or None: Función de interpolación o None si no se puede crear.

    Nota: La función extrapolada fuera de los límites para evitar errores.
    """
    if points is None:
        return None
    numPoints = len(points)
    if numPoints < 2:
        return None  # Necesita al menos 2 puntos para interpolar
    # Separar coordenadas x e y
    xs, ys = zip(*points)
    # Elegir tipo de interpolación: cúbica si >=4 puntos, lineal si menos
    kind = 'cubic' if numPoints >= 4 else 'linear'
    # Crear función de interpolación con extrapolación
    return scipy.interpolate.interp1d(xs, ys, kind=kind, bounds_error=False, fill_value="extrapolate")


def createLookupArray(func, length=256):
    """
    Crea un array de lookup (búsqueda) basado en una función.

    Evalúa la función en cada índice de 0 a length-1, clampando los valores
    al rango [0, length-1] para evitar desbordamientos.

    Args:
        func (callable or None): Función a evaluar (ej. de createCurveFunc).
        length (int): Longitud del array (por defecto 256, para 8 bits).

    Returns:
        numpy.ndarray or None: Array de lookup tipo uint8, o None si func es None.

    Nota: Útil para optimizar aplicaciones repetidas de funciones en imágenes.
    """
    if func is None:
        return None
    # Crear array vacío de floats para cálculos intermedios
    lookup = np.empty(length, dtype=np.float32)
    for i in range(length):
        # Evaluar la función en i
        v = float(func(i))
        # Clamp al rango válido
        if v < 0: v = 0
        if v > length - 1: v = length - 1
        lookup[i] = v
    # Convertir a uint8 para uso en imágenes
    return lookup.astype(np.uint8)


def applyLookupArray(lookupArray, src, dst):
    """
    Aplica un array de lookup a un array fuente, almacenando el resultado en dst.

    Usa indexación avanzada de NumPy para mapear eficientemente los valores.
    Si lookupArray es None, no hace nada (dst queda sin cambios).

    Args:
        lookupArray (numpy.ndarray or None): Array de lookup (tipo uint8).
        src (numpy.ndarray): Array fuente (ej. imagen plana).
        dst (numpy.ndarray): Array destino donde se almacena el resultado.

    Nota: Modifica dst in-place. Aprovecha broadcasting de NumPy.
    """
    if lookupArray is None:
        return
    # Aplicar lookup usando indexación: dst[i] = lookupArray[src[i]]
    # NumPy maneja broadcasting y tipos automáticamente
    dst[:] = lookupArray[src]


def flatView(array):
    """
    Crea una vista plana (1D) de un array multidimensional sin copiar datos.

    Cambia la forma del array a (array.size,), permitiendo acceso lineal
    mientras comparte memoria con el original.

    Args:
        array (numpy.ndarray): Array multidimensional (ej. imagen 3D).

    Returns:
        numpy.ndarray: Vista plana del array (1D).

    Nota: Cualquier cambio en la vista afecta al array original y viceversa.
    """
    # Crear vista del array (no copia)
    flat = array.view()
    # Cambiar forma a 1D con tamaño total
    flat.shape = array.size
    return flat
