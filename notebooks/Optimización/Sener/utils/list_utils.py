import numpy as np


def crear_nvector_like(vector_entrada: list) -> np.ndarray:
    """
    Convierte el vector de entrada a numpy
    """
    vector_nentrada = np.asarray(vector_entrada)

    return vector_nentrada


def crear_nvector_zeros_like(vector: list, dtype=float) -> np.ndarray:
    if dtype is bool:
        resultado = np.zeros_like(vector, dtype=bool)
    else:
        resultado = np.zeros_like(vector, dtype=float)

    return resultado


def identificar_posiciones_nozero(vector_entrada_np: np.ndarray) -> np.ndarray:
    """
    Devuelve la posición del primer elemento no cero en el vector de entrada.
    """

    indices_no_zero = np.flatnonzero(vector_entrada_np)
    if indices_no_zero.size > 0:
        return indices_no_zero
    else:
        return np.array([])


def identificar_primera_posicion_no_zero(vector_entrada_np: np.ndarray) -> int:
    if identificar_posiciones_nozero(vector_entrada_np).size == 0:
        return np.array([])
    return identificar_posiciones_nozero(vector_entrada_np)[0]


def identificar_primer_true_de_cadenas_trues(vector: list[bool]) -> list[bool]:
    """Detecta Trues:
    1. Primera posición
    2. Cambio de False a True
    """
    return [val and (i == 0 or not vector[i - 1]) for i, val in enumerate(vector)]


def expandir_vector(vector_entrada: list, n: int) -> list:
    """
    Expande un vector duplicando cada uno de sus elementos un número determinado de veces.

    Cada elemento del vector original se repite consecutivamente `n` veces en el nuevo vector.

    Parámetros:
    ----------
    vector_entrada : list
        Lista de entrada con los elementos originales.
    n : int
        Número de veces que debe repetirse cada elemento.

    Retorna:
    -------
    list
        Lista con los elementos expandidos.
    """
    return [x / n for elem in vector_entrada for x in [elem] * n]
