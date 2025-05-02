import numpy as np


def crear_nvector_like(vector_entrada: list) -> np.ndarray:
    """Convierte una lista en un array de NumPy usando la misma referencia de memoria, no crea una copia.

    Args:
        vector_entrada (list): La lista de entrada que se va a convertir en un array de NumPy.

    Returns:
        np.ndarray: Un array de NumPy creado a partir de la lista de entrada.
    """
    vector_nentrada = np.asarray(vector_entrada)

    return vector_nentrada


def crear_nvector_zeros_like(vector: list, dtype=float) -> np.ndarray:
    """Crea un array de NumPy lleno de ceros con la misma forma y tipo de datos que el vector dado.

    Args:
        vector (list): El vector de entrada que define la forma del array de ceros.
        dtype (type, optional): El tipo de datos deseado para el array de ceros.  De manera predeterminada, es float.  Si es bool, el array será booleano.

    Returns:
        np.ndarray: Un array de NumPy lleno de ceros con la misma forma y tipo de datos que el vector de entrada.
    """
    if dtype is bool:
        resultado = np.zeros_like(vector, dtype=bool)
    else:
        resultado = np.zeros_like(vector, dtype=float)

    return resultado


def identificar_posiciones_nozero(vector_entrada_np: np.ndarray) -> np.ndarray:
    """Encuentra los índices de los elementos no nulos en un array de NumPy.

    Args:
        vector_entrada_np (np.ndarray): El array de NumPy de entrada.

    Returns:
        np.ndarray: Un array de NumPy que contiene los índices de los elementos no nulos en el array de entrada.
                      Si no hay elementos no nulos, devuelve un array vacío.
    """

    return np.flatnonzero(vector_entrada_np)


def identificar_primera_posicion_no_zero(vector_entrada_np: np.ndarray) -> int:
    """Encuentra el índice del primer elemento no nulo en un array de NumPy.

    Args:
        vector_entrada_np (np.ndarray): El array de NumPy de entrada.

    Returns:
        int: El índice del primer elemento no nulo en el array de entrada.
              Si no hay elementos no nulos, devuelve -1.
    """
    indices = identificar_posiciones_nozero(vector_entrada_np)
    return int(indices[0]) if indices.size > 0 else -1


def identificar_primer_true_de_cadenas_trues(vector: list[bool]) -> np.ndarray:
    """Identifica el primer `True` en una lista de booleanos, considerando también el caso en que un `True` sea precedido por un `False` usando NumPy.

    Args:
        vector (list[bool]): La lista de booleanos de entrada.

    Returns:
        np.ndarray: Un array de booleanos donde `True` indica la primera ocurrencia de `True` en la lista de entrada o un cambio de `False` a `True`.
    """
    vector_np = crear_nvector_like(vector)
    vector_desplazado = np.concatenate(
        ([False], vector_np[:-1])
    )  # Desplaza el vector a la derecha, rellenando con False al principio
    result_np = (
        vector_np & ~vector_desplazado
    )  # Realiza un AND entre el vector original y el NOT del vector desplazado.

    return result_np.tolist()


def expandir_vector(vector_entrada: list, n: int) -> list:
    """
    Expande un vector duplicando cada uno de sus elementos un número determinado de veces.

    Cada elemento del vector original se repite consecutivamente `n` veces en el nuevo vector.
    Ajusta el valor de todos los elementos divideindolos entre n.

    Args:
        vector_entrada (list): Lista de entrada con los elementos originales.
        n (int): Número de veces que debe repetirse cada elemento.

    Returns:
        list: Lista con los elementos expandidos.
    """
    return [x / n for elem in vector_entrada for x in [elem] * n]
