import numpy as np


def _crear_vectores_numpy(vector_entrada, dtype=float):
    """
    Convierte el vector de entrada a numpy y crea un vector de resultados (lleno de 0s) numpy de igual tamaño.
    tipo de dato por defecto float.
    """
    vector_nentrada = np.asarray(vector_entrada)
    if dtype is bool:
        resultado = np.zeros_like(vector_entrada, dtype=bool)
    else:
        resultado = np.zeros_like(vector_entrada, dtype=float)

    return vector_nentrada, resultado


def _identificar_posiciones_nozero(vector_entrada_np, primer_indice=False):
    """
    Devuelve la posición del primer elemento no cero en el vector de entrada.
    """

    indices_no_zero = np.flatnonzero(vector_entrada_np)
    if indices_no_zero.size > 0:
        if primer_indice:
            return indices_no_zero[0]
        else:
            return indices_no_zero
    else:
        return None  # Si no hay elementos no cero, devolver -1


def transference_function(energy_input, data, link=0, verbose=False):
    output = 0.0
    for i in range(len(data["L"][link]["T"]) - 1):
        treshold = float(data["L"][link]["T"][i + 1][1])
        conversion_rate = float(data["L"][link]["T"][i][0])
        if verbose:
            print(
                "If the input is smaller than",
                treshold,
                "then the conversion rate is",
                conversion_rate,
                ". Is it the case?",
            )
        if energy_input < float(treshold):
            output = energy_input * conversion_rate
            break
    treshold = data["L"][link]["T"][-1][1]
    if energy_input > treshold:
        conversion_rate = 0
        if verbose:
            print(
                "As the input is bigger than",
                treshold,
                "then the conversion rate is",
                conversion_rate,
            )
        output = energy_input * conversion_rate
    return output


####################################################################################

##G


def calcular_coste(vector_entrada, coste, solo_primer_dato=False, reemplazar=False):
    """
    Calcula un vector de costes a partir de un vector de entrada binario o numérico.

    La función identifica los valores no nulos en el vector de entrada y asigna un coste en esas posiciones.
    El tipo parámetro `coste` determinara el tipo del vector de resultados.

    Flags:
    - `solo_primer_dato`: Si es True, solo se considera la primera aparición de una posición activa (no cero) en el vector.
    - `reemplazar`: Si es True, se asigna directamente el valor de `coste` en las posiciones activas.


    Parameters
    ----------
    vector_entrada : list or array-like
        Vector de entrada que representa los datos a transformar.

    coste : float or bool
        Coste a aplicar. Puede ser un valor numérico (coste monetario) o un booleano para marcar actividad.

    solo_primer_dato : bool, optional
        Si es True, solo se considera la primera aparición de una posición activa (no cero) en el vector.

    reemplazar : bool, optional
        Si es True, se asigna directamente el valor de `coste` en las posiciones activas.
        Si es False, se multiplica el valor del vector de entrada por `coste`.

    Returns
    -------
    list
        Vector resultante con los costes calculados, en formato lista (compatible con JSON).

    """

    is_boolean = isinstance(coste, bool)
    dtype = type(coste)

    nvector, resultado = _crear_vectores_numpy(vector_entrada, dtype=dtype)
    indices = _identificar_posiciones_nozero(nvector, primer_indice=solo_primer_dato)

    if indices is None or indices.size == 0:
        return resultado.tolist()

    # Coste escalar
    if np.isscalar(coste):
        if is_boolean:
            resultado[indices] = True
        else:
            resultado[indices] = coste if reemplazar else nvector[indices] * coste

    # Coste vectorial
    else:
        coste_array = np.asarray(coste)
        if coste_array.shape != nvector.shape:
            raise ValueError(
                "El vector `coste` debe tener la misma longitud que `vector_entrada`."
            )

        resultado[indices] = (
            coste_array[indices]
            if reemplazar
            else nvector[indices] * coste_array[indices]
        )

    return resultado.tolist()


## C


def vector_a_numpy(vector_entrada, dtype=float):
    """
    Convierte un vector de entrada a un array numpy del tipo especificado.

    Parameters
    ----------
    vector_entrada : list or array-like
        Vector de entrada que se desea convertir.

    dtype : type, optional
        Tipo de dato para el array numpy resultante. Por defecto es float.

    Returns
    -------
    np.ndarray
        Array numpy con los elementos del vector de entrada, convertido al tipo especificado.
    """
    return np.asarray(vector_entrada, dtype=dtype)


def calcular_desvios(vector_oferta, vector_demanda, deficit=False):
    nvector_oferta = vector_a_numpy(vector_oferta)
    nvector_demanda = vector_a_numpy(vector_demanda)

    if deficit:
        resultado = np.where(
            nvector_demanda > nvector_oferta, nvector_demanda - nvector_oferta, 0.0
        )
        return resultado.tolist()
    else:
        resultado = np.where(
            nvector_oferta > nvector_demanda, nvector_oferta - nvector_demanda, 0.0
        )
        return resultado.tolist()


def expandir_vector(vector_entrada, n):
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


def eliminar_energias_precalentamiento(valores, flags):
    """
    Sustituye los valores por 0 si el flag correspondiente es True.
    Soporta tanto entrada en forma de lista como de diccionario de listas.

    Parameters
    ----------
    valores : list[float] or dict[int, list[float]]
        Lista o diccionario con listas de valores (por ejemplo, curva de entrada).

    flags : list[bool] or dict[int, list[bool]]
        Lista o diccionario con listas de flags booleanos.

    Returns
    -------
    list[float] or dict[int, list[float]]
        Misma estructura que `valores`, pero con los valores puestos a 0 si el flag era True.
    """
    if isinstance(valores, dict) and isinstance(flags, dict):
        return {
            k: [0 if flag else valor for valor, flag in zip(valores[k], flags[k])]
            for k in valores
        }
    elif isinstance(valores, list) and isinstance(flags, list):
        return [0 if flag else valor for valor, flag in zip(valores, flags)]
    else:
        raise TypeError("Las entradas deben ser ambas listas o ambos diccionarios.")


## A


def detectar_y_añadir_primer_true(vector):
    return [val and (i == 0 or not vector[i - 1]) for i, val in enumerate(vector)]


def detectar_fin_precalentamiento(vector):
    """
    Detecta los cambios de True a False y pone True desde ese punto en adelante.

    Parameters
    ----------
    vector : list[bool]
        Vector booleano de entrada.

    Returns
    -------
    list[bool]
        Vector booleano con True a partir del primer cambio de True a False.
    """
    resultado = [False] * len(vector)
    activar = False

    for i in range(len(vector)):
        if i > 0 and vector[i - 1] and not vector[i]:
            activar = True
        if activar:
            resultado[i] = True

    return resultado


def sumar_costes_totales(diccionario):
    """
    Suma todos los valores asociados a la clave 'cTot', ya sea directamente en el diccionario
    o en sus subdiccionarios. También devuelve un diccionario con los costes individuales.

    Parameters
    ----------
    diccionario : dict or list
        Diccionario o lista que puede contener directamente 'cTot' o incluirla en subdiccionarios.

    Returns
    -------
    tuple
        (float, dict) -> Suma total y diccionario con los costes individuales por clave o índice.
    """
    total = 0.0
    costes_individuales = {}

    if isinstance(diccionario, dict):
        if "cTot" in diccionario and isinstance(diccionario["cTot"], list):
            subtotal = sum(diccionario["cTot"])
            total += subtotal
            costes_individuales["cTot"] = subtotal

        for clave, valor in diccionario.items():
            if (
                isinstance(valor, dict)
                and "cTot" in valor
                and isinstance(valor["cTot"], list)
            ):
                subtotal = sum(valor["cTot"])
                total += subtotal
                costes_individuales[clave] = subtotal

    elif isinstance(diccionario, list):
        for idx, item in enumerate(diccionario):
            if (
                isinstance(item, dict)
                and "cTot" in item
                and isinstance(item["cTot"], list)
            ):
                subtotal = sum(item["cTot"])
                total += subtotal
                costes_individuales[idx] = subtotal

    return total, costes_individuales
