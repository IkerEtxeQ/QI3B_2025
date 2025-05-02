import numpy as np
import utils.list_utils as list_utils


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


def crear_lista_booleana_primer_valor_mayor_zero_true(vector: list) -> list:
    nvector = list_utils.crear_nvector_like(vector)
    resultado = list_utils.crear_nvector_zeros_like(vector, dtype=bool)

    primer_indice_no_zero = list_utils.identificar_primera_posicion_no_zero(nvector)

    if primer_indice_no_zero != -1:
        resultado[primer_indice_no_zero] = True

    return resultado.tolist()


def crear_lista_booleana_todos_valores_mayor_zero_true(vector: list) -> list:
    nvector = list_utils.crear_nvector_like(vector)
    resultado = list_utils.crear_nvector_zeros_like(vector, dtype=bool)

    indices_no_zero = list_utils.identificar_posiciones_nozero(nvector)

    if indices_no_zero.size > 0:
        resultado[indices_no_zero] = True

    return resultado.tolist()


def calcular_coste_arranque(vector: list, coste: float) -> list:
    nvector = list_utils.crear_nvector_like(vector)
    resultado = list_utils.crear_nvector_zeros_like(vector, dtype=float)

    primer_indice_no_zero = list_utils.identificar_primera_posicion_no_zero(nvector)

    if primer_indice_no_zero != -1:
        resultado[primer_indice_no_zero] = coste

    return resultado.tolist()


def calcular_coste_fijo(vector: list, coste: float) -> list:
    nvector = list_utils.crear_nvector_like(vector)
    resultado = list_utils.crear_nvector_zeros_like(vector, dtype=float)

    indices_no_zero = list_utils.identificar_posiciones_nozero(nvector)

    if indices_no_zero.size > 0:
        resultado[indices_no_zero] = coste

    return resultado.tolist()


def calcular_coste_variable(vector: list, coste: list) -> list:
    nvector = list_utils.crear_nvector_like(vector)
    resultado = list_utils.crear_nvector_zeros_like(vector, dtype=float)
    if np.isscalar(coste):
        ncoste = np.full(len(vector), coste)
    else:
        ncoste = list_utils.crear_nvector_like(coste)

    indices_no_zero = list_utils.identificar_posiciones_nozero(nvector)

    if ncoste.shape != nvector.shape:
        raise ValueError(
            "El vector `coste` debe tener la misma longitud que `vector_entrada`."
        )

    if indices_no_zero.size > 0:
        resultado[indices_no_zero] = nvector[indices_no_zero] * ncoste[indices_no_zero]

    return resultado.tolist()


def calcular_deficit(vector_oferta: list, vector_demanda: list) -> list:
    nvector_oferta = list_utils.crear_nvector_like(vector_oferta)
    nvector_demanda = list_utils.crear_nvector_like(vector_demanda)

    resultado = np.where(
        nvector_demanda > nvector_oferta, nvector_demanda - nvector_oferta, 0.0
    )

    return resultado.tolist()


def calcular_exceso(vector_oferta: list, vector_demanda: list) -> list:
    nvector_oferta = list_utils.crear_nvector_like(vector_oferta)
    nvector_demanda = list_utils.crear_nvector_like(vector_demanda)

    resultado = np.where(
        nvector_oferta > nvector_demanda, nvector_oferta - nvector_demanda, 0.0
    )

    return resultado.tolist()


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


def detectar_fin_precalentamiento(vector_precalentamiento: list[bool]):
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
    resultado = [False] * len(vector_precalentamiento)
    activar = False

    for i in range(len(vector_precalentamiento)):
        if i > 0 and vector_precalentamiento[i - 1] and not vector_precalentamiento[i]:
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
