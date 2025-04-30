import numpy as np


def __extraer_datos(data: dict) -> dict:
    datos_a_extraer = ["G", "A", "R", "L"]

    return {
        x: data.get(x, [{}])[0] if x != "L" else data.get(x, [])
        for x in datos_a_extraer
    }


def __asignar_energias_preaclentamiento(xI: list, Eprec: int) -> list:
    try:
        first_pos_index = next(j for j, val in enumerate(xI) if val > 0)
        for j in range(max(0, first_pos_index - 7), first_pos_index):
            xI[j] = Eprec
    except StopIteration:
        pass  # No hay valores positivos, ignora excepcion


def __extraer_input_curves(L: list) -> dict:
    input_curves = {}
    E_prec = 0.30281338645129247

    for i, link_data in enumerate(L):
        link_number = i + 1  # Asumiendo que los links se numeran desde 1
        xI = [x if isinstance(x, (int, float)) else 0 for x in link_data.get("xI", [])]
        __asignar_energias_preaclentamiento(xI, E_prec)
        input_curves[link_number] = xI

    return input_curves


def __extraer_precalentamiento(L: list) -> dict:
    precalentando = {}

    for i, link_data in enumerate(L):
        link_number = i + 1  # Asumiendo que los links se numeran desde 1
        precalentando[link_number] = link_data.get("p", [])

    return precalentando


def __extraer_R(R: list) -> list:
    RI = [x / 6 if isinstance(x, (int, float)) else 0 for x in R.get("xI", [])]
    RE = [x / 6 if isinstance(x, (int, float)) else 0 for x in R.get("xE", [])]

    RI_array = np.array(RI)
    RE_array = np.array(RE)

    return (RI_array + RE_array).tolist()  # bien


def json_to_vector_entrada_resultados(json_data):
    """
    Convierte un diccionario JSON, generado por el script original,
    de vuelta a la lista 'vector_entrada_resultados'.

    Args:
        json_data (dict): El diccionario JSON cargado desde el archivo.

    Returns:
        list: Una lista conteniendo los datos extraídos del JSON en el
              mismo formato que 'vector_entrada_resultados'.  Retorna None si
              la estructura del json_data no es la esperada.
    """

    try:
        dict_datos = __extraer_datos(json_data)

        input_curves = __extraer_input_curves(dict_datos.get("L", {}))
        precalentando = __extraer_precalentamiento(dict_datos.get("L", {}))

        # Extraer G_result del campo "x" de G y dividir cada elemento por 6
        G_result = [x / 6 for x in dict_datos["G"].get("x", [])]  # bien

        R_result = __extraer_R(dict_datos.get("R", []))

        # Extraer Q_result del campo "C" de A
        Q_result = dict_datos["A"].get("C", [])  # bien

        A_result = [
            -c / 6 + d / 6
            for c, d in zip(
                dict_datos["A"].get("xC", []), dict_datos["A"].get("xD", [])
            )
        ]

        vector_entrada_resultados = [
            input_curves,
            precalentando,
            G_result,
            Q_result,
            A_result,
            R_result,
        ]

        return vector_entrada_resultados
    except (KeyError, TypeError, IndexError) as e:
        print(f"Error al procesar el JSON: {e}")
        print(e)
        return None
