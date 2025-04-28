import json
import numpy as np
import pandas as pd

# Leer el archivo JSON
with open("./outputs/resultados_Q.json", "r") as archivo:
    datos = json.load(archivo)


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
        # Extraer datos del JSON
        G = json_data.get(
            "G", [{}]
        )[
            0
        ]  # Accede al primer elemento de la lista o usa un diccionario vacío por defecto
        C = json_data.get("C", [{}])[0]
        A = json_data.get("A", [{}])[0]
        R = json_data.get("R", [{}])[0]
        L = json_data.get("L", [])  # L puede ser una lista de diccionarios
        # Time = json_data.get("Time", {})  # time no lo uso

        # Extraer input_curves de L
        input_curves = {}
        precalentando = {}
        for i, link_data in enumerate(L):
            link_number = i + 1  # Asumiendo que los links se numeran desde 1
            # Divide la potencia de entrada (xI) entre 6 para volver a MWh
            # si existe la clave "xI" en link_data, de lo contrario asigna una lista vacía.
            input_curves[link_number] = [
                x / 6 if isinstance(x, (int, float)) else 0
                for x in link_data.get("xI", [])
            ]
            precalentando[link_number] = link_data.get("p", [])

        # Extraer G_result del campo "x" de G y dividir cada elemento por 6
        G_result = [x / 6 for x in G.get("x", [])]  # bien

        # Extraer R_result del campo "xI" de R y dividir cada elemento por 6
        RI = [x / 6 if isinstance(x, (int, float)) else 0 for x in R.get("xI", [])]
        RE = [x / 6 if isinstance(x, (int, float)) else 0 for x in R.get("xE", [])]

        RI_array = np.array(RI)
        RE_array = np.array(RE)

        R_result = (RI_array + RE_array).tolist()  # bien

        # Extraer Q_result del campo "C" de A
        Q_result = A.get("C", [])  # bien

        # Extraer A_result (calcular a partir de C del almacenamiento)
        # La lógica original parece calcularlo como la diferencia entre Q[t+1] y Q[t]
        Q_result_np = np.array(Q_result)
        A_result = np.diff(Q_result_np).tolist()
        A_result = [-x for x in A_result]  # Invertir el signo de A_result
        A_result = [
            0.0
        ] + A_result  # Agregar un 0 al inicio para mantener la longitud #bien

        # Reordenar los datos en la estructura vector_entrada_resultados
        vector_entrada_resultados = [
            input_curves,
            precalentando,
            G_result,
            R_result,
            Q_result,
            A_result,
        ]

        return vector_entrada_resultados
    except (KeyError, TypeError, IndexError) as e:
        print(f"Error al procesar el JSON: {e}")
        print(e)
        return None


def visualizar_vector_entrada_resultados(vector_entrada_resultados):
    """
    Presenta el vector_entrada_resultados de una manera más organizada y legible,
    utilizando pandas DataFrames para las series temporales y formateando la salida.

    Args:
        vector_entrada_resultados (list): La lista conteniendo los datos a visualizar.
    """

    if not vector_entrada_resultados or len(vector_entrada_resultados) != 6:
        print("Error: vector_entrada_resultados inválido o incompleto.")
        return

    input_curves, precalentando, G_result, R_result, Q_result, A_result = (
        vector_entrada_resultados
    )

    # Visualización de input_curves (Curvas de Consumo de Electricidad por Link)
    print("\n--- Curvas de Consumo de Electricidad por Link (input_curves) ---")
    df_input_curves = pd.DataFrame(input_curves)
    print(df_input_curves.to_string())  # Imprime el DataFrame completo

    # Visualización de precalentando (Estado de Precalentamiento por Link)
    print("\n--- Estado de Precalentamiento por Link (precalentando) ---")
    df_precalentando = pd.DataFrame(precalentando)
    print(df_precalentando.to_string())

    # Visualización de G_result (Energía del Generador)
    print("\n--- Energía del Generador (G_result) ---")
    df_G_result = pd.DataFrame({"G_result": G_result})
    print(df_G_result.to_string())

    # Visualización de R_result (Conexión a Red)
    print("\n--- Conexión a Red (R_result) ---")
    df_R_result = pd.DataFrame({"R_result": R_result})
    print(df_R_result.to_string())

    # Visualización de Q_result (Nivel de Carga de la Batería)
    print("\n--- Nivel de Carga de la Batería (Q_result) ---")
    df_Q_result = pd.DataFrame({"Q_result": Q_result})
    print(df_Q_result.to_string())

    # Visualización de A_result (Energía Obtenida de la Batería)
    print("\n--- Energía Obtenida de la Batería (A_result) ---")
    df_A_result = pd.DataFrame({"A_result": A_result})
    print(df_A_result.to_string())
