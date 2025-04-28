import json
import numpy as numpy
import pandas as pd
import matplotlib.pyplot as plt

import result_utils
import os

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
        A = json_data.get("A", [{}])[0]
        R = json_data.get("R", [{}])[0]
        L = json_data.get("L", [])  # L puede ser una lista de diccionarios
        # Time = json_data.get("Time", {})  # time no lo uso

        # Extraer input_curves de L
        input_curves = {}
        precalentando = {}
        E_prec = 0.30281338645129247  # Escalar a añadir

        for i, link_data in enumerate(L):
            link_number = i + 1  # Asumiendo que los links se numeran desde 1

            # Divide la potencia de entrada (xI) entre 6 para volver a MWh
            xI = [
                x if isinstance(x, (int, float)) else 0 for x in link_data.get("xI", [])
            ]

            # Añadir escalar a los 7 elementos anteriores al primer valor positivo
            try:
                first_pos_index = next(j for j, val in enumerate(xI) if val > 0)
                for j in range(max(0, first_pos_index - 7), first_pos_index):
                    xI[j] = E_prec
            except StopIteration:
                pass  # No hay valores positivos, no hacer nada

            input_curves[link_number] = xI
            precalentando[link_number] = link_data.get("p", [])

        # Extraer G_result del campo "x" de G y dividir cada elemento por 6
        G_result = [x / 6 for x in G.get("x", [])]  # bien

        # Extraer R_result del campo "xI" de R y dividir cada elemento por 6
        RI = [x / 6 if isinstance(x, (int, float)) else 0 for x in R.get("xI", [])]
        RE = [x / 6 if isinstance(x, (int, float)) else 0 for x in R.get("xE", [])]

        RI_array = numpy.array(RI)
        RE_array = numpy.array(RE)

        R_result = (RI_array + RE_array).tolist()  # bien

        # Extraer Q_result del campo "C" de A
        Q_result = A.get("C", [])  # bien

        # Extraer A_result (calcular a partir de C del almacenamiento)
        # La lógica original parece calcularlo como la diferencia entre Q[t+1] y Q[t]
        Q_result_np = numpy.array(Q_result)
        A_result = numpy.diff(Q_result_np).tolist()
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


with open(
    "./Inputs/IPCEI-Cuantica_kickoff_datosSENER_250228/ejemplo_sener/resultados.json",
    "r",
    encoding="utf-8",
) as f:
    data = json.load(f)

vector_resultados_C = json_to_vector_entrada_resultados(data)

if vector_resultados_C:
    print("vector_entrada_resultados recuperado con éxito:")
    visualizar_vector_entrada_resultados(vector_resultados_C)
else:
    print("No se pudo recuperar vector_entrada_resultados.")


try:
    with open(
        "./outputs/resultados_Q.json",
        "r",
        encoding="utf-8",
    ) as f:
        data = json.load(f)

    vector_resultados_Q = json_to_vector_entrada_resultados(data)

    if vector_resultados_Q:
        print("vector_entrada_resultados recuperado con éxito:")
        visualizar_vector_entrada_resultados(vector_resultados_Q)
    else:
        print("No se pudo recuperar vector_entrada_resultados.")

except FileNotFoundError:
    print("Error: El archivo 'resultados_Q.json' no se encontró.")
except json.JSONDecodeError:
    print("Error: El archivo 'resultados_Q.json' contiene JSON inválido.")
except Exception as e:
    print(f"Ocurrió un error inesperado: {e}")


##TODO: Crear funcion para generar el json de salida a partir del vector con los resultados.
##TODO: Mirar lo de introducir un 0 en la primera posicion de A_result


def vector_entrada_to_Json(vector_entrada, data, nombre_archivo="resultados_Q.json"):
    # Leer el archivo JSON
    with open(
        "inputs/IPCEI-Cuantica_kickoff_datosSENER_250228/ejemplo_sener/datos.json", "r"
    ) as archivo:
        datos = json.load(archivo)

    datos_G = datos["G"][0]
    datos_C = datos["C"][0]
    datos_A = datos["A"][0]
    datos_R = datos["R"][0]

    ## G: Vector que contiene los resultados de cada Generador existente.

    G_result = vector_entrada[
        2
    ]  # G_result = energía obtenida (MWh) del generador G (en este caso, energía solar). Para pasar a potencia (MW), multiplicar por 6 (en realidad: dividir entre 1h/6pasos)

    # G_result = energía obtenida (MWh) del generador G (en este caso, energía solar). Para pasar a potencia (MW), multiplicar por 6 (en realidad: dividir entre 1h/6pasos)
    G_claves = ["x", "y", "z", "cA", "cF", "cV", "cTot", "name"]

    # Nombre
    G_name = "PV"

    # Vector de Potencia [MW] Demandada en cada instante de tiempo
    g_x = [
        x * 6 for x in G_result
    ]  # Supongo que es la energia obtenida del generador #Conversión a MW = MWh * 6pasos/1h

    # Vector de booleanos que indican si el Generador está produciendo (=True) en cada instante de tiempo
    g_y = result_utils.calcular_coste(g_x, True)

    # Vector de booleanos que indican si el Generador se arrancó (=True) en cada instante de tiempo
    g_z = result_utils.calcular_coste(g_y, True, solo_primer_dato=True)

    # Vector de Costes [EUR] de Arranque en cada instante de tiempo
    cA = result_utils.calcular_coste(
        g_x, datos_G.get("Ca", 0), solo_primer_dato=True, reemplazar=True
    )

    # Vector de Costes [EUR] Fijos por estar en Producción en cada instante de tiempo
    cF = result_utils.calcular_coste(
        g_x, datos_G.get("Cf", 0) / 6, reemplazar=True
    )  # Conversión a EUR = EUR/h (cf) * 1h/6pasos

    # Vector de Costes [EUR] Variables en función de la producción en cada instante de tiempo
    cV = result_utils.calcular_coste(
        G_result, datos_G.get("Cv", 0)
    )  # Conversión a EUR =  EUR/MWh (cv) * MWh (G_result)

    # Vector de Costes [EUR] Totales en cada instante de tiempo
    cTot = [
        ca + cf + cv for ca, cf, cv in zip(cA, cF, cV)
    ]  # Arranque + coste fijo + coste variable

    G_valores = [g_x, g_y, g_z, cA, cF, cV, cTot, G_name]
    G = dict(zip(G_claves, G_valores))

    ## C: Vector que contiene los resultados de cada Consumidor existente -> Si es consumidor consume H2????

    input_curves = vector_entrada[0]
    precalentando = vector_entrada[1]

    # Quitar energias de precalentamiento
    input_curves_sin_precalentamiento = result_utils.eliminar_energias_precalentamiento(
        input_curves, precalentando
    )

    ## Aplicar función de transferencia
    H2_Transferido = {
        k: [result_utils.transference_function(x, datos, k - 1) for x in v]
        for k, v in input_curves_sin_precalentamiento.items()
    }
    suma_links_h = list(map(sum, zip(*H2_Transferido.values())))  # kg(H2)/h
    suma_links = [x / 6 for x in suma_links_h]  # Conversión a Kg = kg(H2)/h * 1h/6pasos

    ############ VECTORES AUXILIARES ################
    ### Expandir vectores de Potencia y Costes a 10 minutos (6 pasos de 10 minutos) ###
    vector_demanda = datos_C["Consumo"].get("Pot")  # MW/h
    vector_demanda_expandido_H = [x for elem in vector_demanda for x in [elem] * 6]
    vector_demanda_expandido = [x / 6 for x in vector_demanda_expandido_H]  # MW (kg)

    vector_Cc_d = datos_C["Costes"].get("Cc_d")  # EUR/MWh
    vector_Cc_d_expandido = result_utils.expandir_vector(
        vector_Cc_d, 6
    )  # EUR/MW (EUR/Kg)

    vector_Cc_e = datos_C["Costes"].get("Cc_e")  # EUR/MWh
    vector_Cc_e_expandido = result_utils.expandir_vector(
        vector_Cc_e, 6
    )  # EUR/MW (EUR/Kg)

    ###############
    C_claves = ["x", "d", "e", "y", "cD", "cE", "cTot", "Ereal", "REreal", "name"]

    # Nombre
    name = "H2"

    # Vector de Potencia [MW = kg/h] Recibida en cada instante de tiempo
    c_x = suma_links_h  # Kg

    # Vector de Potencia [MW] Recibida de menos (Déficit) respecto de la Solicitada en cada instante de tiempo
    c_d = result_utils.calcular_desvios(
        suma_links_h, vector_demanda_expandido_H, deficit=True
    )  # Kg

    # Vector de Potencia [MW] Recibida de más (Exceso) respecto de la Solicitada en cada instante de tiempo
    c_e = result_utils.calcular_desvios(suma_links_h, vector_demanda_expandido_H)  # Kg

    # Vector de booleanos que indican si el Consumidor está consumiendo (=True) en cada instante de tiempo
    c_y = result_utils.calcular_coste(
        suma_links_h, True
    )  ## Supongo que si producimos H2, se esta consumiendo.

    # Vector de Costes [EUR] generados por el Déficit de potencia suministrada respecto a solicitada en cada instante de tiempo
    c_cD = result_utils.calcular_coste(c_d, vector_Cc_d_expandido)  # EUR

    # Vector de Costes [EUR] generados por el Exceso de potencia suministrada respecto a solicitada en cada instante de tiempo
    c_cE = result_utils.calcular_coste(c_e, vector_Cc_e_expandido)  # EUR

    # Vector de Costes [EUR] Totales por desvíos en cada instante de tiempo
    c_cTot = [c_cd + c_ce for c_cd, c_ce in zip(c_cD, c_cE)]  # EUR

    # Energía Total [MWh] suministrada al Consumidor durante todos los instantes de tiempo
    c_Ereal = numpy.sum(suma_links)  # Kg

    # Ratio [-] entre la Energía Total suministrada / demandada (*100??)
    c_REreal = c_Ereal / numpy.sum(vector_demanda_expandido)

    C_valores = [c_x, c_d, c_e, c_y, c_cD, c_cE, c_cTot, c_Ereal, c_REreal, name]
    C = dict(zip(C_claves, C_valores))

    ### Vector que contiene los resultados de cada Almacenamiento existente
    Q_result = vector_entrada[4]  # Q (MWh)
    A_result = vector_entrada[5]  # A (MWh)

    #### VECTORES AUXILIARES ####
    aQ = [
        x * 6 for x in Q_result
    ]  # Vector de Energía [MW] que hay en el Almacenamiento al final de cada instante de tiempo
    xCh = [
        -x if x < 0 else 0 for x in A_result
    ]  # Vector de Potencia [MWh] Cargada en cada instante de tiempo
    xDh = [
        x if x > 0 else 0 for x in A_result
    ]  # Vector de Potencia [MWh] Descargada en cada instante de tiempo

    #############################
    A_claves = [
        "xC",
        "xD",
        "C",
        "yC",
        "zC",
        "yD",
        "zD",
        "cAC",
        "cFC",
        "cVC",
        "cAD",
        "cFD",
        "cVD",
        "cNm",
        "cNFinal",
        "cTot",
        "name",
    ]

    # Nombre
    name = "BESS"

    # Vector de Potencia [MW] Cargada en cada instante de tiempo
    xC = [-x * 6 if x < 0 else 0 for x in A_result]  # MW

    # Vector de Potencia [MW] Descargada en cada instante de tiempo
    xD = [x * 6 if x > 0 else 0 for x in A_result]  # MW

    # Vector de Energía [MWh] que hay en el Almacenamiento al final de cada instante de tiempo
    a_C = Q_result  # MWh

    # Vector de booleanos que indican si el Almacenamiento está Cargando (=True) en cada instante de tiempo
    yC = [x < 0 for x in A_result]

    # Vector de booleanos que indican si el Generador?? inició Carga (=True) en cada instante de tiempo
    zC = result_utils.detectar_y_añadir_primer_true(
        yC
    )  # detecta cada vez empieza una carga, no solo la primera

    # Vector de booleanos que indican si el Almacenamiento está Descargando (=True) en cada instante de tiempo
    yD = [x > 0 for x in A_result]

    # Vector de booleanos que indican si el Generador??  inició Carga (=True) en cada instante de tiempo
    zD = result_utils.detectar_y_añadir_primer_true(
        yD
    )  # detecta cada vez empieza una descarga, no solo la primera

    # Vector de Costes [EUR] por arrancar/iniciar Carga en cada instante de tiempo
    cAC = result_utils.calcular_coste(zC, datos_A.get("CaC", 0), reemplazar=True)

    # Vector de Costes [EUR] Fijos por estar en Carga en cada instante de tiempo
    cFC = result_utils.calcular_coste(
        xC, datos_A.get("CfC", 0) / 6, reemplazar=True
    )  # Conversión a EUR = EUR/h (CfC) * 1h/6pasos

    # Vector de Costes [EUR] Variables en función de la potencia Cargada en cada instante de tiempo
    cVC = result_utils.calcular_coste(
        xCh, datos_A.get("CvC", 0)
    )  # Conversión a EUR =  EUR/MWh (CvC) * MWh (xCh)

    # Vector de Costes [EUR] por arrancar/iniciar Descarga en cada instante de tiempo
    cAD = result_utils.calcular_coste(zD, datos_A.get("CaD", 0), reemplazar=True)

    # Vector de Costes [EUR] Fijos por estar en Descarga en cada instante de tiempo
    cFD = result_utils.calcular_coste(
        xD, datos_A.get("CfD", 0) / 6, reemplazar=True
    )  # Conversión a EUR = EUR/h (CfC) * 1h/6pasos

    # Vector de Costes [EUR] Variables en función de la potencia Descarga en cada instante de tiempo
    cVD = result_utils.calcular_coste(
        xDh, datos_A.get("CvD", 0)
    )  # Conversión a EUR =  EUR/MWh (CvC) * MWh (xDh)

    # Vector de Costes/valor [EUR] por mantener un determinado Nivel almacenado en cada instante de tiempo
    cNm = result_utils.calcular_coste(
        aQ, datos_A.get("CaNm", 0)
    )  # not required #Conversión a EUR =  EUR/MW (CaNm) * MW (aQ)

    cNd = []  # not required # no hay info
    cNe = []  # not required # no hay info

    # Vector de Costes/valor [EUR] del Nivel FINAL remanente en el almacenamiento
    cNFinal = [
        Q_result[-1] * datos_A.get("CaN", 0) if i == len(Q_result) - 1 else 0
        for i in range(len(Q_result))
    ]  # not required

    # Vector de Costes [EUR] Totales en cada instante de tiempo
    cTot = [
        cac + cfc + cvc + cad + cfd + cvd + cnfinal
        for cac, cfc, cvc, cad, cfd, cvd, cnfinal in zip(
            cAC, cFC, cVC, cAD, cFD, cVD, cNFinal
        )
    ]  # EUR
    ## arranques carga + fijos carga + variables carga + arranques descarga + fijos descarga + variables descarga

    A_valores = [
        xC,
        xD,
        a_C,
        yC,
        zC,
        yD,
        zD,
        cAC,
        cFC,
        cVC,
        cAD,
        cFD,
        cVD,
        cNm,
        cNFinal,
        cTot,
        name,
    ]
    A = dict(zip(A_claves, A_valores))

    ## R (MWh): Vector que contiene los resultados de cada conexión a Red existente

    R_result = vector_entrada[3]

    #### VECTORES AUXILIARES ####
    vector_potencia_comprometida = datos_R.get("Compromisos", {}).get(
        "Pot", numpy.zeros(144).tolist()
    )  # MW
    vector_potencia_comprometida_exportacion = [
        x if x > 0 else 0 for x in vector_potencia_comprometida
    ]  # MW
    vector_potencia_comprometida_importacion = [
        x if x < 0 else 0 for x in vector_potencia_comprometida
    ]  # MW

    vector_Crd_E = datos_R["Costes"].get("CRd_E", 0)  # EUR/MWh
    vector_Crd_E_expandido = result_utils.expandir_vector(vector_Crd_E, 6)  # EUR/MW

    vector_Ced_E = datos_R["Costes"].get("CRe_E", 0)  # EUR/MWh
    vector_Ced_E_expandido = result_utils.expandir_vector(vector_Ced_E, 6)  # EUR/MW

    vector_Crd_I = datos_R["Costes"].get("CRd_I", 0)  # EUR/MWh
    vector_Crd_I_expandido = result_utils.expandir_vector(vector_Crd_I, 6)  # EUR/MW

    vector_Ced_I = datos_R["Costes"].get("CRe_I", 0)  # EUR/MWh
    vector_Ced_I_expandido = result_utils.expandir_vector(vector_Ced_I, 6)  # EUR/MW

    ######################
    R_claves = [
        "xE",
        "xI",
        "dE",
        "eE",
        "dI",
        "eI",
        "yE",
        "yI",
        "xIextraAvail",
        "xIextraUsed",
        "xBsubir",
        "xBbajar",
        "cDE",
        "cEE",
        "cDI",
        "cEI",
        "cIextra",
        "cTot",
        "name",
    ]

    # Nombre
    name = "Red"

    # Vector de Potencia [MW] Exportada hacia la Red en cada instante de tiempo
    xE = [x * 6 if x < 0 else 0 for x in R_result]  ## MW

    # Vector de Potencia [MW] Importada desde la Red en cada instante de tiempo
    xI = [x * 6 if x > 0 else 0 for x in R_result]  ## MW

    # Vector de Potencia [MW] Exportada de menos (Déficit) respecto de la Comprometida, en cada instante de tiempo
    dE = result_utils.calcular_desvios(
        xE, vector_potencia_comprometida_exportacion, deficit=True
    )  # MW

    # Vector de Potencia [MW] Exportada de más (Exceso) respecto de la Comprometida, en cada instante de tiempo
    eE = result_utils.calcular_desvios(
        xE, vector_potencia_comprometida_exportacion
    )  # MW

    # Vector de Potencia [MW] Importada de menos (Déficit) respecto de la Comprometida, en cada instante de tiempo
    dI = result_utils.calcular_desvios(
        xI, vector_potencia_comprometida_importacion, deficit=True
    )  # MW

    # Vector de Potencia [MW] Importada de más (Exceso) respecto de la Comprometida, en cada instante de tiempo
    eI = result_utils.calcular_desvios(
        xI, vector_potencia_comprometida_importacion
    )  # MW

    # Vector de booleanos que indican si se está Exportando hacia la Red (=True) en cada instante de tiempo
    yE = result_utils.calcular_coste(xE, True)

    # Vector de booleanos que indican si se está Importando desde la Red (=True) en cada instante de tiempo
    yI = result_utils.calcular_coste(xI, True)

    ##Vector de Potencia [MW] Importación extra, por encima de la PmaxI/contratada, a la que se ha recurrido en cada instante de tiempo
    xIextraAvail = result_utils.calcular_desvios(xI, datos_R.get("PmaxI", 0))  # MW

    # Vector de Potencia [MW] Importación extra, por encima de la PmaxI/contratada, que se ha consumido en cada instante de tiempo
    xIextraUsed = result_utils.calcular_desvios(
        xI, datos_R.get("PmaxI", 0)
    )  ## igual que el anterior??

    # Vector de Potencia [MW] de Banda a Subir ofertada y adjudicada en cada instante de tiempo
    xBsubir = []

    # Vector de Potencia [MW] de Banda a Bajar ofertada y adjudicada en cada instante de tiempo
    xBbajar = []

    # Vector de Costes [EUR] generados por el Déficit de potencia Exportada respecto de la comprometida, en cada instante de tiempo
    cDE = result_utils.calcular_coste(dE, vector_Crd_E_expandido)  ##EUR

    # Vector de Costes [EUR] generados por el Exceso de potencia Exportada respecto de la comprometida, en cada instante de tiempo
    cEE = result_utils.calcular_coste(eE, vector_Ced_E_expandido)  ##EUR

    # Vector de Costes [EUR] generados por el Déficit de potencia Importada respecto de la comprometida, en cada instante de tiempo
    cDI = result_utils.calcular_coste(dI, vector_Crd_I_expandido)  ##EUR

    # Vector de Costes [EUR] generados por el Exceso de potencia Importada respecto de la comprometida, en cada instante de tiempo
    cEI = result_utils.calcular_coste(eI, vector_Ced_I_expandido)  ##EUR

    # Vector de Costes [EUR] generados por importar una potencia mayor a la contratada (PmaxI), en cada instante de tiempo
    cIextra = result_utils.calcular_coste(
        xIextraAvail, vector_Ced_I_expandido
    )  ## not required  ##EUR ## mismo coste que el Exceso de potencia Importada??

    # Vector de Costes [EUR] Totales por desvíos respecto a compormisos, en cada instante de tiempo
    cTot = [
        cde + cee + cdi + cei for cde, cee, cdi, cei in zip(cDE, cEE, cDI, cEI)
    ]  # EUR
    # coste deficit exportacion + coste exceso exportacion + coste deficit importacion + coste exceso importacion

    R_valores = [
        xE,
        xI,
        dE,
        eE,
        dI,
        eI,
        yE,
        yI,
        xIextraAvail,
        xIextraUsed,
        xBsubir,
        xBbajar,
        cDE,
        cEE,
        cDI,
        cEI,
        cIextra,
        cTot,
        name,
    ]
    R = dict(zip(R_claves, R_valores))

    ## L: Vector que contiene los resultados de cada Link existente
    # input_curves: Curvas de electricidad consumida por cada uno de los 3 links MW

    ######## VECTORES AUXILIARES ####
    H2_Transferido = {
        k: [result_utils.transference_function(x, datos, k - 1) for x in v]
        for k, v in input_curves.items()
    }  # kg/h

    ############################
    L = []
    L_claves = [
        "xI",
        "xO",
        "y",
        "z",
        "p",
        "xPI",
        "xPO",
        "cA",
        "cF",
        "cVI",
        "cVO",
        "cFp",
        "cTot",
    ]
    for i in range(len(datos["L"])):
        xIh = [x * 6 for x in input_curves.get(i + 1, {})]  # MWh
        xOh = [x * 6 for x in H2_Transferido.get(i + 1, {})]  # MWh (kg)
        np = [not x for x in precalentando.get(i + 1, {})]

        ## Nombre
        # name = "Tiner"

        # Vector de booleanos que indican si el Link está operando (solo cuando produce) (=True) en cada instante de tiempo
        y = result_utils.detectar_fin_precalentamiento(
            precalentando.get(i + 1, {})
        )  # si no esta precalentando esta operando? o puede estar parado y calentado?

        # Vector de Potencia [MW] Input al Link en cada instante de tiempo
        xI = result_utils.eliminar_energias_precalentamiento(
            input_curves.get(i + 1, {}), [not x for x in y]
        )  # MW # Despues del precalentamiento!! (resultado ellos)

        # Vector de Potencia [MW] Output del Link en cada instante de tiempo
        xO = result_utils.eliminar_energias_precalentamiento(
            H2_Transferido.get(i + 1, {}), [not x for x in y]
        )  # MW

        # Vector de booleanos que indican si el Link se arrancó (=True) en cada instante de tiempo
        z = result_utils.detectar_y_añadir_primer_true(y)  # Despues de precalentar!!

        # Vector de booleanos que indican si el Link está precalentando (=True) en cada instante de tiempo
        p = precalentando.get(i + 1, {})

        # Vector de Potencia [MW] Input del Link consumida para Precalentar en cada instante de tiempo
        xPI = result_utils.eliminar_energias_precalentamiento(
            input_curves.get(i + 1, {}), np
        )  # MW

        # Vector de Potencia [MW] Output del Link consumida para Precalentar en cada instante de tiempo. Tiene sentido coger output para precalentar??
        xPO = result_utils.eliminar_energias_precalentamiento(
            H2_Transferido.get(i + 1, {}), np
        )  # Kg(h2)

        # Vector de Costes [EUR] de Arranque en cada instante de tiempo
        cA = result_utils.calcular_coste(
            z, datos["L"][i].get("Ca", 0), reemplazar=True
        )  # EUR

        # Vector de Costes [EUR] Fijos por estar en Operación en cada instante de tiempo
        cF = result_utils.calcular_coste(
            y, datos["L"][i].get("Cf", 0) / 6, reemplazar=True
        )  # EUR

        # Vector de Costes [EUR] Variables en función de la potencia Input en cada instante de tiempo
        cVI = result_utils.calcular_coste(
            xIh, datos["L"][i].get("CvI", 0)
        )  # Conversión a EUR =  EUR/MWh (CvI) * MWh (xIh)

        # Vector de Costes [EUR] Variables en función de la potencia Output en cada instante de tiempo
        cVO = result_utils.calcular_coste(
            xOh, datos["L"][i].get("CvO", 0)
        )  # Conversión a EUR =  EUR/MWh (CvO) * MWh (xOh)

        # Vector de Costes [EUR] Fijos por estar en Precalentamiento en cada instante de tiempo
        cFp = result_utils.calcular_coste(
            xPI, datos["L"][i].get("CFp", 0) / 6, reemplazar=True
        )  # EUR

        # Vector de Costes [EUR] Totales en cada instante de tiempo
        cTot = [
            cfp + ca + cf + cvI + cvO
            for cfp, ca, cf, cvI, cvO in zip(cFp, cA, cF, cVI, cVO)
        ]  # EUR
        # coste precalentar + coste arranque + coste fijo + coste variable input + coste variable output

        # Vector de Potencia [MW] Output del Link en cada instante de tiempo, similar a \"xO\" pero sin aplicar Tiempo de Inercia (adelantado respecto a \"xO\"). Es opcional, solo se retorna en links con Tinercia>0
        xO_noTiner = []  ## No se lo que es # not requires

        L_valores = [xI, xO, y, z, p, xPI, xPO, cA, cF, cVI, cVO, cFp, cTot]

        while len(L) <= i:
            L.append({})  # Añade diccionarios vacíos si hace falta

        L[i].update(dict(zip(L_claves, L_valores)))

    Time_claves = ["N", "dtIni", "IncrT"]

    N = 144  # Número de Instantes de Tiempo Simulados
    dtIni = "2024-01-01T00:00:00+00:00"  # Fecha y hora a la que corresponde el primer instante de tiempo, en formato RFC 3339
    IncrT = 10.0  # Incremento de Tiempo [min] entre instantes simulados

    Time_valores = [N, dtIni, IncrT]
    T = dict(zip(Time_claves, Time_valores))

    Costs_claves = ["Total"]
    resultados = [G, C, A, R, L]

    Total = 0.0
    costes_por_resultado = {}  # Para guardar los costes individuales por entrada

    for nombre, resultado in zip(["G", "C", "A", "R", "L"], resultados):
        subtotal, individuales = result_utils.sumar_costes_totales(
            resultado
        )  # Desempaquetar
        Total += subtotal
        costes_por_resultado[nombre] = individuales  # Guardar los individuales por tipo

    Costs_valores = [Total]
    Costs = dict(zip(Costs_claves, Costs_valores))

    print("Coste total:", Total)
    df_costes = pd.DataFrame(costes_por_resultado)
    print(df_costes)

    # Ruta de la carpeta de salida
    output_dir = "./outputs"
    # Nombre del archivo
    output_file = nombre_archivo
    # Ruta completa del archivo
    output_path = os.path.join(output_dir, output_file)

    # Crear la carpeta si no existe
    os.makedirs(output_dir, exist_ok=True)

    data = {}
    data["G"] = [G]
    data["C"] = [C]
    data["A"] = [A]
    data["R"] = [R]
    data["L"] = L
    data["D"] = []
    data["Time"] = T
    data["Costs"] = Costs

    # Convert the Python dictionary to a JSON string with indentation
    # json_data = json.dumps(data, indent=4) # This is not needed for this approach

    # Write the Python dictionary 'data' to resultados_Q.json

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(
            data, f, indent=4, ensure_ascii=False
        )  # Writing dictionary to JSON file

    return


# Ejemplo de uso
vector_entrada_to_Json(
    vector_resultados_Q,
    "./Inputs/IPCEI-Cuantica_kickoff_datosSENER_250228/ejemplo_sener/datos.json",
    nombre_archivo="resultados_Q2.json",
)

# Leer el archivo JSON
with open(
    "inputs/IPCEI-Cuantica_kickoff_datosSENER_250228/ejemplo_sener/datos.json", "r"
) as archivo:
    datos = json.load(archivo)
C_Demanda = datos["C"][0]["Consumo"].get("Pot")  # MW/h
vector_demanda_expandido_H = [x for elem in C_Demanda for x in [elem] * 6]
# Crear el gráfico
plt.plot(
    vector_demanda_expandido_H, label="Demanda", marker="_", markeredgewidth=6
)  # Opcional: marker='o' para poner puntitos

with open(
    "./outputs/resultados_Q2.json",
    "r",
) as archivo:
    R_GQ = json.load(archivo)

C_x_GQ = R_GQ["C"][0]["x"]
# C_x_GQ = calcular_H2(input_curves_GQ, Precalentado_GQ, datos)

with open(
    "./Inputs/IPCEI-Cuantica_kickoff_datosSENER_250228/ejemplo_sener/resultados.json",
    "r",
) as archivo:
    susResultados = json.load(archivo)

C_x_LC = susResultados["C"][0]["x"]
plt.plot(
    C_x_GQ, label="GeneradoQ", marker="_"
)  # Opcional: marker='o' para poner puntitos

with open(
    "./Inputs/IPCEI-Cuantica_kickoff_datosSENER_250228/ejemplo_sener/resultados.json",
    "r",
) as archivo:
    susResultados = json.load(archivo)

C_x_LC = susResultados["C"][0]["x"]
plt.plot(C_x_LC, label="LeidoC", marker="o")  # Opcional: marker='o' para poner puntitos

with open(
    "./outputs/resultados_C.json",
    "r",
) as archivo:
    R_GC = json.load(archivo)

C_x_GC = R_GC["C"][0]["x"]
plt.plot(
    C_x_GC, label="GeneradoC", marker="_"
)  # Opcional: marker='o' para poner puntitos


plt.title("Comparación de Demanda y Generación")
plt.legend()
plt.xlabel("t")
plt.ylabel("H2/kg")
plt.grid(True)
plt.show()


# Uso de la función
