import utils.result_utils as result_utils
import utils.list_utils as list_utils
import capa_visualizacion as visualizar
import numpy as np


def crear_dict_json(resultados: list) -> dict:
    json = {}
    json["G"] = [resultados[0]]
    json["C"] = [resultados[1]]
    json["A"] = [resultados[2]]
    json["R"] = [resultados[3]]
    json["L"] = resultados[4]
    json["D"] = []
    json["Time"] = resultados[5]
    json["Costs"] = resultados[6]

    return json


def G_json(G_result: list, datos) -> dict:
    datos_G = datos["G"][0]

    ## G: Vector que contiene los resultados de cada Generador existente.

    # G_result = energía obtenida (MWh) del generador G (en este caso, energía solar). Para pasar a potencia (MW), multiplicar por 6 (en realidad: dividir entre 1h/6pasos)
    G_claves = ["x", "y", "z", "cA", "cF", "cV", "cTot", "name"]

    # Nombre
    G_name = "PV"

    # Vector de Potencia [MW] Demandada en cada instante de tiempo
    g_x = [
        x * 6 for x in G_result
    ]  # Supongo que es la energia obtenida del generador #Conversión a MW = MWh * 6pasos/1h

    # Vector de booleanos que indican si el Generador está produciendo (=True) en cada instante de tiempo
    g_y = result_utils.crear_lista_booleana_todos_valores_mayor_zero_true(g_x)

    # Vector de booleanos que indican si el Generador se arrancó (=True) en cada instante de tiempo
    g_z = result_utils.crear_lista_booleana_primer_valor_mayor_zero_true(g_y)

    # Vector de Costes [EUR] de Arranque en cada instante de tiempo
    cA = result_utils.calcular_coste_arranque(g_x, datos_G.get("Ca", 0))

    # Vector de Costes [EUR] Fijos por estar en Producción en cada instante de tiempo
    cF = result_utils.calcular_coste_fijo(
        g_x, datos_G.get("Cf", 0) / 6
    )  # Conversión a EUR = EUR/h (cf) * 1h/6pasos

    # Vector de Costes [EUR] Variables en función de la producción en cada instante de tiempo
    cV = result_utils.calcular_coste_variable(
        G_result, datos_G.get("Cv", 0)
    )  # Conversión a EUR =  EUR/MWh (cv) * MWh (G_result)

    # Vector de Costes [EUR] Totales en cada instante de tiempo
    cTot = [
        ca + cf + cv for ca, cf, cv in zip(cA, cF, cV)
    ]  # Arranque + coste fijo + coste variable

    G_valores = [g_x, g_y, g_z, cA, cF, cV, cTot, G_name]
    G = dict(zip(G_claves, G_valores))

    return G


def C_json(input_curves: dict, precalentando: dict, datos) -> dict:
    datos_C = datos["C"][0]

    ## C: Vector que contiene los resultados de cada Consumidor existente -> Si es consumidor consume H2????

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
    # vector_demanda= [0] + vector_demanda[:-1]
    vector_demanda_expandido_H = [x for elem in vector_demanda for x in [elem] * 6]
    vector_demanda_expandido = [x / 6 for x in vector_demanda_expandido_H]  # MW (kg)

    vector_Cc_d = datos_C["Costes"].get("Cc_d")  # EUR/MWh
    vector_Cc_d_expandido = list_utils.expandir_vector(
        vector_Cc_d, 6
    )  # EUR/MW (EUR/Kg)

    vector_Cc_e = datos_C["Costes"].get("Cc_e")  # EUR/MWh
    vector_Cc_e_expandido = list_utils.expandir_vector(
        vector_Cc_e, 6
    )  # EUR/MW (EUR/Kg)

    ###############
    C_claves = ["x", "d", "e", "y", "cD", "cE", "cTot", "Ereal", "REreal", "name"]

    # Nombre
    name = "H2"

    # Vector de Potencia [MW = kg/h] Recibida en cada instante de tiempo
    c_x = suma_links_h  # Kg

    # Vector de Potencia [MW] Recibida de menos (Déficit) respecto de la Solicitada en cada instante de tiempo
    c_d = result_utils.calcular_deficit(suma_links_h, vector_demanda_expandido_H)  # Kg

    # Vector de Potencia [MW] Recibida de más (Exceso) respecto de la Solicitada en cada instante de tiempo
    c_e = result_utils.calcular_exceso(suma_links_h, vector_demanda_expandido_H)  # Kg

    # Vector de booleanos que indican si el Consumidor está consumiendo (=True) en cada instante de tiempo
    c_y = result_utils.crear_lista_booleana_todos_valores_mayor_zero_true(
        suma_links_h
    )  ## Supongo que si producimos H2, se esta consumiendo.

    # Vector de Costes [EUR] generados por el Déficit de potencia suministrada respecto a solicitada en cada instante de tiempo
    c_cD = result_utils.calcular_coste_variable(c_d, vector_Cc_d_expandido)  # EUR

    # Vector de Costes [EUR] generados por el Exceso de potencia suministrada respecto a solicitada en cada instante de tiempo
    c_cE = result_utils.calcular_coste_variable(c_e, vector_Cc_e_expandido)  # EUR

    # Vector de Costes [EUR] Totales por desvíos en cada instante de tiempo
    c_cTot = [c_cd + c_ce for c_cd, c_ce in zip(c_cD, c_cE)]  # EUR

    # Energía Total [MWh] suministrada al Consumidor durante todos los instantes de tiempo
    c_Ereal = np.sum(suma_links)  # Kg

    # Ratio [-] entre la Energía Total suministrada / demandada (*100??)
    c_REreal = c_Ereal / np.sum(vector_demanda_expandido)

    C_valores = [c_x, c_d, c_e, c_y, c_cD, c_cE, c_cTot, c_Ereal, c_REreal, name]
    C = dict(zip(C_claves, C_valores))

    return C


def A_json(A_result: list, Q_result: list, datos) -> dict:
    datos_A = datos["A"][0]

    ### Vector que contiene los resultados de cada Almacenamiento existente
    # Q (MWh)
    # A (MWh)

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
    zC = list_utils.identificar_primer_true_de_cadenas_trues(
        yC
    )  # detecta cada vez empieza una carga, no solo la primera

    # Vector de booleanos que indican si el Almacenamiento está Descargando (=True) en cada instante de tiempo
    yD = [x > 0 for x in A_result]

    # Vector de booleanos que indican si el Generador??  inició Carga (=True) en cada instante de tiempo
    zD = list_utils.identificar_primer_true_de_cadenas_trues(
        yD
    )  # detecta cada vez empieza una descarga, no solo la primera

    # Vector de Costes [EUR] por arrancar/iniciar Carga en cada instante de tiempo
    cAC = result_utils.calcular_coste_fijo(zC, datos_A.get("CaC", 0))

    # Vector de Costes [EUR] Fijos por estar en Carga en cada instante de tiempo
    cFC = result_utils.calcular_coste_fijo(
        xC, datos_A.get("CfC", 0) / 6
    )  # Conversión a EUR = EUR/h (CfC) * 1h/6pasos

    # Vector de Costes [EUR] Variables en función de la potencia Cargada en cada instante de tiempo
    cVC = result_utils.calcular_coste_variable(
        xCh, datos_A.get("CvC", 0)
    )  # Conversión a EUR =  EUR/MWh (CvC) * MWh (xCh)

    # Vector de Costes [EUR] por arrancar/iniciar Descarga en cada instante de tiempo
    cAD = result_utils.calcular_coste_fijo(zD, datos_A.get("CaD", 0))

    # Vector de Costes [EUR] Fijos por estar en Descarga en cada instante de tiempo
    cFD = result_utils.calcular_coste_fijo(
        xD, datos_A.get("CfD", 0) / 6
    )  # Conversión a EUR = EUR/h (CfC) * 1h/6pasos

    # Vector de Costes [EUR] Variables en función de la potencia Descarga en cada instante de tiempo
    cVD = result_utils.calcular_coste_variable(
        xDh, datos_A.get("CvD", 0)
    )  # Conversión a EUR =  EUR/MWh (CvC) * MWh (xDh)

    # Vector de Costes/valor [EUR] por mantener un determinado Nivel almacenado en cada instante de tiempo
    cNm = result_utils.calcular_coste_variable(
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

    return A


def R_json(R_result: list, datos) -> dict:
    datos_R = datos["R"][0]

    ## R (MWh): Vector que contiene los resultados de cada conexión a Red existente

    #### VECTORES AUXILIARES ####
    vector_potencia_comprometida = datos_R.get("Compromisos", {}).get(
        "Pot", np.zeros(144).tolist()
    )  # MW
    vector_potencia_comprometida_exportacion = [
        x if x > 0 else 0 for x in vector_potencia_comprometida
    ]  # MW
    vector_potencia_comprometida_importacion = [
        x if x < 0 else 0 for x in vector_potencia_comprometida
    ]  # MW

    vector_Crd_E = datos_R["Costes"].get("CRd_E", 0)  # EUR/MWh
    vector_Crd_E_expandido = list_utils.expandir_vector(vector_Crd_E, 6)  # EUR/MW

    vector_Ced_E = datos_R["Costes"].get("CRe_E", 0)  # EUR/MWh
    vector_Ced_E_expandido = list_utils.expandir_vector(vector_Ced_E, 6)  # EUR/MW

    vector_Crd_I = datos_R["Costes"].get("CRd_I", 0)  # EUR/MWh
    vector_Crd_I_expandido = list_utils.expandir_vector(vector_Crd_I, 6)  # EUR/MW

    vector_Ced_I = datos_R["Costes"].get("CRe_I", 0)  # EUR/MWh
    vector_Ced_I_expandido = list_utils.expandir_vector(vector_Ced_I, 6)  # EUR/MW

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
    dE = result_utils.calcular_deficit(
        xE, vector_potencia_comprometida_exportacion
    )  # MW

    # Vector de Potencia [MW] Exportada de más (Exceso) respecto de la Comprometida, en cada instante de tiempo
    eE = result_utils.calcular_exceso(
        xE, vector_potencia_comprometida_exportacion
    )  # MW

    # Vector de Potencia [MW] Importada de menos (Déficit) respecto de la Comprometida, en cada instante de tiempo
    dI = result_utils.calcular_deficit(
        xI, vector_potencia_comprometida_importacion
    )  # MW

    # Vector de Potencia [MW] Importada de más (Exceso) respecto de la Comprometida, en cada instante de tiempo
    eI = result_utils.calcular_exceso(
        xI, vector_potencia_comprometida_importacion
    )  # MW

    # Vector de booleanos que indican si se está Exportando hacia la Red (=True) en cada instante de tiempo
    yE = result_utils.crear_lista_booleana_todos_valores_mayor_zero_true(xE)

    # Vector de booleanos que indican si se está Importando desde la Red (=True) en cada instante de tiempo
    yI = result_utils.crear_lista_booleana_todos_valores_mayor_zero_true(xI)

    ##Vector de Potencia [MW] Importación extra, por encima de la PmaxI/contratada, a la que se ha recurrido en cada instante de tiempo
    xIextraAvail = result_utils.calcular_exceso(xI, datos_R.get("PmaxI", 0))  # MW

    # Vector de Potencia [MW] Importación extra, por encima de la PmaxI/contratada, que se ha consumido en cada instante de tiempo
    xIextraUsed = result_utils.calcular_exceso(
        xI, datos_R.get("PmaxI", 0)
    )  ## igual que el anterior??

    # Vector de Potencia [MW] de Banda a Subir ofertada y adjudicada en cada instante de tiempo
    xBsubir = []

    # Vector de Potencia [MW] de Banda a Bajar ofertada y adjudicada en cada instante de tiempo
    xBbajar = []

    # Vector de Costes [EUR] generados por el Déficit de potencia Exportada respecto de la comprometida, en cada instante de tiempo
    cDE = result_utils.calcular_coste_variable(dE, vector_Crd_E_expandido)  ##EUR

    # Vector de Costes [EUR] generados por el Exceso de potencia Exportada respecto de la comprometida, en cada instante de tiempo
    cEE = result_utils.calcular_coste_variable(eE, vector_Ced_E_expandido)  ##EUR

    # Vector de Costes [EUR] generados por el Déficit de potencia Importada respecto de la comprometida, en cada instante de tiempo
    cDI = result_utils.calcular_coste_variable(dI, vector_Crd_I_expandido)  ##EUR

    # Vector de Costes [EUR] generados por el Exceso de potencia Importada respecto de la comprometida, en cada instante de tiempo
    cEI = result_utils.calcular_coste_variable(eI, vector_Ced_I_expandido)  ##EUR

    # Vector de Costes [EUR] generados por importar una potencia mayor a la contratada (PmaxI), en cada instante de tiempo
    cIextra = result_utils.calcular_coste_variable(
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

    return R


def L_json(input_curves: dict, precalentando: dict, datos) -> list:
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
        z = list_utils.identificar_primer_true_de_cadenas_trues(
            y
        )  # Despues de precalentar!!

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
        cA = result_utils.calcular_coste_arranque(z, datos["L"][i].get("Ca", 0))  # EUR

        # Vector de Costes [EUR] Fijos por estar en Operación en cada instante de tiempo
        cF = result_utils.calcular_coste_fijo(y, datos["L"][i].get("Cf", 0) / 6)  # EUR

        # Vector de Costes [EUR] Variables en función de la potencia Input en cada instante de tiempo
        cVI = result_utils.calcular_coste_variable(
            xIh, datos["L"][i].get("CvI", 0)
        )  # Conversión a EUR =  EUR/MWh (CvI) * MWh (xIh)

        # Vector de Costes [EUR] Variables en función de la potencia Output en cada instante de tiempo
        cVO = result_utils.calcular_coste_variable(
            xOh, datos["L"][i].get("CvO", 0)
        )  # Conversión a EUR =  EUR/MWh (CvO) * MWh (xOh)

        # Vector de Costes [EUR] Fijos por estar en Precalentamiento en cada instante de tiempo
        cFp = result_utils.calcular_coste_fijo(
            xPI, datos["L"][i].get("CFp", 0) / 6
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

    return L


def Time_json() -> dict:
    Time_claves = ["N", "dtIni", "IncrT"]

    N = 144  # Número de Instantes de Tiempo Simulados
    dtIni = "2024-01-01T00:00:00+00:00"  # Fecha y hora a la que corresponde el primer instante de tiempo, en formato RFC 3339
    IncrT = 10.0  # Incremento de Tiempo [min] entre instantes simulados

    Time_valores = [N, dtIni, IncrT]
    T = dict(zip(Time_claves, Time_valores))

    return T


def Costs_json(resultados: list) -> dict:
    Costs_claves = ["Total"]

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

    visualizar.visualizar_costes(Total, costes_por_resultado)

    return Costs
