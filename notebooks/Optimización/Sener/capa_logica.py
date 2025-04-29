import capa_carga_datos as inputs
import utils.crear_json as crear_json
import utils.io as io
import capa_visualizacion as visualiza


def crear_json_desde_resultados(resultados):
    datos = inputs.datos
    resultados = resultados

    input_curves = resultados[0]
    precalentando = resultados[1]
    G_result = resultados[2]
    Q_result = resultados[3]
    A_result = resultados[4]
    R_result = resultados[5]

    G = crear_json.G_json(G_result, datos)

    C = crear_json.C_json(input_curves, precalentando, datos)

    A = crear_json.A_json(A_result, Q_result, datos)

    R = crear_json.R_json(R_result, datos)

    L = crear_json.L_json(input_curves, precalentando, datos)

    resultados = [G, C, A, R, L]

    Costs = crear_json.Costs_json(resultados)

    Time = crear_json.Time_json()

    resultados.append(Time)
    resultados.append(Costs)

    dict_json = crear_json.crear_dict_json(resultados)

    return dict_json


C_Demanda = inputs.datos["C"][0]["Consumo"].get("Pot")  # MW/h
# C_Demanda = [0] + C_Demanda[:-1]
vector_demanda_expandido_H = [x for elem in C_Demanda for x in [elem] * 6]


# vector_oferta_C = inputs.resultados_Sener_json["C"][0]["x"]

# visualiza.visualizar_vector_entrada_resultados(inputs.resultados_Iñigo)
dict_GQ = crear_json_desde_resultados(inputs.resultados_Iñigo)
io.guardar_archivo_json("./outputs", "resultados_Q.json", dict_GQ)

vector_oferta_GQ = dict_GQ["C"][0]["x"]

# visualiza.visualizar_vector_entrada_resultados(inputs.resultados_Sener_desplazados)
dict_GC = crear_json_desde_resultados(inputs.resultados_Sener_desplazados)
io.guardar_archivo_json("./outputs", "resultados_C.json", dict_GQ)

vector_oferta_GC = dict_GC["C"][0]["x"]

# labels = ["Demanda", "GeneradoC", "GeneradoQ", "LeidoC"]
# markers = ["o", "_", "_", "_"]
labels = ["Demanda", "GeneradoC", "GeneradoQ"]
markers = ["o", "_", "_"]
visualiza.graficar_listas(
    vector_demanda_expandido_H,
    vector_oferta_GC,
    vector_oferta_GQ,
    labels=labels,
    markers=markers,
)
