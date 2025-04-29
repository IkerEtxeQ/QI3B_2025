import pandas as pd
import matplotlib.pyplot as plt

mensaje_input_curves = "Curvas de electricidad consumida por cada uno de los 3 links MW {link: [input(t) ] Incluye energía de precalentamiento"
mensaje_precalentando = "Estado de Precalentamiento por Link (precalentando)"
mensaje_G = "G: Energía obtenida (MWh) del generador G (en este caso, energía solar)"
mensaje_Q = "Q: El nivel de carga (MWh) que hay en cada punto del tiempo en la batería"
mensaje_A = "A: Energía obtenida de la batería (MWh)"
mensaje_R = "R: La energía (MWh) que se compra a la red en cada punto del tiempo"


def __print_resultado(mensaje: str, resultado):
    print(f"\n--- {mensaje} ---")

    if resultado is None:
        print("Sin resultados (None)")
        return

    if not isinstance(resultado, (list, dict)):
        print(f"Tipo de resultado no soportado: {type(resultado)}")
        return

    try:
        df_resultado = pd.DataFrame(resultado)
        print(df_resultado.to_string())
    except Exception as e:
        print(f"No se pudo construir el DataFrame: {e}")


def visualizar_costes(Total: float, costes_por_resultado: list):
    print("Coste total:", Total)
    df_costes = pd.DataFrame(costes_por_resultado)
    print(df_costes)


def visualizar_vector_entrada_resultados(vector_entrada_resultados: list):
    """
    Presenta el vector_entrada_resultados de una manera más organizada y legible,
    utilizando pandas DataFrames para las series temporales y formateando la salida.

    Args:
        vector_entrada_resultados (list): La lista conteniendo los datos a visualizar.
    """

    if not vector_entrada_resultados or len(vector_entrada_resultados) != 6:
        print("Error: vector_entrada_resultados inválido o incompleto.")
        return

    mensajes = [
        mensaje_input_curves,
        mensaje_precalentando,
        mensaje_G,
        mensaje_Q,
        mensaje_A,
        mensaje_R,
    ]

    valores = list(zip(mensajes, vector_entrada_resultados))

    dict_mensajes = {f"{i + 1}": valor for i, valor in enumerate(valores)}

    for resultado in dict_mensajes.values():
        __print_resultado(resultado[0], resultado[1])


def graficar_listas(*listas: list, labels=None, markers=None):
    for i, lista in enumerate(listas):
        label_i = labels[i] if labels and i < len(labels) else f"lista {i + 1}"
        marker_i = markers[i] if markers and i < len(markers) else "_"
        plt.plot(lista, label=label_i, marker=marker_i)

    plt.title("Comparación de Demanda y Generación")
    plt.legend()
    plt.xlabel("t")
    plt.ylabel("H2/kg")
    plt.grid(True)
    plt.show()
