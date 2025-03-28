import networkx as nx
import matplotlib.pyplot as plt

import neal

import numpy as np
import time


def definir_posiciones(dic_nodos, radio=1):
    """Define las posiciones de los nodos en un círculo."""
    n = len(dic_nodos)
    posiciones = {}
    for i in range(n):
        angulo = 2 * np.pi * i / n  # Ángulo en radianes
        x = -radio * np.cos(angulo)
        y = radio * np.sin(angulo)
        posiciones[str(i + 1)] = (x, y)
    return posiciones


def crearGrafoNodeWeight(dic_nodos, dic_aristas, titulo):
    # Crear el grafo
    G = nx.Graph()

    # Agregar todos los nodos (para evitar que falten pueblos sin enemigos)
    G.add_nodes_from(dic_nodos.keys())

    # Agregar aristas según las enemistades
    for pueblo, enemigo in dic_aristas.items():
        G.add_edge(pueblo, enemigo)

    # Llamar al método para definir posiciones
    posiciones = definir_posiciones(dic_nodos)

    # Opciones de dibujo
    options = {
        "font_size": 12,
        "node_size": 1000,
        "node_color": "white",
        "edgecolors": "black",
        "linewidths": 2,
        "width": 1,
        "with_labels": True,
        "labels": dic_nodos,
    }

    # Dibujar el grafo
    nx.draw_networkx(G, posiciones, **options)

    # Ajustes finales
    ax = plt.gca()
    ax.margins(0.20)
    plt.axis("off")
    plt.title(titulo)
    plt.show()


def matrizHamiltoniano(bqm, eliminar_bajo_diagonal=True):
    """Muestra la matriz QUBO y los detalles del Hamiltoniano."""

    Q, lineal, interacciones, offset = crearMatrizHamiltoniano(
        bqm, eliminar_bajo_diagonal
    )

    # Configurar la visualización de matrices sin mostrar decimales innecesarios
    np.set_printoptions(precision=2, suppress=True)

    # Mostrar la matriz QUBO
    print("Matriz QUBO:")
    print("-------------------------")
    print(Q)
    print("")

    # Mostrar el Hamiltoniano del sistema
    print("HAMILTONIANO DEL SISTEMA:")
    print("-------------------------")

    # Formatear los términos lineales para eliminar decimales innecesarios
    print(
        "Término lineal:",
        {
            var: f"{val:.2f}" if val % 1 != 0 else f"{int(val)}"
            for var, val in sorted(lineal.items())
        },
    )

    # Formatear los términos cuadráticos
    print(
        "Términos cuadráticos:",
        {
            (var1, var2): f"{val:.2f}" if val % 1 != 0 else f"{int(val)}"
            for (var1, var2), val in interacciones.items()
        },
    )

    # Formatear el offset
    print("Offset:", f"{offset:.2f}" if offset % 1 != 0 else f"{int(offset)}")
    print("")


def crearMatrizHamiltoniano(bqm, eliminar_bajo_diagonal):
    """Crea la matriz QUBO utilizando los términos lineales y cuadráticos del BQM.

    Si `eliminar_bajo_diagonal` es True, se eliminan los valores debajo de la diagonal.
    """
    variables = sorted(bqm.variables)
    var_index = {var: i for i, var in enumerate(variables)}
    lineal = bqm.linear
    sorted_lineal = {var: lineal[var] for var in variables}
    interacciones = bqm.quadratic
    n = len(bqm.variables)

    # Crear la matriz de ceros
    Q = np.zeros((n, n))

    # Crear la lista de valores lineales en función del índice de las variables
    diagonal_values = [
        lineal[var] for var in variables
    ]  # Ordena las variables antes de asignar valores

    # Comprobamos si el número de valores lineales coincide con el tamaño de la matriz Q
    if len(diagonal_values) == Q.shape[0]:
        np.fill_diagonal(Q, diagonal_values)
    else:
        raise ValueError(
            "El número de valores lineales no coincide con el tamaño de la matriz Q."
        )

    # Asignar términos cuadráticos en la parte superior de la matriz
    for (var1, var2), coef in interacciones.items():
        i, j = var_index[var1], var_index[var2]
        Q[i, j] = Q[j, i] = coef  # Solo en la parte superior

    # Si eliminar_bajo_diagonal es True, ponemos los valores debajo de la diagonal en 0
    if eliminar_bajo_diagonal:
        for i in range(n):
            for j in range(i + 1, n):  # Desde la primera fila debajo de la diagonal
                Q[j, i] = 0  # Eliminar valor en la parte inferior

    return Q, sorted_lineal, interacciones, bqm.offset


def ejecucionSimmulatedAnnealing(model, bqm, num_reads=10):
    ######### Resolvemos el problema - Simulador Annealing #########
    sampler = neal.SimulatedAnnealingSampler()  # minimiza la energia del problema que se le proporcina. Si quieres maximizar f(x), minimiza -f(x).
    t_inicial = time.time()
    sampleset = sampler.sample(
        bqm, num_reads=10
    )  # Ejecuta el algoritmo de optimización en el problema 100 veces. Devuelve un conjunto de muestras encontradas de minima energia.
    t_final = time.time()
    decoded_samples = model.decode_sampleset(
        sampleset
    )  # decoded_samples contiene información similar a sampleset, pero en un formato más fácil de entender y usar.
    execution_time_SimulatedAnnealing = t_final - t_inicial
    rounded_time = redondeoDecimalesSignificativos(execution_time_SimulatedAnnealing, 2)

    print("RESULTADOS SIMULATED ANNEALING:")
    print("-------------------------")
    best_sample = min(
        decoded_samples, key=lambda d: d.energy
    )  # min(iterable, key=func) fuc= lambda d: d.energy función anonima que se aplica a cada elemento d. devuelve la energia de cada elemeto.
    print(best_sample)
    print("")

    print("Tiempo de ejecución de Simulated Annealing:", rounded_time, "segundos")


def redondeoDecimalesSignificativos(numero, n_decimales=2):
    """
    Redondea un número a un número específico de decimales significativos,
    asegurando que al menos `n_decimales` dígitos se muestren después del punto decimal.

    Args:
        numero (float): El número a redondear.
        n_decimales (int, opcional): El número mínimo de decimales que se mostrarán.
                                      Debe ser un entero positivo. Por defecto es 2.

    Returns:
        str: El número redondeado formateado como una cadena.
             Retorna "-1" si el número no es decimal o si todos los dígitos decimales son cero.
    """
    numero_str = str(numero)

    if "." not in numero_str:
        return "-1"  # No es un número decimal

    decimales = numero_str[numero_str.index(".") + 1 :]

    primer_no_cero_encontrado = False
    posicion_no_cero = 0

    for i, digito in enumerate(decimales):
        if digito != "0":
            primer_no_cero_encontrado = True
            posicion_no_cero = i
            break  # Salir del bucle al encontrar el primer dígito no cero

    if not primer_no_cero_encontrado:
        return "-1"  # Todos los dígitos decimales son cero

    # Calcular la precisión (número total de decimales a mostrar)
    precision = posicion_no_cero + n_decimales

    # Formatear el número con la precisión calculada
    return f"{numero:.{precision}f}"
