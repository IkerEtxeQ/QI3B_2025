import networkx as nx
import matplotlib.pyplot as plt
import neal
import numpy as np
import time
import re
import itertools
import sympy as sp
from pyqubo import Binary, Placeholder
from IPython.display import display, Math


def definir_posiciones(dic_nodos, radio=1):
    """Define las posiciones de los nodos en un círculo equidistantes.

    Args:
        dic_nodos (dict): Diccionario de nodos.
        radio (float): Radio del círculo.

    Returns:
        dict: Diccionario con las posiciones de los nodos.
    """
    n = len(dic_nodos)
    posiciones = {}
    for i in range(n):
        angulo = 2 * np.pi * i / n  # Ángulo en radianes
        x = -radio * np.cos(angulo)
        y = radio * np.sin(angulo)
        posiciones[str(i + 1)] = (x, y)
    return posiciones


def crear_SimpleGrafo_node_weight(dic_nodos, dic_aristas, titulo=None):
    """Crea y visualiza un grafo con pesos en los nodos.

    Args:
        dic_nodos (dict): Diccionario con los nodos y sus pesos.
        dic_aristas (dict): Diccionario con las aristas del grafo.
        titulo (str): Título del gráfico.
    """
    # Crear el grafo
    G = nx.Graph()

    # Agregar todos los nodos (para evitar que falten pueblos sin enemigos)
    G.add_nodes_from(dic_nodos.keys())

    # Agregar aristas según las enemistades
    for nodo1, nodo2 in dic_aristas.items():
        G.add_edge(nodo1, nodo2)

    # Define las posiciones de los nodos
    posiciones = definir_posiciones(dic_nodos)

    # Opciones de dibujo por defecto
    default_options = {
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
    nx.draw_networkx(G, posiciones, **default_options)

    # Ajustes finales
    ax = plt.gca()
    ax.margins(0.20)
    plt.axis("off")
    if titulo:
        plt.title(titulo)
    plt.show()


def asignar_valores_diagonales(Q, lineal, variables):
    """Asigna los valores lineales a la diagonal de la matriz Q.
    Q: Matriz cuadrada de ceros.
    lineal: Términos lineales del Hamiltoniano.
    variables: Lista de variables del Hamiltoniano.
    """
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


def asignar_terminos_cuadraticos(
    Q, interacciones, var_index, eliminar_bajo_diagonal=True
):
    """Asigna los términos cuadráticos a la matriz Q.
    Q: Matriz cuadrada de ceros.
    interacciones: Términos cuadráticos del Hamiltoniano.
    var_index: Diccionario que mapea las variables a sus índices en la matriz Q.
    eliminar_bajo_diagonal: Si es True, elimina los valores debajo de la diagonal.
    """
    for (var1, var2), coef in interacciones.items():
        i, j = var_index[var1], var_index[var2]
        if eliminar_bajo_diagonal:
            if i < j:
                Q[i, j] = coef
            else:
                Q[j, i] = coef
        else:
            Q[i, j] = Q[j, i] = coef


def mostrar_matriz_hamiltoniano(H, lambda_dict=None, eliminar_bajo_diagonal=True):
    """Muestra la matriz QUBO y los detalles del Hamiltoniano.

    Args:
        H: Expresión del hamiltoniano dependiente de las variables binarias y coeficientes de lagrange (placeholders).
        lambda_dict (dict, optional): Diccionario que mapea los coeficientes de lagrange (Placeholder) a sus valores numéricos.
        eliminar_bajo_diagonal (bool): Indica si se eliminan los valores debajo de la diagonal.
    """
    bqm = compilar_hamiltoniano(H, lambda_dict)[1]  # Compila el Hamiltoniano
    variables = sorted(bqm.variables)
    var_index = {var: i for i, var in enumerate(variables)}
    lineal = bqm.linear  ##  dict-> key: var1, value
    interacciones = bqm.quadratic  ## dict -> key:tupla(var1, var2), values
    n = len(bqm.variables)

    # Crear la matriz de ceros
    Q = np.zeros((n, n))

    # Llamar al método para asignar valores diagonales
    asignar_valores_diagonales(Q, lineal, variables)
    asignar_terminos_cuadraticos(Q, interacciones, var_index, eliminar_bajo_diagonal)

    # Configurar la visualización de matrices
    np.set_printoptions(precision=2, suppress=True)

    # Formatear los términos para eliminar decimales innecesarios
    def formatear_valor(val):
        return (
            f"{val:.2f}" if isinstance(val, float) and val % 1 != 0 else f"{int(val)}"
        )

    imprimir_resultados_hamiltoniano(Q, lineal, interacciones, bqm, formatear_valor)


def imprimir_resultados_hamiltoniano(Q, lineal, interacciones, bqm, formatear_valor):
    """Imprime la matriz QUBO y los detalles del Hamiltoniano.

    Args:
        Q (numpy.ndarray): Matriz QUBO.
        lineal (dict): Términos lineales del Hamiltoniano.
        interacciones (dict): Términos cuadráticos del Hamiltoniano.
        bqm (dimod.BinaryQuadraticModel): Modelo BQM.
        formatear_valor (function): Función para formatear valores.
    """
    print("Matriz QUBO:")
    print("-------------------------")
    print(Q)
    print("")

    print("HAMILTONIANO DEL SISTEMA:")
    print("-------------------------")
    print(
        "Término lineal:",
        {var: formatear_valor(val) for var, val in sorted(lineal.items())},
    )
    print(
        "Términos cuadráticos:",
        {
            (var1, var2): formatear_valor(val)
            for (var1, var2), val in interacciones.items()
        },
    )
    print("Offset:", formatear_valor(bqm.offset))
    print("")


def ejecucion_simulated_annealing(H, lambda_dict=None, num_reads=10, n_decimales=4):
    """Ejecuta Simulated Annealing para resolver el problema QUBO.

    Args:
        H: El Hamiltoniano de PyQUBO (expresión simbólica).
        lambda_dict (dict, optional): Diccionario que mapea los coeficientes de lagrange (Placeholder) a sus valores numéricos.
        num_reads (int): Número de lecturas para el sampler.
        n_decimales (int): Número de decimales significativos para el tiempo de ejecución.
    """
    model, bqm = compilar_hamiltoniano(H, lambda_dict)  # Compila el Hamiltoniano
    sampler = neal.SimulatedAnnealingSampler()
    t_inicial = time.time()
    sampleset = sampler.sample(bqm, num_reads=num_reads)
    t_final = time.time()
    if lambda_dict:
        decoded_samples = model.decode_sampleset(sampleset, feed_dict=lambda_dict)
    else:
        decoded_samples = model.decode_sampleset(sampleset)
    execution_time_SimulatedAnnealing = t_final - t_inicial
    rounded_time = redondeo_decimales_significativos(
        execution_time_SimulatedAnnealing, n_decimales
    )

    imprimir_resultados_simulated_annealing(decoded_samples, rounded_time)


def imprimir_resultados_simulated_annealing(decoded_samples, rounded_time):
    print("RESULTADOS SIMULATED ANNEALING:")
    print("-------------------------")
    best_sample = min(decoded_samples, key=lambda d: d.energy)
    print(best_sample)
    print("")

    print("Tiempo de ejecución de Simulated Annealing:", rounded_time, "segundos")


def redondeo_decimales_significativos(numero, n_decimales=2):
    """Redondea un número a un número específico de decimales significativos.

    Args:
        numero (float): El número a redondear.
        n_decimales (int): El número de decimales a mostrar.

    Returns:
        str: El número redondeado formateado como una cadena.

    Raises:
        ValueError: Si el número no es decimal o si todos los dígitos decimales son cero.
    """
    if not isinstance(numero, (int, float)):
        raise ValueError("El número debe ser un entero o un float.")

    if not isinstance(
        numero, float
    ):  # Si es entero, lo convertimos a float para formatear
        numero = float(numero)

    formato = f".{n_decimales}f"  # Creamos el string de formato
    return f"{numero:{formato}}"  # Aplicamos el formato


def calculate_energy(H, solution, lambda_dict):
    """Calcula la energía del Hamiltoniano de PyQUBO para una solución dada.

    Args:
        H: El Hamiltoniano de PyQUBO (expresión simbólica sin compilar).
        solution (dict): Diccionario donde las claves son los nombres de las variables
                       (e.g., 'x0', 'x1') y los valores son 0 o 1.
        feed_dict (dict, optional): Diccionario que mapea los Placeholder a sus valores numéricos.
                                     Defaults to None.

    Returns:
        float: El valor de la energía.
    """
    bqm = compilar_hamiltoniano(H, lambda_dict)[1]  # Compila el Hamiltoniano

    # Calcula la energía para la solución dada
    energy = bqm.energy(solution)

    return energy


def generate_all_solutions(H):
    """Genera todas las soluciones posibles para las variables binarias en el Hamiltoniano.

    Args:
        H: El Hamiltoniano de PyQUBO (expresión simbólica).

    Returns:
        list: Una lista de diccionarios, donde cada diccionario es una solución.
    """
    # Extrae y ordena los nombres de las variables por su índice
    variable_names = sorted(
        set(re.findall(r"x_\d+", str(H))), key=lambda x: int(x.split("_")[1])
    )

    n_variables = len(variable_names)
    all_combinations = itertools.product([0, 1], repeat=n_variables)

    solutions = []
    for combination in all_combinations:
        solution = {variable_names[i]: combination[i] for i in range(n_variables)}
        solutions.append(solution)

    return solutions


def visualize_energies(hamiltonian, lambda_dict: dict = None) -> None:
    """Visualiza las energías para cada solución con el Hamiltoniano dado usando un scatter plot.

    Args:
        hamiltonian: El Hamiltoniano de PyQUBO (expresión simbólica, sin compilar).
        lambda_dict (dict, optional): Diccionario que mapea los Placeholder a sus valores numéricos.
                                     Defaults to None.
    """

    all_solutions = generate_all_solutions(hamiltonian)
    energies = []
    for solution in all_solutions:
        energy = calculate_energy(hamiltonian, solution, lambda_dict)
        energies.append(energy)

    # Crear etiquetas para las soluciones (solo los valores de x_i)
    solution_labels = []
    for solution in all_solutions:
        label = "".join(
            str(solution[var]) for var in sorted(solution.keys())
        )  # Combina los valores de x_i
        solution_labels.append(label)

    # Crear el gráfico de dispersión (scatter plot)
    plt.figure(figsize=(12, 6))  # Ajusta el tamaño de la figura
    x_values = range(
        len(all_solutions)
    )  # Crear valores para el eje x (índices de las soluciones)
    plt.scatter(x_values, energies)  # Usar plt.scatter en lugar de plt.bar

    # Añadir etiquetas y título
    plt.xlabel("Solución (valores de x_i)")  # Cambiar la etiqueta del eje x
    plt.ylabel("Energía")

    title = "Energías para todas las soluciones"
    if lambda_dict:
        lambda_str = ", ".join(
            [f"{k}={v}" for k, v in lambda_dict.items()]
        )  # Formatear los valores de lambda
        title += f" ({lambda_str})"
    plt.title(title)

    # Personalizar los ticks del eje x para mostrar las soluciones
    plt.xticks(x_values, solution_labels, rotation=45, ha="right")

    # Ajustar márgenes para evitar que las etiquetas se corten
    plt.tight_layout()

    # Mostrar el gráfico
    plt.show()


def compilar_hamiltoniano(H, lambda_dict):
    """Compila el Hamiltoniano de PyQUBO y lo convierte a un modelo BQM.
    H: El Hamiltoniano de PyQUBO (expresión simbólica, sin compilar).
    lambda_dict (dict, optional): Diccionario que mapea los coeficientes de lagrange (Placeholder) a sus valores numéricos.



    Args:
        H: El Hamiltoniano de PyQUBO (expresión simbólica).
        lambda_dict (dict, optional): Diccionario que mapea los Placeholder a sus valores numéricos.

    Returns:
        tuple: El modelo compilado y el modelo BQM.
    """
    # Compila el Hamiltoniano de PyQUBO
    model = H.compile()

    # Si hay un lambda_dict, usarlo al compilar a BQM
    if lambda_dict:
        bqm = model.to_bqm(feed_dict=lambda_dict)
    else:
        bqm = model.to_bqm()

    return model, bqm


def construccion_HR(NR, omega, R):
    """
    Genera la expresión QUBO del Hamiltoniano de Regularización H^R y lo visualiza en LaTeX.
    Define los valores simbólicos de lambda dentro de la función.

    Args:
        NR (int): Número de términos en la suma.
        omegas (dict): Diccionario con los conjuntos Omega_k (clave: k, valor: lista de índices).
        Rs (dict): Diccionario con los valores de R_k (clave: k).

    Returns:
        tuple: (qubo_expression, H_R, term_expressions)
            qubo_expression (pyqubo.Express): Expresión lista para PyQUBO
            H_R (sympy.Expr): Expresión de SymPy del Hamiltoniano.
            term_expressions (list): Lista de expresiones de SymPy para cada término.
    """

    # 0. Definición de los valores simbólicos de lambda
    lambdas = {k: sp.symbols(f"lambda_{k}") for k in range(1, NR + 1)}

    # 1. Construcción de la expresión simbólica (SymPy)
    x = sp.IndexedBase("x")  # Define x como una variable indexada (x_i)
    H_R = 0  # Inicializa el Hamiltoniano
    terminos_HR = []  # Lista para guardar las expresiones de cada término

    for k in range(1, NR + 1):
        sum_omega = sum(x[i] for i in omega[k])  # Sumatoria de x_i para i en Omega_k
        term = lambdas[k] * (sum_omega - R[k]) ** 2
        H_R += term
        terminos_HR.append(term)

    # 2. Visualización en LaTeX
    print("Hamiltoniano de Restricciones:")
    display(Math(sp.latex(H_R)))

    # 3. Preparación para PyQUBO
    qubo_expression = 0
    for k in range(1, NR + 1):
        # Usamos Binary en PyQUBO para las variables x_i
        X = {i: Binary(f"x_{i}") for i in omega[k]}
        sum_omega = sum(X[i] for i in omega[k])
        # Usamos Placeholder para los coeficientes lambda_k
        lambda_k = Placeholder(f"lambda_{k}")
        qubo_expression += lambda_k * (sum_omega - R[k]) ** 2

    return qubo_expression, terminos_HR


def visualizar_parábolas_HR(
    term_expressions, lambdas_valores, x_range=(-5, 5), num_points=100
):
    """
    Grafica los términos del Hamiltoniano, sustituyendo las x_i por una variable continua x,
    manteniendo el centro de la parábola.

    Args:
        term_expressions (list): Lista de expresiones de SymPy para cada término.
        lambdas_valores (dict): Diccionario con los valores numéricos de lambda_k (clave: "lambda_k").
        x_range (tuple): Rango de valores para x (min, max).
        num_points (int): Número de puntos para la graficación.
    """

    # Crea la figura y los ejes
    fig, ax = plt.subplots()

    x = sp.Symbol("x")  # Variable continua x
    xi = sp.IndexedBase("x")  # Indexed Base
    x_index_to_vary = 1  # Index of x_i to vary

    # Itera sobre cada término
    for k, term in enumerate(term_expressions, 1):
        print(f"\n---Visualizando Termino {k} con variable continua x---")
        print("Term (original):", term)

        # 0. Crear los símbolos lambda_k
        lambdas = {
            k: sp.symbols(f"lambda_{k}") for k in range(1, len(term_expressions) + 1)
        }

        # 1.  Sustituir los lambdas por sus valores numéricos
        # Convertir las claves de lambdas_valores a símbolos de SymPy
        lambda_subs = {
            lambdas[i]: lambdas_valores[f"lambda_{i}"]
            for i in range(1, len(lambdas) + 1)
        }

        term_sustituido_lambda = term.subs(lambda_subs)
        print("Term después de sustituir lambdas:", term_sustituido_lambda)

        # 2. Sustituir TODAS las x_i *EXCEPTO x_index_to_vary* por un valor constante (ej: 0)
        sustituciones = {}
        for sym in term_sustituido_lambda.free_symbols:
            if (
                isinstance(sym, sp.Indexed)
                and sym.base == xi
                and sym.indices[0] != x_index_to_vary
            ):
                sustituciones[sym] = 0  # Sustituir por 0

        term_con_xi_fijas = term_sustituido_lambda.subs(sustituciones)

        # 3. Sustituir la xi que no hemos fijado por la variable continua x
        term_con_x = term_con_xi_fijas.subs(xi[x_index_to_vary], x)
        print("Term después de sustituir x[i] por x:", term_con_x)

        # 4. Convertir la expresión de SymPy a una función numérica de NumPy
        try:
            f = sp.lambdify(x, term_con_x, modules=["numpy"])

        except Exception as e:
            print("Error in sp.lambdify:", e)
            continue

        # 5. Generar los valores de x y calcular los valores del término
        x_values = np.linspace(x_range[0], x_range[1], num_points)
        try:
            term_values = f(x_values)
        except Exception as e:
            print("Error evaluating the function:", e)
            continue

        # 6. Graficar
        ax.plot(x_values, term_values, label=f"Term {k}")

    # Personaliza la gráfica
    ax.set_xlabel(f"x")
    ax.set_ylabel("Energía")
    ax.set_title("Términos de $H^R$")
    ax.legend()
    ax.grid(True)
    plt.show()
