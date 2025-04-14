import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from pyqubo import Binary, Placeholder
from IPython.display import display, Math


def construccion_hamiltoniano(NR, omegas, Rs):
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
    term_expressions = []  # Lista para guardar las expresiones de cada término

    for k in range(1, NR + 1):
        sum_omega = sum(x[i] for i in omegas[k])  # Sumatoria de x_i para i en Omega_k
        term = lambdas[k] * (sum_omega - Rs[k]) ** 2
        H_R += term
        term_expressions.append(term)

    # 2. Visualización en LaTeX
    print("Hamiltoniano de Restricciones:")
    display(Math(sp.latex(H_R)))

    # 3. Preparación para PyQUBO
    qubo_expression = 0
    for k in range(1, NR + 1):
        # Usamos Binary en PyQUBO para las variables x_i
        X = {i: Binary(f"x_{i}") for i in omegas[k]}
        sum_omega = sum(X[i] for i in omegas[k])
        # Usamos Placeholder para los coeficientes lambda_k
        lambda_k = Placeholder(f"lambda_{k}")
        qubo_expression += lambda_k * (sum_omega - Rs[k]) ** 2

    return qubo_expression, H_R, term_expressions


def visualizar_terminos_continuo(
    term_expressions, lambdas_valores, x_range=(-5, 5), num_points=100
):
    """
    Grafica los términos del Hamiltoniano, sustituyendo las x_i por una variable continua x,
    manteniendo el centro de la parábola.

    Args:
        term_expressions (list): Lista de expresiones de SymPy para cada término.
        lambdas_valores (dict): Diccionario con los valores numéricos de lambda_k (clave: k).
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
        term_sustituido_lambda = term.subs(
            {lambdas[k]: lambdas_valores[k] for k in lambdas}
        )
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


# Ejemplo de uso
NR = 2

omegas = {
    1: [1, 2, 3, 4, 5],
    2: [1, 3],
}  # Índices deben coincidir con la definición de x
Rs = {1: 4.0, 2: 0.5}

# 1. Generar la expresión QUBO y visualizar H^R
qubo_expression, H_R, term_expressions = construccion_hamiltoniano(NR, omegas, Rs)

# 2. Definir *valores* para los lambdas
lambdas_valores = {1: 1, 2: 1}

# 3. Visualizar los términos con x continua
visualizar_terminos_continuo(
    term_expressions, lambdas_valores, x_range=(-20, 20), num_points=100
)
