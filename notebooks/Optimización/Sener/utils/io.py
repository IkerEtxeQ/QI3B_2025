import json
import os


def leer_archivo_json(ruta: str) -> dict:
    """
    Lee un archivo JSON desde la ruta especificada y devuelve su contenido como un diccionario.

    Args:
        ruta (str): La ruta al archivo JSON.

    Returns:
        dict: El contenido del archivo JSON como un diccionario.
              Devuelve un diccionario vacío si el archivo no existe o está vacío.
    """
    try:
        with open(ruta, "r") as archivo:
            archivo = json.load(archivo)

        return archivo
    except FileNotFoundError:
        print(f"Error: El archivo JSON no se encontró en la ruta: {ruta}")
        return {}
    except json.JSONDecodeError:
        print(f"Error: El archivo JSON en la ruta {ruta} no es un JSON válido.")
        return {}


def guardar_archivo_json(ruta: str, nombre: str, data: dict) -> None:
    """
    Guarda un diccionario como un archivo JSON en la ruta especificada.

    Args:
        ruta (str): La ruta del directorio donde se guardará el archivo JSON.
        nombre (str): El nombre del archivo JSON (con la extensión .json).
        data (dict): El diccionario que se guardará como JSON.

    Returns:
        None: Esta función no devuelve nada.  Crea un archivo en el disco.
    """
    output_path = os.path.join(ruta, nombre)
    os.makedirs(ruta, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(
            data, f, indent=4, ensure_ascii=False
        )  # Writing dictionary to JSON file
