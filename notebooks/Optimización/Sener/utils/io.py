import json
import os


def leer_archivo_json(ruta: str) -> dict:
    with open(ruta, "r") as archivo:
        archivo = json.load(archivo)

    return archivo


def guardar_archivo_json(ruta: str, nombre: str, data: dict) -> None:
    output_path = os.path.join(ruta, nombre)
    os.makedirs(ruta, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(
            data, f, indent=4, ensure_ascii=False
        )  # Writing dictionary to JSON file
