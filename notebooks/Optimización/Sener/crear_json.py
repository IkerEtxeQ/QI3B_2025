
import json

def cargar_datos_json(ruta) -> dict:
    """
    Carga y devuelve los datos de un archivo JSON.

    Parameters
    ----------
    ruta : str
        Ruta al archivo JSON.

    Returns
    -------
    dict
        Contenido del archivo JSON como diccionario.
    """
    with open(ruta, 'r') as archivo:
        datos = json.load(archivo)
    return datos

def _cargar_lista_claves_resultados(datos: dict)-> dict:
    
    claves = {}
    
    for clave in datos["Properties"].keys():
        
    try :
        datos.get("Properties")
        
    

def crear_dict_resultados(resultados:list , datos: dict) -> dict:
    
    
