import igraph as ig
import time
import logging
import fileinput
import pickle
import gc
import multiprocessing as mp
from itertools import chain

# Configuración del logging para registrar información y advertencias en archivo y consola
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("grafo_parkingson_paralelizado.log", mode='w', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

def cargar_ubicaciones_directo(ubicaciones_path):
    """
    Carga las ubicaciones directamente desde el archivo de texto.
    Cada línea debe tener el formato: 'longitud,latitud'
    Retorna una lista de tuplas (lat, lon), el número de nodos y un diccionario con listas separadas.
    """
    logging.info("Cargando ubicaciones directamente desde el archivo...")
    latitudes = []
    longitudes = []
    with open(ubicaciones_path, 'r') as f:
        for line in f:
            try:
                lon_str, lat_str = line.strip().split(',')
                latitudes.append(float(lat_str))
                longitudes.append(float(lon_str))
            except ValueError:
                logging.warning(f"Línea malformada en el archivo de ubicaciones: '{line.strip()}'")
    ubicaciones = list(zip(latitudes, longitudes))
    num_nodos = len(ubicaciones)
    logging.info(f"Se cargaron {num_nodos} ubicaciones.")
    return ubicaciones, num_nodos, {'lat': latitudes, 'lon': longitudes}

def procesar_linea_usuarios(linea_idx_contenido, num_nodos):
    """
    Procesa una línea del archivo de usuarios para extraer las conexiones.
    Cada línea representa las conexiones de un usuario (nodo).
    Retorna una lista de aristas locales (src, dst).
    """
    idx, linea = linea_idx_contenido
    aristas_locales = []
    conexiones = set()
    for x in linea.strip().split(','):
        x_strip = x.strip()
        if x_strip.isdigit():
            dst = int(x_strip)
            if 1 <= dst <= num_nodos:
                conexiones.add(dst - 1)  # Ajuste a índice base 0
    aristas_locales.extend((idx - 1, dst) for dst in conexiones)
    return aristas_locales

def procesar_usuarios_paralelizado(usuarios_path, num_nodos, num_procesos=mp.cpu_count()):
    """
    Procesa el archivo de usuarios en paralelo para extraer las aristas.
    Utiliza multiprocessing para acelerar el procesamiento de grandes archivos.
    Retorna una lista de aristas filtradas válidas.
    """
    logging.info(f"Procesando usuarios en paralelo con {num_procesos} procesos...")
    start_time_procesamiento = time.time()
    with fileinput.input(usuarios_path) as f:
        # Enumerar líneas hasta num_nodos para evitar procesar líneas extra
        lineas_enumeradas = ((idx, linea) for idx, linea in enumerate(f, start=1) if idx <= num_nodos)
        with mp.Pool(processes=num_procesos) as pool:
            resultados = pool.starmap(procesar_linea_usuarios, [(item, num_nodos) for item in lineas_enumeradas])
    # Aplanar la lista de resultados
    aristas = list(chain.from_iterable(resultados))
    # Filtrar aristas fuera de rango
    aristas_filtradas = [(src, dst) for src, dst in aristas if 0 <= src < num_nodos and 0 <= dst < num_nodos]
    logging.info(f"Se procesaron {len(aristas_filtradas)} aristas en {time.time() - start_time_procesamiento:.2f} s.")
    return aristas_filtradas

def crear_grafo_igraph_paralelizado(ubicaciones_path, usuarios_path, output_grafo_path):
    """
    Crea el grafo de igraph y guarda el grafo con los atributos de ubicación.
    Lee ubicaciones y conexiones, construye el grafo y lo serializa a disco.
    """
    start_time = time.time()

    # Cargar ubicaciones y atributos
    ubicaciones, num_nodos, atributos_ubicacion = cargar_ubicaciones_directo(ubicaciones_path)
    # Procesar conexiones de usuarios en paralelo
    aristas = procesar_usuarios_paralelizado(usuarios_path, num_nodos)

    logging.info("Creando grafo de igraph...")
    g = ig.Graph(directed=True)
    g.add_vertices(num_nodos)
    g.add_edges(aristas)

    # Añadir atributos de ubicación al grafo
    for key, values in atributos_ubicacion.items():
        g.vs[key] = values

    # Guardar el grafo en un archivo pickle
    logging.info(f"Guardando grafo con atributos en '{output_grafo_path}'...")
    with open(output_grafo_path, 'wb') as f:
        pickle.dump(g, f, protocol=pickle.HIGHEST_PROTOCOL)

    logging.info(f"Grafo guardado. Tiempo total: {time.time() - start_time:.2f} s")
    del ubicaciones, aristas, atributos_ubicacion
    gc.collect()
    return g

def cargar_grafo_con_atributos(grafo_path):
    """
    Carga el grafo y sus atributos desde un archivo pickle.
    Retorna el objeto grafo cargado o None si hay error.
    """
    start_time = time.time()
    try:
        with open(grafo_path, 'rb') as f:
            g = pickle.load(f)
        logging.info(f"Grafo cargado correctamente desde '{grafo_path}' en {time.time() - start_time:.4f} segundos.")
    except FileNotFoundError:
        logging.error(f"Archivo del grafo no encontrado en '{grafo_path}'.")
        return None
    except Exception as e:
        logging.error(f"Error al cargar el grafo desde '{grafo_path}': {e}")
        return None
    return g

if __name__ == "__main__":
    # Definir rutas de archivos de entrada y salida
    ubicaciones_archivo = '10_million_location.txt'
    usuarios_archivo = '10_million_user.txt'
    grafo_archivo = 'grafo_igraph_paralelizado.pkl'

    # Crear y guardar el grafo
    grafo = crear_grafo_igraph_paralelizado(ubicaciones_archivo, usuarios_archivo, grafo_archivo)
    # Cargar el grafo para verificar su integridad
    grafo_cargado = cargar_grafo_con_atributos(grafo_archivo)
    if grafo_cargado:
        logging.info(f"Grafo cargado con {grafo_cargado.vcount()} nodos y {grafo_cargado.ecount()} aristas.")
        if grafo_cargado.vcount() > 0:
            # Mostrar atributos de latitud y longitud del primer nodo si existen
            if 'lat' in grafo_cargado.vs[0].attributes():
                print(f"Latitud del nodo 1: {grafo_cargado.vs[0]['lat']}") 
            else:
                print("El nodo 1 no tiene el atributo 'lat'.")

            if 'lon' in grafo_cargado.vs[0].attributes():
                print(f"Longitud del nodo 1: {grafo_cargado.vs[0]['lon']}") 
            else:
                print("El nodo 1 no tiene el atributo 'lon'.")
    else:
        logging.error("No se pudo cargar el grafo para la prueba.")

    # Liberar memoria
    del grafo_cargado
    gc.collect()