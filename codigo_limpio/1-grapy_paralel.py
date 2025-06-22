import igraph as ig
import time
import logging
import fileinput
import pickle
import gc
import multiprocessing as mp
from itertools import chain

NUM_NODOS = 10_000_000

# Configuración del logging para registrar información y advertencias en archivo y consola
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("grafo_paralelizado.log", mode='w', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

def cargar_ubicaciones_directo(ubicaciones_path, max_nodos):
    logging.info(f"Cargando un máximo de {max_nodos} ubicaciones desde el archivo...")
    latitudes = []
    longitudes = []
    with open(ubicaciones_path, 'r') as f:
        for i, line in enumerate(f):
            if i >= max_nodos:
                logging.info(f"Límite de {max_nodos} nodos alcanzado. Se detiene la lectura de ubicaciones.")
                break
            try:
                # El formato en el archivo original parece ser lat,lon
                lat_str, lon_str = line.strip().split(',')
                latitudes.append(float(lat_str))
                longitudes.append(float(lon_str))
            except ValueError:
                logging.warning(f"Línea malformada en el archivo de ubicaciones: '{line.strip()}'")
    
    ubicaciones = list(zip(latitudes, longitudes))
    num_nodos_reales = len(ubicaciones)
    
    # Advertir si el archivo tiene menos nodos que el esperado
    if num_nodos_reales < max_nodos:
        logging.warning(f"El archivo de ubicaciones contiene {num_nodos_reales} nodos, menos que el máximo esperado de {max_nodos}.")
    
    logging.info(f"Se cargaron {num_nodos_reales} ubicaciones.")
    return ubicaciones, num_nodos_reales, {'lat': latitudes, 'lon': longitudes}

def procesar_linea_usuarios(linea_idx_contenido, num_nodos_max):
    idx, linea = linea_idx_contenido
    aristas_locales = []
    conexiones = set()
    for x in linea.strip().split(','):
        x_strip = x.strip()
        if x_strip.isdigit():
            dst = int(x_strip)
            # Validar contra el número máximo de nodos permitidos
            if 1 <= dst <= num_nodos_max:
                conexiones.add(dst - 1)  # Ajuste a índice base 0
    
    # El nodo origen (idx-1) también debe ser válido
    if 0 <= (idx - 1) < num_nodos_max:
        aristas_locales.extend((idx - 1, dst) for dst in conexiones)
    
    return aristas_locales

def procesar_usuarios_paralelizado(usuarios_path, num_nodos, num_procesos=mp.cpu_count()):
    logging.info(f"Procesando usuarios para {num_nodos} nodos en paralelo con {num_procesos} procesos...")
    start_time_procesamiento = time.time()
    
    with fileinput.input(usuarios_path) as f:
        # Creamos un generador que solo lee hasta la línea `num_nodos`
        lineas_a_procesar = ((idx, linea) for idx, linea in enumerate(f, start=1) if idx <= num_nodos)
        
        with mp.Pool(processes=num_procesos) as pool:
            # Pasamos num_nodos a cada proceso para la validación
            resultados = pool.starmap(procesar_linea_usuarios, [(item, num_nodos) for item in lineas_a_procesar])
            
    # Aplanar la lista de resultados
    aristas = list(chain.from_iterable(resultados))
    
    # El filtrado ya se hace dentro de `procesar_linea_usuarios`, pero una verificación final no está de más.
    aristas_filtradas = [(src, dst) for src, dst in aristas if 0 <= src < num_nodos and 0 <= dst < num_nodos]
    
    logging.info(f"Se procesaron {len(aristas_filtradas)} aristas en {time.time() - start_time_procesamiento:.2f} s.")
    return aristas_filtradas

def crear_grafo_igraph_paralelizado(ubicaciones_path, usuarios_path, output_grafo_path, max_nodos):
    start_time = time.time()

    # 1. Cargar ubicaciones, usando max_nodos como límite.
    ubicaciones, num_nodos_reales, atributos_ubicacion = cargar_ubicaciones_directo(ubicaciones_path, max_nodos)
    
    # 2. Procesar conexiones, usando el número real de nodos como límite para la consistencia.
    aristas = procesar_usuarios_paralelizado(usuarios_path, num_nodos_reales)

    logging.info("Creando grafo de igraph...")
    g = ig.Graph(directed=True)
    # Añadimos la cantidad real de vértices encontrados
    g.add_vertices(num_nodos_reales)
    g.add_edges(aristas)

    # Añadir atributos de ubicación al grafo
    for key, values in atributos_ubicacion.items():
        g.vs[key] = values

    # Guardar el grafo en un archivo pickle
    logging.info(f"Guardando grafo con atributos en '{output_grafo_path}'...")
    with open(output_grafo_path, 'wb') as f:
        pickle.dump(g, f, protocol=pickle.HIGHEST_PROTOCOL)

    logging.info(f"Grafo guardado. Tiempo total: {time.time() - start_time:.2f} s")
    del ubicaciones, aristas, atributos_ubicacion, g
    gc.collect()

if __name__ == "__main__":
    ubicaciones_archivo = '10_million_location.txt'
    usuarios_archivo = '10_million_user.txt'
    grafo_archivo = 'grafo_igraph_paralelizado.pkl'

    # Crear y guardar el grafo, pasando la constante global como el límite máximo.
    crear_grafo_igraph_paralelizado(ubicaciones_archivo, usuarios_archivo, grafo_archivo, max_nodos=NUM_NODOS)
    
    gc.collect()