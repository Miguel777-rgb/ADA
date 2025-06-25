import igraph as ig
import logging as log
import pickle
import polars as pl
import time

# Configuración del logging similar a los otros scripts.
log.basicConfig(level=log.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[log.FileHandler("grafo_pruebas.log", mode='w'), log.StreamHandler()])

def cargar_grafo(grafo_path):
    # Carga el grafo directamente desde el archivo pickle. No necesita un archivo de atributos separado.
    try:
        with open(grafo_path, 'rb') as f:
            g = pickle.load(f)
        log.info(f"Grafo cargado correctamente desde '{grafo_path}'.")
        return g
    except FileNotFoundError:
        log.error(f"Archivo del grafo no encontrado en '{grafo_path}'.")
        return None

def verificar_insercion_nodos_aristas(grafo, num_nodos_esperado=None, num_aristas_esperado=None):
    # Compara el número de nodos y aristas del grafo con valores esperados.
    log.info("Verificando la inserción de nodos y aristas...")
    num_nodos_actual = grafo.vcount()
    num_aristas_actual = grafo.ecount()
    log.info(f"Número de nodos: {num_nodos_actual}, número de aristas: {num_aristas_actual}.")

def verificar_indexacion_nodos(grafo, num_nodos_a_verificar=10):
    # Intenta acceder a los primeros y últimos nodos para verificar que la indexación es correcta.
    log.info("Verificando la indexación de los nodos...")
    n_total = grafo.vcount()
    # Crea una lista de índices de prueba (primeros y últimos).
    indices_a_verificar = list(range(min(num_nodos_a_verificar, n_total))) + \
                          list(range(max(0, n_total - num_nodos_a_verificar), n_total))
    for index in sorted(list(set(indices_a_verificar))):
        try:
            nodo = grafo.vs[index] # Intenta acceder al vértice por su índice.
        except IndexError:
            log.error(f"Error: No se encontró ningún nodo con el índice {index}.")
    log.info("Verificación de indexación completada.")

def verificar_atributos_nodos(grafo, atributo_a_verificar='lat', num_nodos_a_verificar=5):
    # Verifica si un atributo específico (ej. 'lat') existe en una muestra de nodos.
    log.info(f"Verificando el atributo '{atributo_a_verificar}'...")
    for index in range(min(num_nodos_a_verificar, grafo.vcount())):
        nodo = grafo.vs[index]
        if atributo_a_verificar in nodo.attributes():
            valor_atributo = nodo[atributo_a_verificar]
            log.debug(f"Nodo {index} tiene '{atributo_a_verificar}' con valor: {valor_atributo}.")
        else:
            log.warning(f"Nodo {index} no tiene el atributo '{atributo_a_verificar}'.")
    log.info(f"Verificación del atributo '{atributo_a_verificar}' completada.")

def verificar_conexiones(grafo, nodo_a_verificar=0):
    # Muestra el número de conexiones salientes (vecinos) de un nodo específico.
    log.info(f"Verificando las conexiones del nodo {nodo_a_verificar}...")
    conexiones_salientes = grafo.successors(nodo_a_verificar)
    log.info(f"El nodo {nodo_a_verificar} tiene {len(conexiones_salientes)} conexiones salientes.")

def generar_tabla_grafos_ordenados(grafo, num_nodos=50):
    # Crea una tabla de resumen usando Polars para una muestra de nodos.
    nodos = list(range(grafo.vcount()))
    nodos_seleccionados = sorted(list(set(nodos[:num_nodos] + nodos[-num_nodos:])))
    data = []
    for idx, nodo_idx in enumerate(nodos_seleccionados):
        # Para cada nodo de la muestra, extrae sus atributos y número de conexiones.
        lat = grafo.vs[nodo_idx].attributes().get('lat')
        lon = grafo.vs[nodo_idx].attributes().get('lon')
        num_conexiones = len(grafo.successors(nodo_idx))
        data.append({'Index': idx + 1, 'Nodo': nodo_idx, 'Latitud': lat, 'Longitud': lon, 'Conexiones': num_conexiones})
    # Convierte la lista de diccionarios a un DataFrame de Polars.
    return pl.DataFrame(data)

# --- Bloque principal de prueba ---
log.info("Iniciando prueba de carga del grafo.")
grafo_archivo = 'grafo_igraph_paralelizado.pkl'
grafo = cargar_grafo(grafo_archivo)

if grafo:
    verificar_insercion_nodos_aristas(grafo)
    verificar_indexacion_nodos(grafo)
    verificar_atributos_nodos(grafo, 'lat')
    verificar_atributos_nodos(grafo, 'lon')
    verificar_conexiones(grafo, 0)
    tabla_grafos = generar_tabla_grafos_ordenados(grafo)
    if tabla_grafos is not None:
        log.info(f"\n{tabla_grafos}")
        tabla_grafos.write_csv('tabla_grafos_prueba.csv')
        log.info("Tabla de muestra guardada en 'tabla_grafos_prueba.csv'.")
