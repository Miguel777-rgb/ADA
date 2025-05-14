import igraph as ig
import logging as log
import pickle  # Para cargar el grafo
import polars as pl  # Para crear y manejar la tabla
import time

log.basicConfig(
    level=log.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        log.FileHandler("grafo_prueba.log", mode='w', encoding='utf-8'),
        log.StreamHandler()
    ]
)

def cargar_grafo_con_atributos(grafo_path, atributos_path):
    """Carga el grafo y sus atributos desde archivos pickle."""
    start_time = time.time()
    try:
        with open(grafo_path, 'rb') as f:
            g = pickle.load(f)
        log.info(f"Grafo cargado correctamente desde '{grafo_path}' en {time.time() - start_time:.4f} segundos.")
    except FileNotFoundError:
        log.error(f"Archivo del grafo no encontrado en '{grafo_path}'.")
        return None
    except Exception as e:
        log.error(f"Error al cargar el grafo desde '{grafo_path}': {e}")
        return None

    start_time_atributos = time.time()
    try:
        with open(atributos_path, 'rb') as f:
            atributos = pickle.load(f)
        for key, val in atributos.items():
            g.vs[key] = val
        log.info(f"Atributos cargados correctamente desde '{atributos_path}' en {time.time() - start_time_atributos:.4f} segundos.")
    except FileNotFoundError:
        log.warning(f"Archivo de atributos no encontrado en '{atributos_path}'.")
    except Exception as e:
        log.warning(f"Error al cargar atributos desde '{atributos_path}': {e}")
    return g

def verificar_insercion_nodos_aristas(grafo, num_nodos_esperado=None, num_aristas_esperado=None):
    """Verifica si el número de nodos y aristas coincide con lo esperado."""
    log.info("Verificando la inserción de nodos y aristas...")
    if grafo is None:
        log.warning("No se puede verificar, el grafo no se cargó correctamente.")
        return

    num_nodos_actual = grafo.vcount()
    num_aristas_actual = grafo.ecount()

    if num_nodos_esperado is not None:
        if num_nodos_actual == num_nodos_esperado:
            log.info(f"Número de nodos verificado: {num_nodos_actual} (esperado: {num_nodos_esperado}).")
        else:
            log.error(f"Error en el número de nodos: {num_nodos_actual} (esperado: {num_nodos_esperado}).")

    if num_aristas_esperado is not None:
        if num_aristas_actual == num_aristas_esperado:
            log.info(f"Número de aristas verificado: {num_aristas_actual} (esperado: {num_aristas_esperado}).")
        else:
            log.error(f"Error en el número de aristas: {num_aristas_actual} (esperado: {num_aristas_esperado}).")
    elif num_nodos_esperado is not None:
        log.info(f"Número de nodos: {num_nodos_actual}, número de aristas: {num_aristas_actual}.")
    else:
        log.info(f"Número de nodos: {num_nodos_actual}, número de aristas: {num_aristas_actual}.")

def verificar_indexacion_nodos(grafo, num_nodos_a_verificar=10):
    """Verifica la indexación de los primeros y últimos nodos del grafo."""
    log.info("Verificando la indexación de los nodos...")
    if grafo is None or grafo.vcount() == 0:
        log.warning("No se puede verificar la indexación, el grafo está vacío o no se cargó.")
        return

    n_total = grafo.vcount()
    indices_a_verificar = list(range(min(num_nodos_a_verificar, n_total))) + list(range(max(0, n_total - num_nodos_a_verificar), n_total))
    indices_a_verificar = sorted(list(set(indices_a_verificar)))

    for index in indices_a_verificar:
        try:
            nodo = grafo.vs[index]
            log.debug(f"Nodo con índice {index} encontrado.")
        except IndexError:
            log.error(f"Error: No se encontró ningún nodo con el índice {index}.")

    log.info("Verificación de indexación de nodos completada.")

def verificar_atributos_nodos(grafo, atributo_a_verificar='location', num_nodos_a_verificar=5):
    """Verifica la existencia y el tipo de un atributo específico en algunos nodos."""
    log.info(f"Verificando el atributo '{atributo_a_verificar}' en algunos nodos...")
    if grafo is None or grafo.vcount() == 0:
        log.warning("No se pueden verificar los atributos, el grafo está vacío o no se cargó.")
        return

    n_total = grafo.vcount()
    indices_a_verificar = list(range(min(num_nodos_a_verificar, n_total)))

    for index in indices_a_verificar:
        try:
            nodo = grafo.vs[index]
            if atributo_a_verificar in nodo.attributes():
                valor_atributo = nodo[atributo_a_verificar]
                log.debug(f"Nodo {index} tiene el atributo '{atributo_a_verificar}' con valor: {valor_atributo} (tipo: {type(valor_atributo)}).")
            else:
                log.warning(f"Nodo {index} no tiene el atributo '{atributo_a_verificar}'.")
        except IndexError:
            log.error(f"Error al acceder al nodo con índice {index} para verificar el atributo.")

    log.info(f"Verificación del atributo '{atributo_a_verificar}' completada.")

def verificar_conexiones(grafo, nodo_a_verificar=0, num_conexiones_esperadas=None):
    """Verifica las conexiones de un nodo específico."""
    log.info(f"Verificando las conexiones del nodo {nodo_a_verificar + 1} (índice {nodo_a_verificar})...")
    if grafo is None or nodo_a_verificar >= grafo.vcount():
        log.warning(f"No se pueden verificar las conexiones, el nodo con índice {nodo_a_verificar} no existe o el grafo no se cargó.")
        return

    conexiones_salientes = grafo.successors(nodo_a_verificar)
    num_conexiones = len(conexiones_salientes)
    log.info(f"El nodo {nodo_a_verificar + 1} tiene {num_conexiones} conexiones salientes.")
    log.debug(f"Conexiones salientes: {[c + 1 for c in conexiones_salientes]}.") # Mostrar en base 1

    if num_conexiones_esperadas is not None:
        if num_conexiones == num_conexiones_esperadas:
            log.info(f"Número de conexiones salientes verificado: {num_conexiones} (esperado: {num_conexiones_esperadas}).")
        else:
            log.error(f"Error en el número de conexiones salientes: {num_conexiones} (actual: {num_conexiones}, esperado: {num_conexiones_esperadas}).")

def generar_tabla_grafos_ordenados(grafo, num_nodos=50):
    """
    Genera una tabla con información de los primeros y últimos nodos del grafo.
    """
    if grafo is None:
        log.warning("No se puede generar la tabla, el grafo no se cargó correctamente.")
        return None

    nodos = list(range(grafo.vcount()))
    primeros_nodos = nodos[:num_nodos]
    ultimos_nodos = nodos[-num_nodos:]
    nodos_seleccionados = sorted(list(set(primeros_nodos + ultimos_nodos)))

    data = []
    for idx, nodo in enumerate(nodos_seleccionados, start=1):
        atributos_nodo = grafo.vs[nodo].attributes()
        lat = atributos_nodo.get('lat')
        lon = atributos_nodo.get('lon')
        location = atributos_nodo.get('location')
        num_conexiones = len(grafo.successors(nodo))
        data.append({
            'Index': idx,
            'Nodo': nodo + 1,  # Mostrar base 1
            'Latitud': lat if lat is not None else (location[0] if location else None),
            'Longitud': lon if lon is not None else (location[1] if location else None),
            'Conexiones': num_conexiones
        })
    tabla = pl.DataFrame(data)
    return tabla

log.info("Iniciando prueba de carga del grafo.")
# Ruta del archivo del grafo
grafo_archivo = 'grafo_igraph_paralelizado.pkl'
grafo_atributos_archivo = 'atributos_igraph_paralelizado.parquet'

# Cargar el grafo desde el archivo
grafo = cargar_grafo_con_atributos(grafo_archivo, grafo_atributos_archivo)

if grafo:
    # --- Pruebas de verificación ---
    # (Puedes descomentar y modificar estos parámetros según tu caso)

    # 1. Verificar el número total de nodos y aristas
    # Para hacer esta verificación útil, necesitarías saber de antemano
    # cuántos nodos y aristas DEBERÍA tener tu grafo.
    # verificar_insercion_nodos_aristas(grafo, num_nodos_esperado=1000000, num_aristas_esperado=...)

    # 2. Verificar la indexación de los nodos
    verificar_indexacion_nodos(grafo, num_nodos_a_verificar=10)

    # 3. Verificar la existencia de atributos en los nodos
    verificar_atributos_nodos(grafo, atributo_a_verificar='lat', num_nodos_a_verificar=5)
    verificar_atributos_nodos(grafo, atributo_a_verificar='lon', num_nodos_a_verificar=5)

    # 4. Verificar las conexiones de un nodo específico
    verificar_conexiones(grafo, nodo_a_verificar=0, num_conexiones_esperadas=None) # Puedes especificar el número esperado

    # --- Generar y guardar tabla de muestra ---
    tabla_grafos = generar_tabla_grafos_ordenados(grafo, num_nodos=50)
    if tabla_grafos is not None:
        log.info(f"\n{tabla_grafos}")
        tabla_grafos.write_csv('tabla_grafos_prueba.csv')
        log.info("Tabla de muestra guardada en 'tabla_grafos_prueba.csv'.")