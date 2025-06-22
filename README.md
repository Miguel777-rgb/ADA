# ADA
# Explicación Detallada del Código

Este script de Python está diseñado para construir un grafo a partir de datos de ubicaciones y usuarios, y luego realizar un análisis exploratorio de datos (EDA) básico. Utiliza las bibliotecas `igraph`, `time`, `logging`, `polars`, `pickle`, y `numpy`.
Drive (contiene el grafo.pkl):
https://drive.google.com/drive/folders/1gLylv7wahcpLzcefsCbAXcYFj1ut7yfo?usp=sharing 

¡Excelente! He preparado una explicación completa y detallada de todo el proyecto, perfecta para tu repositorio de GitHub. La estructura está pensada para guiar a cualquier persona desde los conceptos teóricos hasta la ejecución y comprensión de cada línea de código.

---

# Análisis de Redes Geográficas a Gran Escala con Python

Este repositorio contiene un proyecto completo para la generación, construcción y análisis de un grafo a gran escala con datos geográficos. El flujo de trabajo abarca desde la creación de datos sintéticos hasta la aplicación de algoritmos de grafos como Búsqueda en Profundidad (DFS), Dijkstra para caminos más cortos y Detección de Comunidades (LPA), culminando en visualizaciones interactivas.

## Índice

1.  [Descripción del Proyecto](#descripción-del-proyecto)
2.  [Temas Clave y Conceptos](#temas-clave-y-conceptos)
3.  [Librerías y Dependencias](#librerías-y-dependencias)
4.  [Cómo Recrear el Proyecto](#cómo-recrear-el-proyecto)
5.  [Explicación Detallada de los Scripts](#explicación-detallada-de-los-scripts)
    *   [0-createdata.py](#-archivo-0-createdatapy)
    *   [1-grapy_paralel.py](#-archivo-1-grapy-paralelpy)
    *   [2-prueba_graphote.py](#-archivo-2-prueba-graphotepy)
    *   [3-dfs.py](#-archivo-3-dfspy)
    *   [4-dijkstra.py](#-archivo-4-dijkstrapy)
    *   [5-community.py](#-archivo-5-communitypy)

## Descripción del Proyecto

El objetivo de este proyecto es demostrar un pipeline de análisis de grafos de principio a fin, manejando una cantidad significativa de datos (millones de nodos y aristas).

El proceso se divide en los siguientes pasos:

1.  **Generación de Datos**: Se crean archivos de texto con millones de ubicaciones geográficas (latitud, longitud) y conexiones de usuario (quién se conecta con qué ubicación).
2.  **Construcción del Grafo**: Se leen los datos generados y se construye un objeto de grafo dirigido utilizando la librería `python-igraph`. Este proceso está optimizado para ser ejecutado en paralelo, aprovechando todos los núcleos de la CPU.
3.  **Verificación del Grafo**: Se ejecuta un script de prueba para asegurar que el grafo se ha construido correctamente, verificando el número de nodos/aristas, sus atributos y conexiones.
4.  **Análisis Algorítmico**: Se aplican tres algoritmos fundamentales de la teoría de grafos:
    *   **DFS (Depth-First Search)**: Para explorar un componente conectado del grafo y visualizar la travesía.
    *   **Dijkstra**: Para encontrar el camino más corto (geográficamente) entre los dos nodos más distantes del grafo.
    *   **LPA (Label Propagation Algorithm)**: Para detectar comunidades o clústeres de nodos densamente conectados.
5.  **Visualización**: Los resultados de los análisis se presentan en mapas interactivos generados con `folium`, lo que permite una exploración intuitiva de los datos y los resultados algorítmicos.

## Temas Clave y Conceptos

*   **Teoría de Grafos**: Nodos (vértices), aristas (conexiones), grafos dirigidos, grado de un nodo, componentes conectados.
*   **Algoritmos de Grafos**:
    *   **DFS**: Un algoritmo de recorrido que explora tan lejos como sea posible a lo largo de cada rama antes de retroceder.
    *   **Dijkstra**: Un algoritmo para encontrar los caminos más cortos entre nodos en un grafo ponderado.
    *   **Detección de Comunidades (LPA)**: Un algoritmo para encontrar grupos de nodos que están más densamente conectados entre sí que con el resto del grafo.
*   **Computación Paralela**: Uso del módulo `multiprocessing` de Python para acelerar tareas computacionalmente intensivas (como el procesamiento de archivos de texto grandes) dividiendo el trabajo entre múltiples núcleos de CPU.
*   **Visualización de Datos Geográficos**: Mapeo de coordenadas (latitud, longitud) a un mapa interactivo. Uso de marcadores, líneas y clústeres para representar datos de manera efectiva.
*   **Manejo de Grandes Volúmenes de Datos**: Técnicas como la lectura de archivos línea por línea, el uso de generadores y la recolección de basura para trabajar con datos que podrían no caber completamente en la RAM de una sola vez.
*   **Serialización de Datos**: Uso de `pickle` para guardar y cargar objetos complejos de Python (como un grafo de `igraph`) de manera eficiente.

## Librerías y Dependencias

Para ejecutar este proyecto, necesitarás Python 3.7 o superior y las siguientes librerías. Puedes instalarlas usando `pip`.

*   `python-igraph`: La librería principal para la creación y análisis de grafos. Es extremadamente rápida ya que su núcleo está escrito en C.
*   `polars`: Una librería de DataFrames ultrarrápida (alternativa a pandas) usada en el script de prueba.
*   `folium`: Para crear los mapas interactivos en HTML.
*   `matplotlib`: Utilizada aquí para generar escalas de colores para la visualización de comunidades.
*   `tqdm`: Para mostrar barras de progreso elegantes y útiles en procesos largos.

Crea un archivo `requirements.txt` con el siguiente contenido:

```txt
python-igraph
polars
folium
matplotlib
tqdm
```

Y luego instálalas con:

```bash
pip install -r requirements.txt
```

## Cómo Recrear el Proyecto

Sigue estos pasos en orden para ejecutar el proyecto desde cero:

1.  **Clona el Repositorio**:
    ```bash
    git clone <URL-de-tu-repositorio>
    cd <nombre-del-repositorio>
    ```

2.  **Crea un Entorno Virtual (Recomendado)**:
    ```bash
    python -m venv venv
    source venv/bin/activate  # En Windows: venv\Scripts\activate
    ```

3.  **Instala las Dependencias**:
    ```bash
    pip install -r requirements.txt
    ```

4.  **Ejecuta los Scripts en Orden**: El orden es crucial ya que cada script depende de los archivos generados por el anterior.

    ```bash
    # 1. Genera los archivos de datos (esto puede tardar un poco)
    python 0-createdata.py

    # 2. Construye el grafo a partir de los datos (el paso más largo)
    python 1-grapy_paralel.py

    # 3. (Opcional) Verifica la integridad del grafo generado
    python 2-prueba_graphote.py

    # 4. Ejecuta los scripts de análisis (pueden ejecutarse en cualquier orden)
    python 3-dfs.py
    python 4-dijkstra.py
    python 5-community.py
    ```

5.  **Revisa los Resultados**: Después de ejecutar los scripts, encontrarás varios archivos de salida:
    *   `*.log`: Archivos de registro con detalles de la ejecución de cada script.
    *   `*.pkl`: El objeto del grafo serializado.
    *   `*.html`: Mapas interactivos que puedes abrir en tu navegador.
    *   `*.csv`: Una tabla de muestra del grafo generada por el script de prueba.

---

## Explicación Detallada de los Scripts

A continuación, se detalla cada script línea por línea.

### ➤ Archivo: `0-createdata.py`

**Propósito**: Generar dos archivos de texto: uno con coordenadas geográficas (`1_million_location.txt`) y otro con conexiones entre usuarios y ubicaciones (`1_million_user.txt`).

```python
# Importa las librerías necesarias.
import random  # Para generar números aleatorios.
import logging # Para registrar mensajes de estado y errores.

# Configuración del logging.
logging.basicConfig(
    # Establece el nivel mínimo de mensajes a registrar (INFO, WARNING, ERROR, etc.).
    level=logging.INFO,
    # Define el formato de cada mensaje de log.
    format='%(asctime)s - %(levelname)s - %(message)s',
    # Especifica a dónde enviar los logs.
    handlers=[
        # Un manejador para escribir los logs en un archivo llamado "creardata.log".
        # mode='w' significa que el archivo se sobrescribe en cada ejecución.
        # encoding='utf-8' asegura la compatibilidad con caracteres especiales.
        logging.FileHandler("creardata.log", mode='w', encoding='utf-8'),
        # Otro manejador para mostrar los logs en la consola.
        logging.StreamHandler()
    ]
)

# Define una función para generar el archivo de ubicaciones.
def generar_ubicaciones(num_ubicaciones, nombre_archivo="1_million_location.txt"):
    try: # Inicia un bloque try-except para manejar posibles errores de archivo.
        # Abre el archivo especificado en modo escritura ('w').
        # 'with' asegura que el archivo se cierre automáticamente al final.
        with open(nombre_archivo, 'w') as archivo:
            # Bucle que se repite 'num_ubicaciones' veces.
            # El guion bajo (_) es una convención para una variable que no se usará.
            for _ in range(num_ubicaciones):
                # Genera una latitud aleatoria, un número flotante entre -90 y 90.
                latitud = random.uniform(-90, 90)
                # Genera una longitud aleatoria, un número flotante entre -180 y 180.
                longitud = random.uniform(-180, 180)
                # Escribe la coordenada en el archivo usando un f-string.
                # \n es el carácter de nueva línea, para que cada coordenada esté en una línea separada.
                archivo.write(f"{latitud},{longitud}\n")
        # Imprime un mensaje de éxito en la consola.
        print(f"Se ha generado el archivo '{nombre_archivo}' con {num_ubicaciones} ubicaciones.")
    except Exception as e: # Si ocurre cualquier error en el bloque 'try'.
        # Imprime un mensaje de error detallado.
        print(f"Ocurrió un error al generar el archivo: {e}")

# Define una función para generar el archivo de conexiones de usuario.
def generar_conexiones(num_conexiones, nombre_archivo="1_million_user.txt"):
    """
    Docstring que explica lo que hace la función, sus argumentos (Args) y su propósito.
    """
    try: # Bloque para manejo de errores.
        # Abre el archivo especificado en modo escritura.
        with open(nombre_archivo, 'w') as archivo:
            # Bucle que se repite 'num_conexiones' veces, usando 'i' como el ID del usuario.
            for i in range(num_conexiones):
                # Genera un número entero aleatorio entre 0 y 100 para las visitas.
                num_ubicaciones_visitadas = random.randint(0, 100)
                # Crea una lista de IDs de ubicaciones visitadas.
                # random.randint(1, 1000700) genera un ID de ubicación aleatorio.
                # Se asume que los IDs de ubicación van de 1 a 1,000,700.
                # Esto se hace 'num_ubicaciones_visitadas' veces usando una list comprehension.
                ubicaciones_visitadas = [random.randint(1, 1000700) for _ in range(num_ubicaciones_visitadas)]
                # Escribe la línea en el archivo.
                # f"{i}," es el ID del usuario (el origen de la conexión).
                # ','.join(map(str, ubicaciones_visitadas)) convierte cada ID de ubicación a string
                # y luego los une todos con comas.
                archivo.write(f"{i},{','.join(map(str, ubicaciones_visitadas))}\n")
        # Imprime un mensaje de éxito.
        print(f"Se ha generado el archivo '{nombre_archivo}' con {num_conexiones} conexiones.")
    except Exception as e: # Manejo de errores.
        print(f"Ocurrió un error al generar el archivo: {e}")

# Este bloque se ejecuta solo si el script es el programa principal.
if __name__ == "__main__":
    # Llama a la función para generar 1,000,000 de ubicaciones.
    generar_ubicaciones(1000000)
    # Llama a la función para generar 1,000,000 de registros de conexiones de usuario.
    generar_conexiones(1000000)
```

### ➤ Archivo: `1-grapy_paralel.py`

**Propósito**: Construir un grafo `igraph` a partir de los archivos de datos. Utiliza procesamiento en paralelo para acelerar la lectura del archivo de conexiones, que es la parte más lenta.

```python
# Importa las librerías necesarias.
import igraph as ig                 # Librería principal para grafos.
import time                         # Para medir el tiempo de ejecución.
import logging                      # Para registrar el progreso.
import fileinput                    # Para leer archivos de forma eficiente.
import pickle                       # Para guardar/cargar objetos Python (el grafo).
import gc                           # Garbage Collector, para liberar memoria manualmente.
import multiprocessing as mp        # Para procesamiento en paralelo.
from itertools import chain         # Para concatenar iterables eficientemente.

# Define una constante para el número máximo de nodos a procesar.
NUM_NODOS = 10_000_000

# Configuración del logging (similar al script anterior).
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("grafo_paralelizado.log", mode='w', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

# Función para cargar las ubicaciones desde el archivo.
def cargar_ubicaciones_directo(ubicaciones_path, max_nodos):
    logging.info(f"Cargando un máximo de {max_nodos} ubicaciones desde el archivo...")
    latitudes = []  # Lista para almacenar latitudes.
    longitudes = [] # Lista para almacenar longitudes.
    with open(ubicaciones_path, 'r') as f: # Abre el archivo de ubicaciones en modo lectura.
        # Itera sobre cada línea del archivo, con un índice 'i'.
        for i, line in enumerate(f):
            if i >= max_nodos: # Si ya hemos leído el máximo de nodos, nos detenemos.
                logging.info(f"Límite de {max_nodos} nodos alcanzado. Se detiene la lectura de ubicaciones.")
                break
            try:
                # Divide la línea por la coma para obtener latitud y longitud como strings.
                lat_str, lon_str = line.strip().split(',')
                # Convierte los strings a flotantes y los añade a las listas.
                latitudes.append(float(lat_str))
                longitudes.append(float(lon_str))
            except ValueError: # Si la línea no tiene el formato esperado.
                logging.warning(f"Línea malformada en el archivo de ubicaciones: '{line.strip()}'")

    # Combina las listas de latitud y longitud en una lista de tuplas (lat, lon).
    ubicaciones = list(zip(latitudes, longitudes))
    # El número real de nodos es la cantidad de ubicaciones que logramos cargar.
    num_nodos_reales = len(ubicaciones)
    
    # Advierte si el archivo tenía menos nodos de los que esperábamos.
    if num_nodos_reales < max_nodos:
        logging.warning(f"El archivo de ubicaciones contiene {num_nodos_reales} nodos, menos que el máximo esperado de {max_nodos}.")
    
    logging.info(f"Se cargaron {num_nodos_reales} ubicaciones.")
    # Devuelve las ubicaciones, el número real de nodos y un diccionario de atributos.
    return ubicaciones, num_nodos_reales, {'lat': latitudes, 'lon': longitudes}

# Función "worker" que será ejecutada por cada proceso en paralelo.
# Procesa una sola línea del archivo de usuarios.
def procesar_linea_usuarios(linea_idx_contenido, num_nodos_max):
    idx, linea = linea_idx_contenido # Desempaqueta el índice y el contenido de la línea.
    aristas_locales = [] # Lista para las aristas (conexiones) encontradas en esta línea.
    conexiones = set() # Un 'set' para evitar conexiones duplicadas desde el mismo origen.
    # Divide la línea por las comas para obtener los IDs de las ubicaciones.
    for x in linea.strip().split(','):
        x_strip = x.strip() # Limpia espacios en blanco.
        if x_strip.isdigit(): # Comprueba si es un número válido.
            dst = int(x_strip) # Convierte el string a entero (nodo de destino).
            # Valida que el nodo de destino esté dentro del rango de nodos existentes.
            if 1 <= dst <= num_nodos_max:
                conexiones.add(dst - 1)  # Añade la conexión. Restamos 1 para indexación base 0.
    
    # Valida que el nodo de origen (el ID del usuario) también sea válido.
    if 0 <= (idx - 1) < num_nodos_max:
        # Crea las tuplas de arista (origen, destino) para todas las conexiones.
        aristas_locales.extend((idx - 1, dst) for dst in conexiones)
    
    return aristas_locales # Devuelve la lista de aristas de esta línea.

# Función que orquesta el procesamiento en paralelo del archivo de usuarios.
def procesar_usuarios_paralelizado(usuarios_path, num_nodos, num_procesos=mp.cpu_count()):
    logging.info(f"Procesando usuarios para {num_nodos} nodos en paralelo con {num_procesos} procesos...")
    start_time_procesamiento = time.time()
    
    # 'fileinput.input' permite leer el archivo de forma eficiente.
    with fileinput.input(usuarios_path) as f:
        # Crea un "generador". No carga todo el archivo en memoria.
        # Solo procesa las líneas hasta el límite de 'num_nodos'.
        lineas_a_procesar = ((idx, linea) for idx, linea in enumerate(f, start=1) if idx <= num_nodos)
        
        # Crea un "pool" de procesos trabajadores.
        with mp.Pool(processes=num_procesos) as pool:
            # 'starmap' distribuye el trabajo (las líneas) entre los procesos del pool.
            # Llama a 'procesar_linea_usuarios' para cada ítem en 'lineas_a_procesar'.
            resultados = pool.starmap(procesar_linea_usuarios, [(item, num_nodos) for item in lineas_a_procesar])
            
    # 'resultados' es una lista de listas. 'chain.from_iterable' la "aplana" en una sola lista de aristas.
    aristas = list(chain.from_iterable(resultados))
    
    # Un filtro final por seguridad, aunque la validación ya se hizo en el worker.
    aristas_filtradas = [(src, dst) for src, dst in aristas if 0 <= src < num_nodos and 0 <= dst < num_nodos]
    
    logging.info(f"Se procesaron {len(aristas_filtradas)} aristas en {time.time() - start_time_procesamiento:.2f} s.")
    return aristas_filtradas

# Función principal que crea y guarda el grafo.
def crear_grafo_igraph_paralelizado(ubicaciones_path, usuarios_path, output_grafo_path, max_nodos):
    start_time = time.time()

    # 1. Carga las ubicaciones y obtiene el número real de nodos.
    ubicaciones, num_nodos_reales, atributos_ubicacion = cargar_ubicaciones_directo(ubicaciones_path, max_nodos)
    
    # 2. Procesa las conexiones de usuario en paralelo, usando el número real de nodos.
    aristas = procesar_usuarios_paralelizado(usuarios_path, num_nodos_reales)

    logging.info("Creando grafo de igraph...")
    # Crea un objeto de grafo vacío y dirigido.
    g = ig.Graph(directed=True)
    # Añade el número exacto de vértices (nodos) al grafo.
    g.add_vertices(num_nodos_reales)
    # Añade todas las aristas (conexiones) de una sola vez, que es muy eficiente.
    g.add_edges(aristas)

    # Añade los atributos de latitud y longitud a los vértices del grafo.
    for key, values in atributos_ubicacion.items():
        g.vs[key] = values # Asigna la lista completa de valores al atributo correspondiente.

    # Guarda el objeto de grafo completo en un archivo usando pickle.
    logging.info(f"Guardando grafo con atributos en '{output_grafo_path}'...")
    with open(output_grafo_path, 'wb') as f: # 'wb' es para escritura en modo binario.
        # 'pickle.dump' serializa el objeto 'g' en el archivo 'f'.
        # 'HIGHEST_PROTOCOL' es para la máxima compresión y eficiencia.
        pickle.dump(g, f, protocol=pickle.HIGHEST_PROTOCOL)

    logging.info(f"Grafo guardado. Tiempo total: {time.time() - start_time:.2f} s")
    # Libera la memoria de los objetos grandes que ya no se necesitan.
    del ubicaciones, aristas, atributos_ubicacion, g
    gc.collect() # Llama explícitamente al recolector de basura.

if __name__ == "__main__":
    ubicaciones_archivo = '10_million_location.txt'
    usuarios_archivo = '10_million_user.txt'
    grafo_archivo = 'grafo_igraph_paralelizado.pkl'

    # Llama a la función principal para crear el grafo.
    crear_grafo_igraph_paralelizado(ubicaciones_archivo, usuarios_archivo, grafo_archivo, max_nodos=NUM_NODOS)
    
    gc.collect()
```
### ➤ Archivo: `2-prueba_graphote.py`

**Propósito**: Realizar una serie de pruebas sobre el archivo de grafo (`.pkl`) generado para verificar su integridad. Comprueba el número de nodos, aristas, atributos y conexiones, y genera una tabla de muestra.

```python
# Importa las librerías necesarias.
import igraph as ig      # Para trabajar con el objeto grafo.
import logging as log    # Para los mensajes de registro.
import pickle            # Para cargar el grafo desde el archivo .pkl.
import polars as pl      # Para crear y manejar la tabla de datos (DataFrame).
import time              # Para medir tiempos de carga.

# Configuración del logging.
log.basicConfig(
    level=log.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        log.FileHandler("grafo_pruebas.log", mode='w', encoding='utf-8'),
        log.StreamHandler()
    ]
)

# Función para cargar el grafo y sus atributos.
# NOTA: Este script asume un archivo de atributos separado que no se usa en los otros scripts,
# pero está diseñado para ser robusto.
def cargar_grafo(grafo_path, atributos_path):
    start_time = time.time()
    try:
        # Abre el archivo .pkl en modo lectura binaria ('rb').
        with open(grafo_path, 'rb') as f:
            # Carga el objeto grafo desde el archivo.
            g = pickle.load(f)
        log.info(f"Grafo cargado correctamente desde '{grafo_path}' en {time.time() - start_time:.4f} segundos.")
    except FileNotFoundError:
        log.error(f"Archivo del grafo no encontrado en '{grafo_path}'.")
        return None
    except Exception as e:
        log.error(f"Error al cargar el grafo desde '{grafo_path}': {e}")
        return None

    # Intenta cargar un archivo de atributos adicional.
    start_time_atributos = time.time()
    try:
        with open(atributos_path, 'rb') as f:
            atributos = pickle.load(f)
        # Asigna los atributos cargados al grafo.
        for key, val in atributos.items():
            g.vs[key] = val
        log.info(f"Atributos cargados correctamente desde '{atributos_path}' en {time.time() - start_time_atributos:.4f} segundos.")
    except FileNotFoundError:
        log.warning(f"Archivo de atributos no encontrado en '{atributos_path}'.") # Es una advertencia, no un error.
    except Exception as e:
        log.warning(f"Error al cargar atributos desde '{atributos_path}': {e}")
    return g

# Función para verificar el número de nodos y aristas.
def verificar_insercion_nodos_aristas(grafo, num_nodos_esperado=None, num_aristas_esperado=None):
    log.info("Verificando la inserción de nodos y aristas...")
    if grafo is None:
        log.warning("No se puede verificar, el grafo no se cargó correctamente.")
        return

    # Obtiene el número de vértices (nodos) y aristas del grafo.
    num_nodos_actual = grafo.vcount()
    num_aristas_actual = grafo.ecount()

    if num_nodos_esperado is not None: # Si se proporcionó un número esperado.
        if num_nodos_actual == num_nodos_esperado:
            log.info(f"Número de nodos verificado: {num_nodos_actual} (esperado: {num_nodos_esperado}).")
        else:
            log.error(f"Error en el número de nodos: {num_nodos_actual} (esperado: {num_nodos_esperado}).")
    
    # (El código se repite para aristas, pero la lógica es la misma)
    # ...

# Función para verificar que se puede acceder a los nodos por su índice.
def verificar_indexacion_nodos(grafo, num_nodos_a_verificar=10):
    log.info("Verificando la indexación de los nodos...")
    if grafo is None or grafo.vcount() == 0:
        log.warning("No se puede verificar la indexación, el grafo está vacío o no se cargó.")
        return

    n_total = grafo.vcount()
    # Crea una lista de índices a verificar: los primeros y los últimos.
    indices_a_verificar = list(range(min(num_nodos_a_verificar, n_total))) + list(range(max(0, n_total - num_nodos_a_verificar), n_total))
    indices_a_verificar = sorted(list(set(indices_a_verificar))) # Elimina duplicados y ordena.

    for index in indices_a_verificar:
        try:
            nodo = grafo.vs[index] # Intenta acceder al vértice por su índice.
            log.debug(f"Nodo con índice {index} encontrado.")
        except IndexError:
            log.error(f"Error: No se encontró ningún nodo con el índice {index}.")
    log.info("Verificación de indexación de nodos completada.")

# Función para verificar que los nodos tienen los atributos esperados.
def verificar_atributos_nodos(grafo, atributo_a_verificar='location', num_nodos_a_verificar=5):
    log.info(f"Verificando el atributo '{atributo_a_verificar}' en algunos nodos...")
    if grafo is None or grafo.vcount() == 0:
        log.warning("No se pueden verificar los atributos, el grafo está vacío o no se cargó.")
        return

    n_total = grafo.vcount()
    indices_a_verificar = list(range(min(num_nodos_a_verificar, n_total))) # Verifica los primeros N nodos.

    for index in indices_a_verificar:
        try:
            nodo = grafo.vs[index]
            # Comprueba si el atributo existe en el nodo.
            if atributo_a_verificar in nodo.attributes():
                valor_atributo = nodo[atributo_a_verificar]
                log.debug(f"Nodo {index} tiene el atributo '{atributo_a_verificar}' con valor: {valor_atributo} (tipo: {type(valor_atributo)}).")
            else:
                log.warning(f"Nodo {index} no tiene el atributo '{atributo_a_verificar}'.")
        except IndexError:
            log.error(f"Error al acceder al nodo con índice {index} para verificar el atributo.")
    log.info(f"Verificación del atributo '{atributo_a_verificar}' completada.")

# Función para verificar las conexiones de un nodo específico.
def verificar_conexiones(grafo, nodo_a_verificar=0, num_conexiones_esperadas=None):
    log.info(f"Verificando las conexiones del nodo {nodo_a_verificar + 1} (índice {nodo_a_verificar})...")
    if grafo is None or nodo_a_verificar >= grafo.vcount():
        log.warning(f"No se pueden verificar las conexiones, el nodo con índice {nodo_a_verificar} no existe o el grafo no se cargó.")
        return

    # Obtiene los "sucesores" del nodo, es decir, los nodos a los que apunta en un grafo dirigido.
    conexiones_salientes = grafo.successors(nodo_a_verificar)
    num_conexiones = len(conexiones_salientes)
    log.info(f"El nodo {nodo_a_verificar + 1} tiene {num_conexiones} conexiones salientes.")
    # ... (código de verificación)

# Función para crear una tabla con una muestra de los datos del grafo.
def generar_tabla_grafos_ordenados(grafo, num_nodos=50):
    if grafo is None:
        log.warning("No se puede generar la tabla, el grafo no se cargó correctamente.")
        return None

    nodos = list(range(grafo.vcount()))
    # Selecciona los primeros 'num_nodos' y los últimos 'num_nodos'.
    primeros_nodos = nodos[:num_nodos]
    ultimos_nodos = nodos[-num_nodos:]
    nodos_seleccionados = sorted(list(set(primeros_nodos + ultimos_nodos)))

    data = [] # Lista para almacenar los datos de la tabla.
    for idx, nodo in enumerate(nodos_seleccionados, start=1):
        atributos_nodo = grafo.vs[nodo].attributes()
        # Obtiene los atributos de forma segura con .get(), que devuelve None si la clave no existe.
        lat = atributos_nodo.get('lat')
        lon = atributos_nodo.get('lon')
        location = atributos_nodo.get('location') # Atributo legado
        num_conexiones = len(grafo.successors(nodo))
        # Añade un diccionario con la información del nodo a la lista.
        data.append({
            'Index': idx,
            'Nodo': nodo + 1,  # Muestra el ID del nodo en base 1.
            'Latitud': lat if lat is not None else (location[0] if location else None),
            'Longitud': lon if lon is not None else (location[1] if location else None),
            'Conexiones': num_conexiones
        })
    # Crea un DataFrame de Polars a partir de la lista de diccionarios.
    tabla = pl.DataFrame(data)
    return tabla

log.info("Iniciando prueba de carga del grafo.")
# Rutas de los archivos de entrada.
grafo_archivo = 'grafo_igraph_paralelizado.pkl'
# Este archivo no se crea en el pipeline, por lo que la carga fallará con una advertencia, como se espera.
grafo_atributos_archivo = 'atributos_igraph_paralelizado.parquet'

# Carga el grafo.
grafo = cargar_grafo(grafo_archivo, grafo_atributos_archivo)

if grafo:
    # Llama a todas las funciones de verificación.
    verificar_indexacion_nodos(grafo, num_nodos_a_verificar=10)
    verificar_atributos_nodos(grafo, atributo_a_verificar='lat', num_nodos_a_verificar=5)
    verificar_atributos_nodos(grafo, atributo_a_verificar='lon', num_nodos_a_verificar=5)
    verificar_conexiones(grafo, nodo_a_verificar=0, num_conexiones_esperadas=None)

    # Genera y muestra la tabla de muestra.
    tabla_grafos = generar_tabla_grafos_ordenados(grafo, num_nodos=50)
    if tabla_grafos is not None:
        log.info(f"\n{tabla_grafos}") # Imprime la tabla en la consola.
        tabla_grafos.write_csv('tabla_grafos_prueba.csv') # Guarda la tabla en un archivo CSV.
        log.info("Tabla de muestra guardada en 'tabla_grafos_prueba.csv'.")
```
### ➤ Archivo: `3-dfs.py`

**Propósito**: Realizar una travesía DFS (Búsqueda en Profundidad) desde un nodo aleatorio y visualizar el camino recorrido en un mapa interactivo.

```python
# Importa las librerías necesarias.
import pickle
import time
import logging
import random
import igraph as ig
from typing import Optional, List, Tuple # Para anotaciones de tipo, mejoran la legibilidad.
import folium # Para crear el mapa.
from folium.plugins import BeautifyIcon # Para iconos de marcador personalizados.

# Configuración del logging.
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
handlers=[logging.FileHandler("analisis_dfs.log", mode='w', encoding='utf-8'),
logging.StreamHandler()])

# Función para cargar el grafo desde el archivo .pkl.
def cargar_grafo(grafo_path: str) -> Optional[ig.Graph]:
    logging.info(f"Cargando el grafo desde '{grafo_path}'...")
    start_time = time.time()
    try:
        with open(grafo_path, 'rb') as f:
            g = pickle.load(f)
        end_time = time.time()
        logging.info(f"Grafo cargado en {end_time - start_time:.2f} segundos. {g.summary()}")
        return g
    except FileNotFoundError: # Manejo de error específico.
        logging.error(f"Error crítico: El archivo de grafo '{grafo_path}' no fue encontrado.")
        return None
    except Exception as e:
        logging.error(f"Error crítico al cargar el grafo: {e}")
        return None

# --- ALGORITMO DFS ---
# Implementación de la Búsqueda en Profundidad (DFS).
def dfs_algoritmo(graph: ig.Graph, source: int) -> Tuple[List[int], float]:
    logging.info(f"Iniciando travesía DFS completa desde el nodo {source}.")
    start_time = time.time()
    stack = [source] # La 'pila' para el DFS, inicializada con el nodo de origen.
    visitados = {source} # Un 'set' para llevar registro de los nodos ya visitados (búsqueda rápida).
    nodos_explorados_en_orden = [] # Lista para guardar el orden en que se exploran los nodos.
    
    while stack: # Mientras la pila no esté vacía.
        current_node = stack.pop() # Saca el último nodo de la pila.
        nodos_explorados_en_orden.append(current_node)
        # Itera sobre los vecinos del nodo actual en orden inverso.
        # 'reversed' ayuda a que la exploración sea más predecible (profundiza por el vecino de menor índice primero).
        for neighbor in reversed(graph.neighbors(current_node, mode='out')):
            if neighbor not in visitados: # Si el vecino no ha sido visitado.
                visitados.add(neighbor) # Lo marca como visitado.
                stack.append(neighbor) # Lo añade a la pila para explorarlo después.
    end_time = time.time()
    logging.info(f"Travesía DFS completada en {end_time - start_time:.4f} segundos. Total de nodos visitados: {len(nodos_explorados_en_orden)}.")
    # Devuelve la lista de nodos en el orden en que fueron visitados y el tiempo de ejecución.
    return nodos_explorados_en_orden, end_time - start_time

# --- Función de Visualización ---
# Crea un mapa de Folium con múltiples capas para visualizar el resultado del DFS.
def crear_mapa_multicapa_dfs(
    graph: ig.Graph, 
    source_node: int,
    nodos_visitados: List[int], 
    max_puntos_linea: int = 5000, # Límite para el número de puntos en la línea del mapa (rendimiento).
    num_nodos_muestra: int = 2000 # Límite para el número de marcadores de puntos (rendimiento).
) -> Optional[folium.Map]:
    
    if not nodos_visitados: # Si el DFS no encontró ningún nodo.
        logging.warning("No hay nodos visitados para mostrar en el mapa.")
        return None

    logging.info("Creando mapa multicapa con visualización consistente.")
    try:
        # Coordenadas del nodo de inicio para centrar el mapa.
        start_coords = (graph.vs[source_node]['lat'], graph.vs[source_node]['lon'])
        # Crea el objeto de mapa base, con un estilo minimalista (CartoDB positron).
        m = folium.Map(location=start_coords, zoom_start=4, tiles="CartoDB positron")
        
        # --- 1. Submuestreo del camino para la visualización ---
        # Si el camino es muy largo, se toma una muestra para no sobrecargar el navegador.
        if len(nodos_visitados) > max_puntos_linea:
            # Calcula un 'paso' para tomar un nodo de cada 'step' nodos.
            step = len(nodos_visitados) // max_puntos_linea
            camino_visualizado = nodos_visitados[::step]
            # Asegura que el último nodo del camino real siempre esté incluido.
            if camino_visualizado[-1] != nodos_visitados[-1]:
                camino_visualizado.append(nodos_visitados[-1])
        else:
            camino_visualizado = nodos_visitados
        
        # --- CAPA 1: La línea que dibuja el camino ---
        # FeatureGroup es una capa que se puede activar/desactivar en el mapa.
        fg_camino = folium.FeatureGroup(name="Camino DFS (visual)", show=True)
        # Extrae las coordenadas (lat, lon) de los nodos del camino visualizado.
        puntos_del_camino = [(graph.vs[n]['lat'], graph.vs[n]['lon']) for n in camino_visualizado if graph.vs[n]['lat'] is not None]
        # Crea un objeto PolyLine para dibujar la línea en el mapa.
        folium.PolyLine(
            puntos_del_camino, color="#ffaf0e", weight=2, opacity=0.8,
            tooltip=f"Camino DFS ({len(camino_visualizado)} de {len(nodos_visitados)} nodos)"
        ).add_to(fg_camino)
        m.add_child(fg_camino)

        # --- CAPA 2: Una muestra de puntos a lo largo del camino ---
        fg_puntos = folium.FeatureGroup(name=f"Muestra de Nodos (del camino visual)", show=True)
        end_node_id = nodos_visitados[-1]
        
        # Los candidatos para la muestra son los nodos del camino visualizado (no el original).
        # Esto asegura que los puntos que se dibujan caen sobre la línea visible.
        candidatos = [n for n in camino_visualizado if n != source_node and n != end_node_id]
        
        # Toma una muestra aleatoria de los candidatos.
        num_muestras_a_tomar = min(len(candidatos), num_nodos_muestra - 2)
        nodos_a_mostrar_aleatorios = random.sample(candidatos, num_muestras_a_tomar) if num_muestras_a_tomar > 0 else []

        # La muestra final siempre incluye el nodo de inicio y fin, más los aleatorios.
        nodos_muestra_final = [source_node, end_node_id] + nodos_a_mostrar_aleatorios
        
        for node_id in nodos_muestra_final:
            coords = (graph.vs[node_id]['lat'], graph.vs[node_id]['lon'])
            if coords[0] is not None:
                # Dibuja un pequeño círculo para cada nodo de la muestra.
                folium.CircleMarker(
                    location=coords, radius=3.5, color='#1f77b4', fill=True, fill_color='#1f77b4',
                    fill_opacity=0.7, popup=f"Nodo de muestra: {node_id}"
                ).add_to(fg_puntos)
        m.add_child(fg_puntos)

        # --- MARCADORES de Inicio y Fin ---
        # Añade marcadores especiales y permanentes para el inicio y fin del DFS.
        folium.Marker(
            location=start_coords, popup=f"INICIO DFS: Nodo {source_node}", 
            tooltip="Punto de Inicio", icon=BeautifyIcon(icon='play', border_color='#2ca02c', text_color='#2ca02c'), z_index_offset=1000
        ).add_to(m)
        # ... (código similar para el marcador de fin)

        # --- Añade el control para activar/desactivar las capas ---
        folium.LayerControl().add_to(m)

        return m
    except Exception as e:
        logging.error(f"No se pudo crear el mapa. Error: {e}")
        return None

# --- Bloque Principal ---
if __name__ == "__main__":
    GRAFO_PKL_ENTRADA = 'grafo_igraph_paralelizado.pkl'
    MAPA_HTML_SALIDA = 'mapa_dfs_multicapa_corregido.html'

    mi_grafo = cargar_grafo(GRAFO_PKL_ENTRADA)
    
    if mi_grafo:
        try:
            # Selecciona un nodo de inicio al azar de todos los nodos del grafo.
            SOURCE_NODE = random.randrange(mi_grafo.vcount())
            logging.info(f"Nodo de inicio de la travesía DFS seleccionado al azar: {SOURCE_NODE}")
        except ValueError:
            logging.error("El grafo está vacío y no se puede seleccionar un nodo de inicio.")
            exit()

        # Ejecuta el algoritmo DFS.
        nodos_visitados_dfs, tiempo_dfs = dfs_algoritmo(mi_grafo, SOURCE_NODE)
        
        if nodos_visitados_dfs:
            # Crea el mapa con los resultados.
            mapa = crear_mapa_multicapa_dfs(mi_grafo, SOURCE_NODE, nodos_visitados_dfs)
            
            if mapa:
                # Guarda el mapa en un archivo HTML.
                mapa.save(MAPA_HTML_SALIDA)
                
                # Imprime un resumen en la consola.
                print("\n" + "="*55)
                # ... (código de impresión de resumen)
```

### ➤ Archivo: `4-dijkstra.py`

**Propósito**: Encontrar el camino más corto entre los dos nodos geográficamente más distantes del grafo. Utiliza la distancia Haversine como peso para las aristas y el algoritmo de Dijkstra para el cálculo. El resultado se visualiza en un mapa.

```python
# Importa las librerías necesarias.
import pickle
import time
import logging
import random
from math import radians, sin, cos, sqrt, atan2 # Funciones matemáticas para la distancia Haversine.
from itertools import combinations          # Para generar combinaciones de nodos.
import igraph as ig
from typing import Dict, Optional, Tuple, List, Any
import folium
from folium.plugins import BeautifyIcon
from tqdm import tqdm # Para barras de progreso.

# Configuración del logging.
# ...

# Función para cargar el grafo.
def cargar_grafo(grafo_path: str) -> Optional[ig.Graph]:
    # ... (idéntica a la de los otros scripts)

# Función para calcular la distancia Haversine entre dos puntos (lat, lon).
def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calcula la distancia en kilómetros entre dos puntos geográficos."""
    R = 6371  # Radio de la Tierra en km.
    # Convierte grados a radianes.
    dlat, dlon = radians(lat2 - lat1), radians(lon2 - lon1)
    # Fórmula Haversine.
    a = sin(dlat / 2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2)**2
    return R * 2 * atan2(sqrt(a), sqrt(1 - a))

# Función para añadir pesos a las aristas del grafo.
def preparar_pesos_geograficos(graph: ig.Graph, force_recalc: bool = False) -> None:
    """Asegura que cada arista tenga un atributo 'weight' con su distancia haversine."""
    # Si las aristas ya tienen peso y no forzamos el recálculo, no hace nada.
    if 'weight' in graph.es.attributes() and not force_recalc:
        logging.info("El atributo 'weight' ya existe en las aristas. No se recalculará.")
        return

    logging.info("Calculando pesos geográficos (distancia Haversine) para cada arista...")
    pesos = []
    # Itera sobre cada arista del grafo.
    for edge in graph.es:
        source_v = graph.vs[edge.source] # Vértice de origen de la arista.
        target_v = graph.vs[edge.target] # Vértice de destino.
        # Calcula la distancia entre los dos vértices.
        dist = haversine_distance(source_v['lat'], source_v['lon'], target_v['lat'], target_v['lon'])
        pesos.append(dist)
    # Asigna la lista de pesos calculados al atributo 'weight' de todas las aristas a la vez.
    graph.es['weight'] = pesos
    logging.info("Pesos geográficos calculados y asignados a las aristas.")

# Función para encontrar una aproximación de los dos nodos más distantes.
def encontrar_nodos_mas_distantes_aprox(graph: ig.Graph) -> Optional[Tuple[int, int]]:
    """Encuentra un par aproximado de los nodos más distantes usando la caja delimitadora."""
    logging.info("Buscando el par de nodos geográficamente más distantes (aproximación)...")
    # Diccionario para almacenar los nodos en los extremos geográficos (min/max lat/lon).
    extremos: Dict[str, Tuple[float, Any]] = {
        'min_lat': (float('inf'), None), 'max_lat': (float('-inf'), None),
        'min_lon': (float('inf'), None), 'max_lon': (float('-inf'), None)
    }
    
    # Itera sobre todos los vértices para encontrar los 4 puntos extremos.
    for v in graph.vs:
        # ... (código para encontrar y guardar los nodos con lat/lon min/max)
    
    # Crea un conjunto con los IDs de los nodos extremos encontrados.
    nodos_candidatos_ids = {node_id for _, node_id in extremos.values() if node_id is not None}
    if len(nodos_candidatos_ids) < 2:
        logging.error("No se encontraron suficientes nodos con coordenadas para determinar un rango.")
        return None

    # Compara las distancias entre todos los pares de nodos candidatos.
    max_dist, par_mas_distante = -1.0, None
    # 'combinations' genera todos los pares únicos de los nodos candidatos.
    for u_id, v_id in combinations(nodos_candidatos_ids, 2):
        dist = haversine_distance(graph.vs[u_id]['lat'], graph.vs[u_id]['lon'], graph.vs[v_id]['lat'], graph.vs[v_id]['lon'])
        if dist > max_dist:
            max_dist, par_mas_distante = dist, (u_id, v_id)
            
    if par_mas_distante:
        logging.info(f"Par más distante encontrado: Nodos {par_mas_distante} (Distancia: {max_dist:.2f} km)")
    return par_mas_distante

# Función para ejecutar el algoritmo de Dijkstra.
def encontrar_camino_mas_corto_dijkstra(graph: ig.Graph, source: int, sink: int) -> Tuple[Optional[List[int]], Optional[float], float]:
    logging.info(f"Iniciando algoritmo de Dijkstra de nodo {source} a {sink}.")
    start_time = time.time()
    
    try:
        # La función 'shortest_paths' de igraph es muy rápida. Devuelve el coste (distancia total).
        # Se especifica 'weights='weight'' para que use nuestro atributo de distancia Haversine.
        coste_total = graph.shortest_paths(source=source, target=sink, weights='weight')[0][0]
        # 'get_shortest_paths' devuelve la secuencia de nodos que forman el camino.
        camino_nodos = graph.get_shortest_paths(v=source, to=sink, weights='weight', output='vpath')[0]
        
        tiempo_total = time.time() - start_time
        logging.info(f"Dijkstra completado en {tiempo_total:.4f} segundos.")
        
        if not camino_nodos: # Si no hay camino, igraph devuelve una lista vacía.
            return None, None, tiempo_total
            
        return camino_nodos, coste_total, tiempo_total
    except Exception as e:
        # ... (manejo de errores)

# --- Funciones de Visualización ---
# Crea el mapa base con el camino más corto encontrado.
def crear_mapa_camino_corto(graph: ig.Graph, camino: List[int], coste: float, source: int, sink: int) -> Optional[folium.Map]:
    # ... (código para crear el mapa, marcador de inicio y fin)
    
    # Crea una capa para el camino más corto.
    path_group = folium.FeatureGroup(name='Camino Más Corto (Dijkstra)', show=True)
    puntos = [(graph.vs[nid]['lat'], graph.vs[nid]['lon']) for nid in camino if graph.vs[nid]['lat'] is not None]
    
    # Dibuja una línea azul gruesa para representar el camino.
    folium.PolyLine(
        puntos, color='#1f77b4', weight=5, opacity=0.9, tooltip=f"Camino más corto: {coste:.2f} km"
    ).add_to(path_group)
    
    path_group.add_to(m)
    return m

# Función para añadir caminos aleatorios al mapa para dar contexto.
def agregar_caminos_random_al_mapa(m: folium.Map, graph: ig.Graph, num_caminos: int):
    logging.info(f"Generando {num_caminos} caminos aleatorios para dar contexto a la red...")
    # ... (código para seleccionar pares de nodos aleatorios y calcular sus caminos más cortos)
    
    # Crea una capa para los caminos aleatorios, inicialmente oculta ('show=False').
    random_group = folium.FeatureGroup(name=f'{num_caminos} Caminos Aleatorios (Contexto)', show=False)

    # Itera sobre los caminos generados.
    for camino_info in caminos_generados:
        puntos = [(graph.vs[nid]['lat'], graph.vs[nid]['lon']) for nid in camino_info['path']]
        # Resalta el más corto (rojo) y el más largo (verde) de la muestra.
        if camino_info == camino_mas_corto_muestra:
            color, weight, opacity, tooltip = '#d62728', 4, 1.0, f"Más corto de la muestra ({camino_info['length_km']:.2f} km)"
        elif camino_info == camino_mas_largo_muestra:
            color, weight, opacity, tooltip = '#2ca02c', 4, 1.0, f"Más largo de la muestra ({camino_info['length_km']:.2f} km)"
        else: # Los demás caminos se dibujan en gris claro.
            color, weight, opacity, tooltip = '#555555', 1, 0.5, f"Camino de {camino_info['length_km']:.2f} km"
        
        folium.PolyLine(puntos, color=color, weight=weight, opacity=opacity, tooltip=tooltip).add_to(random_group)
        
    random_group.add_to(m)

# --- Bloque Principal ---
if __name__ == "__main__":
    # ... (configuración y carga del grafo)
    if mi_grafo:
        # 1. Añade los pesos a las aristas.
        preparar_pesos_geograficos(mi_grafo)

        # 2. Encuentra los nodos de inicio y fin.
        par_distante = encontrar_nodos_mas_distantes_aprox(mi_grafo)

        if par_distante:
            SOURCE_NODE, SINK_NODE = par_distante
            
            # 3. Ejecuta Dijkstra.
            camino_optimo, coste_total, tiempo_dijkstra = encontrar_camino_mas_corto_dijkstra(mi_grafo, SOURCE_NODE, SINK_NODE)
            
            if camino_optimo and coste_total is not None:
                # 4. Crea el mapa base.
                mapa_resultado = crear_mapa_camino_corto(mi_grafo, camino_optimo, coste_total, SOURCE_NODE, SINK_NODE)

                if mapa_resultado:
                    # 5. Añade la capa de contexto.
                    agregar_caminos_random_al_mapa(mapa_resultado, mi_grafo, NUM_CAMINOS_RANDOM)
                    
                    folium.LayerControl().add_to(mapa_resultado) # Añade el control de capas.
                    mapa_resultado.save(MAPA_HTML_SALIDA) # Guarda el mapa.
                    
                    # 7. Imprime el resumen.
                    # ...
```

### ➤ Archivo: `5-community.py`

**Propósito**: Implementar el Algoritmo de Propagación de Etiquetas (LPA) para detectar comunidades en el grafo. Visualiza las comunidades más relevantes (la más grande, la más pequeña y una muestra aleatoria) en un mapa interactivo, utilizando optimizaciones para manejar un gran número de nodos.

```python
# Importa las librerías necesarias.
import pickle, time, logging, random
from collections import Counter # Estructura de datos para contar elementos (muy útil para LPA).
import igraph as ig
from typing import Optional, List, Dict
import folium
from folium.plugins import MarkerCluster # Para agrupar marcadores cercanos en el mapa.
import matplotlib # Para acceder a mapas de colores.
import matplotlib.colors as colors # Para normalizar y convertir colores.
from tqdm import tqdm

# --- Configuración del Logging ---
# ...

# --- UMBRALES DE VISUALIZACIÓN (Optimizaciones de rendimiento) ---
# Si una comunidad es más grande que esto, no se dibujan sus aristas internas.
UMBRAL_MAX_VISUALIZACION_ARISTAS = 2000
# Si una comunidad es más grande que esto, solo se dibuja una muestra de sus nodos.
UMBRAL_MAX_NODOS_A_DIBUJAR = 5000

# --- Función de Carga de Grafo ---
def cargar_grafo(grafo_path: str) -> Optional[ig.Graph]:
    # ... (idéntica a la de otros scripts)

# --- Implementación del LPA ---
def detectar_comunidades_lpa(graph: ig.Graph, max_iter: int = 10) -> Dict[int, List[int]]:
    logging.info(f"Iniciando detección de comunidades con algoritmo propio (LPA) con {max_iter} iteraciones...")
    start_time = time.time()
    
    # Inicialización: cada nodo tiene su propio índice como etiqueta.
    labels = {v.index: v.index for v in graph.vs}
    
    # Bucle principal de iteraciones de LPA.
    for i in range(max_iter):
        logging.info(f"LPA - Iteración {i + 1}/{max_iter}...");
        changes_count = 0 # Contador para ver si el algoritmo converge.
        
        # Se procesan los nodos en un orden aleatorio en cada iteración para evitar sesgos.
        nodes_to_process = list(range(graph.vcount())); random.shuffle(nodes_to_process)
        iterator = tqdm(nodes_to_process, desc=f"Iteración {i+1}")
        
        # Bucle de actualización de etiquetas.
        for node_id in iterator:
            neighbors = graph.neighbors(node_id, mode='all') # Obtiene todos los vecinos.
            if not neighbors: continue # Si un nodo está aislado, no hace nada.
            
            # Cuenta la frecuencia de cada etiqueta entre los vecinos.
            label_counts = Counter(labels[n] for n in neighbors)
            max_freq = max(label_counts.values()) # Encuentra la frecuencia más alta.
            
            # Obtiene todas las etiquetas que tienen la frecuencia más alta.
            most_frequent_labels = [label for label, count in label_counts.items() if count == max_freq]
            # Elige una de las etiquetas más frecuentes al azar para romper empates.
            new_label = random.choice(most_frequent_labels)
            
            # Si la etiqueta del nodo cambia, se actualiza y se cuenta el cambio.
            if labels[node_id] != new_label:
                labels[node_id] = new_label
                changes_count += 1
                
        logging.info(f"Fin de la iteración {i + 1}. Hubo {changes_count} cambios de etiqueta.")
        # Condición de parada: si no hubo cambios, el algoritmo ha convergido.
        if changes_count == 0:
            logging.info("Convergencia alcanzada antes del máximo de iteraciones."); break
            
    # Advertencia si no se alcanzó la convergencia.
    if i == max_iter - 1 and changes_count > 0:
        logging.warning("Se alcanzó el máximo de iteraciones sin convergencia completa.")

    # Agrupa los nodos por su etiqueta final para formar las comunidades.
    comunidades = {};
    # ... (código para construir el diccionario de comunidades)
        
    end_time = time.time()
    logging.info(f"LPA completado en {end_time - start_time:.2f} s. Se encontraron {len(comunidades)} comunidades.")
    return comunidades

# --- Función de Análisis ---
# Selecciona qué comunidades visualizar para no sobrecargar el mapa.
def analizar_y_seleccionar_comunidades(comunidades_dict: Dict[int, List[int]], num_random: int = 20) -> Dict[str, List[int]]:
    logging.info("Analizando y seleccionando comunidades para visualización...");
    # ... (código para encontrar la comunidad más grande, la más pequeña y una muestra aleatoria)
    # La muestra aleatoria se toma de comunidades de tamaño "manejable".
    posibles_ids_random = [cid for cid, size in comunidades_con_tamaño if cid not in ids_extremos and 5 < size < UMBRAL_MAX_VISUALIZACION_ARISTAS]
    ids_random = random.sample(posibles_ids_random, min(num_random, len(posibles_ids_random))) if posibles_ids_random else []
    # Devuelve un diccionario con las listas de IDs de comunidades seleccionadas.
    seleccion = {'grande': [id_grande], 'pequena': [id_pequeña], 'random': ids_random}
    return seleccion

# --- Función de Colores ---
# Crea una función que mapea el tamaño de una comunidad a un color.
def crear_mapa_de_colores(tamaño_min: int, tamaño_max: int):
    # Usa un mapa de colores predefinido de matplotlib ('coolwarm': azul a rojo).
    colormap = matplotlib.colormaps.get_cmap('coolwarm')
    # Usa una escala logarítmica porque los tamaños de comunidad pueden variar mucho.
    normalizador = colors.LogNorm(vmin=max(1, tamaño_min), vmax=tamaño_max)
    # Devuelve una función 'lambda' que toma un tamaño y devuelve un color hexadecimal.
    return lambda tamaño: colors.to_hex(colormap(normalizador(tamaño)))

# --- Función de Visualización ---
def visualizar_comunidades(graph: ig.Graph, comunidades_dict: Dict[int, List[int]], seleccion: Dict[str, List[int]], output_filename: str):
    # ... (código para crear el mapa base)
    
    # Crea la función de mapeo de color.
    mapa_color = crear_mapa_de_colores(min(tamaños.values()), max(tamaños.values()))
    ids_a_visualizar = seleccion['grande'] + seleccion['pequena'] + seleccion['random']
    iterator = tqdm(ids_a_visualizar, desc="Creando capas de comunidades")
    
    for com_id in iterator:
        # ... (código para obtener miembros, tamaño, color y nombre de la capa)
        
        # OPTIMIZACIÓN: Usa MarkerCluster para comunidades grandes. Agrupa los marcadores cercanos.
        container = MarkerCluster(name=nombre_capa, show=show_layer) if tamaño_original > 200 else folium.FeatureGroup(name=nombre_capa, show=show_layer)
        container.add_to(m)

        # OPTIMIZACIÓN: Muestreo de nodos para comunidades gigantes.
        nodos_a_dibujar = miembros_originales
        if tamaño_original > UMBRAL_MAX_NODOS_A_DIBUJAR:
            logging.warning(f"Comunidad {com_id} ({tamaño_original} nodos) excede umbral. Mostrando muestra de {UMBRAL_MAX_NODOS_A_DIBUJAR}.")
            nodos_a_dibujar = random.sample(miembros_originales, UMBRAL_MAX_NODOS_A_DIBUJAR)

        # Dibuja los marcadores de los nodos.
        for nodo_id in nodos_a_dibujar:
            # ... (código para dibujar CircleMarker)
        
        # OPTIMIZACIÓN: Dibujo de aristas solo para comunidades pequeñas.
        if tamaño_original <= UMBRAL_MAX_VISUALIZACION_ARISTAS:
            if len(nodos_visibles_con_coords) > 1:
                # Crea un subgrafo solo con los nodos de esta comunidad.
                subgrafo_comunidad = graph.subgraph(nodos_visibles_con_coords)
                # Dibuja las aristas internas de este subgrafo.
                for arista in subgrafo_comunidad.es:
                    # ... (código para dibujar PolyLine para cada arista)
        else:
            logging.warning(f"Omitiendo dibujo de aristas para comunidad {com_id} (tamaño: {tamaño_original} > {UMBRAL_MAX_VISUALIZACION_ARISTAS}).")

    folium.LayerControl(collapsed=False).add_to(m)
    logging.info("Guardando el mapa en el archivo HTML. Este paso puede tardar...")
    m.save(output_filename)
    logging.info(f"Mapa guardado correctamente en '{output_filename}'.")

# --- Bloque Principal ---
if __name__ == "__main__":
    # ... (configuración y carga del grafo)
    if mi_grafo:
        # 1. Detecta las comunidades con LPA.
        coms_dict = detectar_comunidades_lpa(mi_grafo, max_iter=MAX_ITER_LPA)
        if coms_dict:
            # 2. Selecciona las comunidades a visualizar.
            comunidades_seleccionadas = analizar_y_seleccionar_comunidades(coms_dict, NUM_COMUNIDADES_RANDOM)
            if comunidades_seleccionadas:
                # 3. Crea y guarda el mapa.
                visualizar_comunidades(mi_grafo, coms_dict, comunidades_seleccionadas, MAPA_HTML_SALIDA)
                
                # 4. Imprime el resumen final.
                # ...
```

## Referencias del Código de Generación y Carga de Grafos Paralelizada (Formato APA)

A continuación, se presentan las referencias de las librerías y conceptos clave utilizados en el código, formateadas según el estilo de la American Psychological Association (APA):

### Librerías de Python

1.  **igraph.** (s. f.). *igraph documentation*. Recuperado de [https://igraph.org/python/doc/](https://igraph.org/python/doc/)

2.  **The Python Standard Library.** (s. f.). *time --- Time access and conversions*. Recuperado de [https://docs.python.org/3/library/time.html](https://docs.python.org/3/library/time.html)

3.  **The Python Standard Library.** (s. f.). *logging --- Logging facility for Python*. Recuperado de [https://docs.python.org/3/library/logging.html](https://docs.python.org/3/library/logging.html)

4.  **The Python Standard Library.** (s. f.). *fileinput --- Facilitate looping over lines of standard input or a list of files*. Recuperado de [https://docs.python.org/3/library/fileinput.html](https://docs.python.org/3/library/fileinput.html)

5.  **The Python Standard Library.** (s. f.). *pickle --- Python object serialization*. Recuperado de [https://docs.python.org/3/library/pickle.html](https://docs.python.org/3/library/pickle.html)

6.  **The Python Standard Library.** (s. f.). *gc --- Garbage Collector interface*. Recuperado de [https://docs.python.org/3/library/gc.html](https://docs.python.org/3/library/gc.html)

7.  **The Python Standard Library.** (s. f.). *multiprocessing --- Process-based parallelism*. Recuperado de [https://docs.python.org/3/library/multiprocessing.html](https://docs.python.org/3/library/multiprocessing.html)

8.  **The Python Standard Library.** (s. f.). *itertools --- Functions creating iterators for efficient looping*. Recuperado de [https://docs.python.org/3/library/itertools.html](https://docs.python.org/3/library/itertools.html)

## Recursos Adicionales (Tutoriales y Conceptos)

Si bien las documentaciones oficiales son las referencias primarias, los siguientes tipos de recursos pueden complementar la comprensión:

* **Tutoriales de Python sobre:** Realiza búsquedas en línea en sitios como:
    * Real Python ([https://realpython.com/](https://realpython.com/))
    * Python.org Tutorials ([https://docs.python.org/3/tutorial/](https://docs.python.org/3/tutorial/))
    * GeeksforGeeks ([https://www.geeksforgeeks.org/python/](https://www.geeksforgeeks.org/python/))
    * Stack Overflow ([https://stackoverflow.com/](https://stackoverflow.com/)) 

* **Libros sobre Python:** Busca libros como "Automate the Boring Stuff with Python" de Al Sweigart, "Python Crash Course" de Eric Matthes, o "Fluent Python" de Luciano Ramalho para una comprensión más profunda.

* **Cursos en línea:** Explora plataformas como:
    * Coursera ([https://www.coursera.org/](https://www.coursera.org/))
    * edX ([https://www.edx.org/](https://www.edx.org/))
    * Udemy ([https://www.udemy.com/](https://www.udemy.com/))
    * DataCamp ([https://www.datacamp.com/](https://www.datacamp.com/))

* **Comunidades en línea:**
    * Reddit (subreddits como r/learnpython ([https://www.reddit.com/r/learnpython/](https://www.reddit.com/r/learnpython/)), r/python ([https://www.reddit.com/r/python/](https://www.reddit.com/r/python/)))

**Apunte:** Las fechas de recuperación ("Recuperado de") se indican como "s. f." (sin fecha) ya que la documentación de las librerías suele estar en constante actualización y no siempre tiene una fecha de publicación específica para la totalidad del sitio web. Si estuvieras citando una sección específica con una fecha, deberías incluirla. Los enlaces a los recursos adicionales son generales para facilitar la búsqueda de tutoriales y cursos relevantes.

## Concluyendo

* El parámetro `num_procesos` en la función `procesar_usuarios_paralelizado` se establece por defecto al número de núcleos de la CPU, lo que generalmente proporciona un buen rendimiento. Puede ajustar este valor según las capacidades de su sistema.
* Se realiza una limpieza de memoria explícita (`del` y `gc.collect()`) después de la creación del grafo para liberar recursos, especialmente importante cuando se trabaja con grandes conjuntos de datos.
* El código incluye validaciones básicas para asegurar que los IDs de los usuarios en el archivo de conexiones estén dentro del rango válido de nodos.
* El acceso a los atributos de los vértices en `igraph` se realiza mediante la indexación del objeto `VertexSeq` (e.g., `grafo_cargado.vs[0]['lat']`).

