# ADA
# Explicación Detallada del Código

Este script de Python está diseñado para construir un grafo a partir de datos de ubicaciones y usuarios, y luego realizar un análisis exploratorio de datos (EDA) básico. Utiliza las bibliotecas `igraph`, `time`, `logging`, `polars`, `pickle`, y `numpy`.
Drive (contiene el grafo.pkl):
https://drive.google.com/drive/folders/1gLylv7wahcpLzcefsCbAXcYFj1ut7yfo?usp=sharing 

# Generación y Carga de Grafos Paralelizada con Atributos de Ubicación

Este script de Python utiliza la biblioteca `igraph` para crear y manipular grafos dirigidos. El proceso se optimiza mediante el uso de la biblioteca `multiprocessing` para la lectura y procesamiento paralelo de grandes archivos de datos. Además, se implementa un sistema de logging para el seguimiento detallado de la ejecución.

## Funcionalidades Principales

1.  **Carga Directa de Ubicaciones:**
    * La función `cargar_ubicaciones_directo` lee un archivo de texto donde cada línea contiene una latitud y una longitud separadas por comas.
    * Extrae las coordenadas y las almacena en listas separadas.
    * Retorna una lista de tuplas con las ubicaciones (latitud, longitud), el número total de ubicaciones y un diccionario con las latitudes y longitudes por separado.

    ```python
    def cargar_ubicaciones_directo(ubicaciones_path):
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
    ```

2.  **Procesamiento Paralelizado de Conexiones de Usuarios:**
    * La función `procesar_linea_usuarios` toma una línea del archivo de usuarios y el número total de nodos.
    * Cada línea del archivo de usuarios se espera que contenga una lista de IDs de usuario (potenciales destinos de una conexión) separados por comas.
    * Identifica los IDs válidos dentro del rango del número de nodos y genera una lista de aristas locales (pares de nodos origen-destino).
    * La función `procesar_usuarios_paralelizado` utiliza un pool de procesos (`multiprocessing.Pool`) para aplicar `procesar_linea_usuarios` a cada línea del archivo de usuarios en paralelo, acelerando significativamente el procesamiento de grandes archivos.
    * Combina los resultados de todos los procesos para obtener la lista final de aristas del grafo.

    ```python
    def procesar_linea_usuarios(linea_idx_contenido, num_nodos):
        idx, linea = linea_idx_contenido
        aristas_locales = []
        conexiones = set()
        for x in linea.strip().split(','):
            x_strip = x.strip()
            if x_strip.isdigit():
                dst = int(x_strip)
                if 1 <= dst <= num_nodos:
                    conexiones.add(dst - 1)
        aristas_locales.extend((idx - 1, dst) for dst in conexiones)
        return aristas_locales

    def procesar_usuarios_paralelizado(usuarios_path, num_nodos, num_procesos=mp.cpu_count()):
        logging.info(f"Procesando usuarios en paralelo con {num_procesos} procesos...")
        start_time_procesamiento = time.time()
        with fileinput.input(usuarios_path) as f:
            lineas_enumeradas = ((idx, linea) for idx, linea in enumerate(f, start=1) if idx <= num_nodos)
            with mp.Pool(processes=num_procesos) as pool:
                resultados = pool.starmap(procesar_linea_usuarios, [(item, num_nodos) for item in lineas_enumeradas])
        aristas = list(chain.from_iterable(resultados))
        aristas_filtradas = [(src, dst) for src, dst in aristas if 0 <= src < num_nodos and 0 <= dst < num_nodos]
        logging.info(f"Se procesaron {len(aristas_filtradas)} aristas en {time.time() - start_time_procesamiento:.2f} s.")
        return aristas_filtradas
    ```

3.  **Creación y Guardado del Grafo de `igraph` con Atributos:**
    * La función `crear_grafo_igraph_paralelizado` coordina la carga de ubicaciones y el procesamiento de usuarios.
    * Crea un grafo dirigido de `igraph` con el número de nodos correspondiente a las ubicaciones cargadas.
    * Añade las aristas procesadas al grafo.
    * Incorpora las latitudes y longitudes como atributos de los vértices del grafo, lo que permite asociar información geográfica a cada nodo.
    * Finalmente, guarda el grafo completo (estructura y atributos) en un archivo binario utilizando la biblioteca `pickle` para su posterior carga.

    ```python
    def crear_grafo_igraph_paralelizado(ubicaciones_path, usuarios_path, output_grafo_path):
        start_time = time.time()

        ubicaciones, num_nodos, atributos_ubicacion = cargar_ubicaciones_directo(ubicaciones_path)
        aristas = procesar_usuarios_paralelizado(usuarios_path, num_nodos)

        logging.info("Creando grafo de igraph...")
        g = ig.Graph(directed=True)
        g.add_vertices(num_nodos)
        g.add_edges(aristas)

        # Añadir atributos de ubicación al grafo
        for key, values in atributos_ubicacion.items():
            g.vs[key] = values

        logging.info(f"Guardando grafo con atributos en '{output_grafo_path}'...")
        with open(output_grafo_path, 'wb') as f:
            pickle.dump(g, f, protocol=pickle.HIGHEST_PROTOCOL)

        logging.info(f"Grafo guardado. Tiempo total: {time.time() - start_time:.2f} s")
        del ubicaciones, aristas, atributos_ubicacion
        gc.collect()
        return g
    ```

4.  **Carga del Grafo con Atributos:**
    * La función `cargar_grafo_con_atributos` se encarga de leer el archivo `.pkl` generado y cargar el grafo de `igraph` junto con sus atributos de los vértices.
    * Incluye manejo de errores en caso de que el archivo no se encuentre o si ocurre algún problema durante la carga.

    ```python
    def cargar_grafo_con_atributos(grafo_path):
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
    ```

5.  **Logging Detallado:**
    * Se utiliza la biblioteca `logging` para registrar información relevante sobre el proceso de ejecución, incluyendo tiempos de inicio y fin de las diferentes etapas, número de ubicaciones y aristas procesadas, y posibles advertencias o errores.
    * Los logs se escriben tanto a un archivo (`grafo_parkingson_paralelizado.log`) como a la consola.

    ```python
    import logging

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("grafo_parkingson_paralelizado.log", mode='w', encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    ```

## Uso

Para utilizar este script, asegúrese de tener instaladas las bibliotecas necesarias: `igraph`, `pickle` y `multiprocessing` (esta última suele estar incluida en la instalación estándar de Python).

1.  **Prepare los archivos de entrada:**
    * Cree un archivo de texto (por ejemplo, `10_million_location.txt`) donde cada línea contenga la longitud y la latitud de una ubicación separadas por una coma. Ejemplo: `-77.0369,38.8951`.
    * Cree un archivo de texto (por ejemplo, `10_million_user.txt`) donde cada línea represente un usuario (correspondiente a la misma fila en el archivo de ubicaciones). Cada línea debe contener una lista de IDs de otros usuarios (basados en el índice de fila, comenzando desde 1) a los que este usuario está conectado, separados por comas. Ejemplo: `2,5,10,15`.

2.  **Ejecute el script:**
    ```bash
    python your_script_name.py
    ```
    Reemplace `your_script_name.py` con el nombre del archivo donde guardó el código.

3.  **Examine la salida:**
    * Se generará un archivo llamado `grafo_igraph_paralelizado.pkl` que contendrá el grafo de `igraph` con los atributos de latitud y longitud de los nodos.
    * Se creará un archivo de log llamado `grafo_parkingson_paralelizado.log` con detalles de la ejecución.
    * En la consola, se mostrará información sobre la carga y el número de nodos y aristas del grafo cargado, así como los atributos de latitud y longitud del primer nodo (si el grafo no está vacío).

## Notas Adicionales

* El parámetro `num_procesos` en la función `procesar_usuarios_paralelizado` se establece por defecto al número de núcleos de la CPU, lo que generalmente proporciona un buen rendimiento. Puede ajustar este valor según las capacidades de su sistema.
* Se realiza una limpieza de memoria explícita (`del` y `gc.collect()`) después de la creación del grafo para liberar recursos, especialmente importante cuando se trabaja con grandes conjuntos de datos.
* El código incluye validaciones básicas para asegurar que los IDs de los usuarios en el archivo de conexiones estén dentro del rango válido de nodos.
* El acceso a los atributos de los vértices en `igraph` se realiza mediante la indexación del objeto `VertexSeq` (e.g., `grafo_cargado.vs[0]['lat']`).

