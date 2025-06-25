# ADA
# Explicación Detallada del Código

Este script de Python está diseñado para construir un grafo a partir de datos de ubicaciones y usuarios, y luego realizar un análisis exploratorio de datos (EDA) básico. Utiliza las bibliotecas `igraph`, `time`, `logging`, `polars`, `pickle`, y `numpy`.
Drive (contiene el grafo.pkl):
https://drive.google.com/drive/folders/1gLylv7wahcpLzcefsCbAXcYFj1ut7yfo?usp=sharing 

También, se pueden ver los resultados a partir de:
```bash
/vista_grafo/1-index.html
```
---

# Análisis de Redes a Gran Escala

Este proyecto es una inmersión profunda en el mundo del análisis de grafos, cubriendo el ciclo de vida completo: desde la generación de datos masivos y la construcción eficiente de la red, hasta la aplicación y visualización de algoritmos fundamentales. Utilizando Python y librerías de alto rendimiento como `igraph`, `multiprocessing` y `folium`, demostramos cómo abordar problemas de redes con millones de nodos y aristas.

## 🌟 Características Clave

*   **Escalabilidad:** Diseñado para manejar grafos con millones de nodos, utilizando técnicas de paralelización para la construcción y serialización eficiente para el almacenamiento y la carga.
*   **Generación de Datos Sintéticos:** Incluye un script para crear conjuntos de datos a gran escala, permitiendo pruebas y benchmarks reproducibles.
*   **Algoritmos Fundamentales Implementados:**
    *   **Búsqueda en Amplitud (BFS):** Explora la red nivel por nivel, ideal para encontrar los vecinos más cercanos o el camino más corto en términos de saltos.
    *   **Búsqueda en Profundidad (DFS):** Explora la red siguiendo un camino hasta su final antes de retroceder, útil para topologías y detección de ciclos.
    *   **Algoritmo de Dijkstra:** Implementado manualmente para encontrar el camino geográficamente más corto entre dos puntos, utilizando distancias reales como pesos de las aristas.
    *   **Algoritmo de Louvain:** Implementado desde cero para la detección de comunidades, un algoritmo estándar en la industria para el análisis de clústeres en redes sociales, biológicas y más.
*   **Visualización Interactiva y Significativa:** Cada análisis produce un mapa HTML interactivo que no solo muestra los resultados, sino que también utiliza técnicas de visualización (como clustering de marcadores, capas, colores y animaciones) para hacerlos comprensibles y explorables.

---

## 🚀 Guía de Inicio Rápido: Recreando el Proyecto

Sigue estos pasos para configurar tu entorno, generar los datos y ejecutar los análisis.

### 1. Requisitos Previos

*   Python 3.8 o superior.
*   Git para clonar el repositorio.
*   Suficiente memoria RAM (se recomiendan **16 GB o más** para trabajar con 10 millones de nodos).

### 2. Instalación

Primero, clona el repositorio y navega hasta el directorio del proyecto.

```bash
git clone <tu-repositorio-url>
cd <nombre-del-directorio>
```

Instala todas las librerías necesarias desde el archivo `requirements.txt` que deberías crear con el siguiente contenido:

**`requirements.txt`:**
```txt
collections
folium
folium.plugins
heapq
igraph
itertools
logging
matplotlib
pickle
polars
typing
tqdm

```

**Comando de instalación:**
```bash
pip install -r requirements.txt
```

### 3. Flujo de Ejecución

Los scripts están numerados para indicar el orden de ejecución.

#### Paso 1: Generar los Datos
Ejecuta el script `0-createdata.py` para generar los archivos de texto con ubicaciones y conexiones. Por defecto, crea datos para 1 millón de nodos. Puedes modificar los números dentro del script si lo deseas.

```bash
python 0-createdata.py
```
*   **Salida:** `1_million_location.txt` y `1_million_user.txt`.

#### Paso 2: Construir el Grafo
Este es el paso más intensivo en recursos. El script `1-grapy_paralel.py` leerá los archivos de texto y construirá el grafo en paralelo.

```bash
python 1-grapy_paralel.py
```
*   **Salida:** Un archivo binario `grafo_igraph_paralelizado.pkl` y un archivo de log `grafo_paralelizado.log`.
*   **Nota:** Asegúrate de que la variable `NUM_NODOS` dentro del script coincida con los datos que generaste.

#### Paso 3: Verificar el Grafo (Opcional pero Recomendado)
Este script de prueba rápida cargará el grafo y mostrará estadísticas básicas para confirmar que la construcción fue exitosa.

```bash
python 2-prueba_graphote.py
```
*   **Salida:** Información en la consola, un log `grafo_pruebas.log` y una tabla de muestra `tabla_grafos_prueba.csv`.

#### Paso 4: Ejecutar los Análisis
Ahora puedes ejecutar cualquiera de los scripts de análisis. Cada uno cargará el mismo archivo `grafo_igraph_paralelizado.pkl` y producirá un mapa HTML interactivo y su propio archivo de log.

```bash
# Ejecutar análisis DFS
python 3-dfs.py

# Ejecutar análisis Dijkstra
python 4-dijkstra.py

# Ejecutar análisis de comunidades (Louvain)
python 5-communityLouvain.py

# Ejecutar análisis BFS
python 6-bfs.py
```
*   **Salida:** Los archivos `mapa_dfs_...html`, `analisis_de_red_dijkstra.html`, `analisis_comunidades_louvain_manual.html` y `bfs.html`, junto con sus respectivos logs. Abre estos archivos HTML en tu navegador para explorar los resultados.

---

## 📄 Explicación Profunda de los Scripts

Cada script cumple un rol específico y crucial en el pipeline de análisis.

### `0-createdata.py`: La Semilla del Proyecto

Este script es el punto de partida que genera los datos crudos. Sin datos, no hay análisis. Su diseño es simple pero efectivo: crea dos archivos de texto que representan los nodos y las aristas de nuestra red de una manera desacoplada y fácil de procesar.
-   El **archivo de ubicaciones** es un catálogo de todos los posibles nodos, cada uno con un atributo geográfico (latitud y longitud).
-   El **archivo de conexiones** define la estructura de la red, especificando qué nodos se conectan con qué otros. Este formato es común en la exportación de bases de datos de grafos.

### `1-grapy_paralel.py`: El Arquitecto del Grafo

Este script es el corazón de la fase de preparación. Su principal desafío es transformar los datos de texto plano en una estructura de grafo altamente optimizada en memoria de la manera más rápida posible.
-   **Paralelización con `multiprocessing`:** Su característica más importante es el uso del procesamiento en paralelo. Leer y procesar millones de líneas de texto de forma secuencial sería extremadamente lento. El script divide inteligentemente el trabajo de leer las conexiones entre todos los núcleos de la CPU, logrando una reducción drástica en el tiempo de construcción.
-   **Gestión de Memoria:** Lee los archivos de manera eficiente para no agotar la memoria del sistema y, al final, utiliza `pickle` para **serializar** el objeto de grafo de `igraph`. La serialización convierte el complejo objeto en memoria en un flujo de bytes que se puede guardar en un archivo. Esto permite que los scripts de análisis posteriores carguen el grafo completo en segundos, en lugar de tener que reconstruirlo desde cero cada vez.

### `2-prueba_graphote.py`: El Inspector de Calidad

Construir un grafo de millones de nodos es un proceso propenso a errores sutiles (índices incorrectos, atributos faltantes, etc.). Este script actúa como una suite de control de calidad. Carga el grafo serializado y realiza una serie de "pruebas de cordura" para validar su integridad. Confirma que el número de nodos y aristas es correcto, que los atributos están presentes y que la estructura es accesible. Genera también una tabla de resumen con `Polars`, una librería de DataFrames ultrarrápida, ofreciendo una vista previa tangible de los datos del grafo.

### Los Scripts de Análisis: La Inteligencia Aplicada

Estos cuatro scripts son donde la teoría de grafos cobra vida. Todos comparten el mismo primer paso: cargar el grafo `.pkl`. A partir de ahí, cada uno aplica un algoritmo distinto para revelar una faceta diferente de la red.

#### `3-dfs.py`: El Explorador de Profundidad

*   **¿Qué hace?** Implementa una Búsqueda en Profundidad (DFS), un algoritmo que explora una ruta hasta el final antes de retroceder. Es como resolver un laberinto poniendo una mano en la pared y siguiéndola.
*   **¿Para qué sirve?** Es útil para entender la topología de la red, encontrar caminos largos y detectar componentes conectados.
*   **Visualización Clave:** El mapa muestra el camino exacto del recorrido. Para evitar el desorden visual, el camino se divide en capas y se simplifica si es excesivamente largo, destacando el inicio, el fin y el nodo más profundo alcanzado.

#### `4-dijkstra.py`: El Navegador Óptimo

*   **¿Qué hace?** Implementa el algoritmo de Dijkstra para encontrar el camino más corto entre dos nodos. Crucialmente, no busca el camino con menos "saltos", sino el que tiene la menor suma de **pesos** en sus aristas.
*   **¿Para qué sirve?** En nuestro caso, los pesos son las distancias geográficas reales. Por lo tanto, este script encuentra la ruta *físicamente más corta* entre dos puntos, como lo haría un GPS.
*   **Visualización Clave:** El mapa resalta la ruta óptima en un color vibrante. Para dar contexto, añade una capa con caminos aleatorios, mostrando visualmente cuán eficiente es la ruta encontrada en comparación con otras rutas posibles en la red.

#### `5-communityLouvain.py`: El Sociólogo de la Red

*   **¿Qué hace?** Implementa el algoritmo de Louvain, un método de clustering jerárquico para la detección de comunidades. Su objetivo es agrupar nodos que están más densamente conectados entre sí que con el resto de la red.
*   **¿Para qué sirve?** Es fundamental en el análisis de redes sociales (para encontrar grupos de amigos), en biología (para identificar grupos de proteínas con funciones similares) o en cualquier red donde existan clústeres naturales.
*   **Visualización Clave:** El mapa utiliza colores para diferenciar las comunidades y `MarkerCluster` para agrupar nodos en áreas densas, permitiendo al usuario explorar la estructura de los clústeres de forma interactiva. El color de cada comunidad a menudo se mapea a su tamaño, ofreciendo una pista visual inmediata sobre los grupos más dominantes.

#### `6-bfs.py`: El Divulgador de Ondas

*   **¿Qué hace?** Implementa una Búsqueda en Amplitud (BFS), que explora la red en "ondas" o niveles: primero el nodo de inicio, luego todos sus vecinos directos (nivel 1), luego los vecinos de los vecinos (nivel 2), y así sucesivamente.
*   **¿Para qué sirve?** Es el algoritmo ideal para encontrar el camino más corto en términos de número de saltos. Se usa en redes sociales para encontrar "grados de separación" o en enrutamiento de redes para encontrar el menor número de saltos.
*   **Visualización Clave:** El mapa resultante es muy informativo. Muestra todo el **árbol de expansión** explorado por el BFS con líneas grises. Sobre este árbol, superpone una **línea animada** que traza el camino más largo (en saltos) encontrado, ilustrando de manera dinámica y clara la extensión máxima de la búsqueda desde el punto de partida.
                


1.  **El grafo tiene un "componente gigante" muy denso.** El análisis de comunidades (imagen original) muestra que casi 9 millones de los 10 millones de nodos pertenecen a una única comunidad masiva. Esto indica que la red está altamente interconectada, formando un núcleo central donde la mayoría de los puntos están conectados entre sí, directa o indirectamente.

2.  **Es posible encontrar rutas muy eficientes entre puntos lejanos.** El mapa de Dijkstra (`analisis_de_red_dijkstra.html`) demuestra que, a pesar del enorme tamaño de la red, el algoritmo encontró una ruta óptima de aproximadamente **19,600 km**. Este camino es notablemente más corto que muchas otras rutas aleatorias, lo que prueba la eficacia del algoritmo para optimizar conexiones en una red compleja.

3.  **Explorar la red sin un objetivo de distancia revela su complejidad.** El mapa de recorrido DFS (`mapa_dfs_multicapa.html`) muestra un camino largo y sinuoso que salta entre continentes. A diferencia de Dijkstra, el DFS explora la profundidad de las conexiones, revelando cómo los nodos están enlazados de formas no siempre intuitivas geográficamente y demostrando la complejidad estructural del grafo.

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



