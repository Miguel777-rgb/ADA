# analisis_completo_de_red.py
import pickle
import time
import logging
import random
from math import radians, sin, cos, sqrt, atan2
from itertools import combinations
import igraph as ig
from typing import Dict, Optional, Tuple, List, Any
import folium
from folium.plugins import BeautifyIcon
from tqdm import tqdm


# --- Configuración del Logging ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("analisis_de_red.log", mode='w', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

# --- Funciones de Carga y Cálculo ---

def cargar_grafo(grafo_path: str) -> Optional[ig.Graph]:
    """Carga un grafo igraph desde un archivo .pkl."""
    logging.info(f"Cargando el grafo desde '{grafo_path}'...")
    start_time = time.time()
    try:
        with open(grafo_path, 'rb') as f:
            g = pickle.load(f)
        end_time = time.time()
        logging.info(f"Grafo cargado en {end_time - start_time:.2f} segundos. {g.summary()}")
        return g
    except Exception as e:
        logging.error(f"Error crítico al cargar el grafo: {e}")
        return None

def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calcula la distancia en kilómetros entre dos puntos geográficos."""
    R = 6371  # Radio de la Tierra en km
    dlat, dlon = radians(lat2 - lat1), radians(lon2 - lon1)
    a = sin(dlat / 2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2)**2
    return R * 2 * atan2(sqrt(a), sqrt(1 - a))

def preparar_pesos_geograficos(graph: ig.Graph, force_recalc: bool = False) -> None:
    """Asegura que cada arista tenga un atributo 'weight' con su distancia haversine."""
    if 'weight' in graph.es.attributes() and not force_recalc:
        logging.info("El atributo 'weight' ya existe en las aristas. No se recalculará.")
        return

    logging.info("Calculando pesos geográficos (distancia Haversine) para cada arista...")
    pesos = []
    for edge in graph.es:
        source_v = graph.vs[edge.source]
        target_v = graph.vs[edge.target]
        dist = haversine_distance(source_v['lat'], source_v['lon'], target_v['lat'], target_v['lon'])
        pesos.append(dist)
    graph.es['weight'] = pesos
    logging.info("Pesos geográficos calculados y asignados a las aristas.")


def encontrar_nodos_mas_distantes_aprox(graph: ig.Graph) -> Optional[Tuple[int, int]]:
    """Encuentra un par aproximado de los nodos más distantes usando la caja delimitadora."""
    logging.info("Buscando el par de nodos geográficamente más distantes (aproximación)...")
    extremos: Dict[str, Tuple[float, Any]] = {
        'min_lat': (float('inf'), None), 'max_lat': (float('-inf'), None),
        'min_lon': (float('inf'), None), 'max_lon': (float('-inf'), None)
    }
    
    # Itera sobre los vértices para encontrar los 4 puntos extremos
    for v in graph.vs:
        if v['lat'] is None or v['lon'] is None: continue
        lat, lon = v['lat'], v['lon']
        if lat < extremos['min_lat'][0]: extremos['min_lat'] = (lat, v.index)
        if lat > extremos['max_lat'][0]: extremos['max_lat'] = (lat, v.index)
        if lon < extremos['min_lon'][0]: extremos['min_lon'] = (lon, v.index)
        if lon > extremos['max_lon'][0]: extremos['max_lon'] = (lon, v.index)

    nodos_candidatos_ids = {node_id for _, node_id in extremos.values() if node_id is not None}
    if len(nodos_candidatos_ids) < 2:
        logging.error("No se encontraron suficientes nodos con coordenadas para determinar un rango.")
        return None

    # Compara las distancias entre los nodos extremos para encontrar el par más distante
    max_dist, par_mas_distante = -1.0, None
    for u_id, v_id in combinations(nodos_candidatos_ids, 2):
        dist = haversine_distance(graph.vs[u_id]['lat'], graph.vs[u_id]['lon'], graph.vs[v_id]['lat'], graph.vs[v_id]['lon'])
        if dist > max_dist:
            max_dist, par_mas_distante = dist, (u_id, v_id)
            
    if par_mas_distante:
        logging.info(f"Par más distante encontrado: Nodos {par_mas_distante} (Distancia: {max_dist:.2f} km)")
    return par_mas_distante

def encontrar_camino_mas_corto_dijkstra(graph: ig.Graph, source: int, sink: int) -> Tuple[Optional[List[int]], Optional[float], float]:
    """Calcula el camino más corto usando Dijkstra y devuelve el camino, su coste y el tiempo de ejecución."""
    logging.info(f"Iniciando algoritmo de Dijkstra de nodo {source} a {sink}.")
    start_time = time.time()
    
    # igraph es extremadamente rápido para esto.
    # shortest_paths devuelve el coste (longitud total del camino)
    # get_shortest_paths devuelve la secuencia de nodos (vértices)
    try:
        coste_total = graph.shortest_paths(source=source, target=sink, weights='weight')[0][0]
        camino_nodos = graph.get_shortest_paths(v=source, to=sink, weights='weight', output='vpath')[0]
        
        tiempo_total = time.time() - start_time
        logging.info(f"Dijkstra completado en {tiempo_total:.4f} segundos.")
        
        if not camino_nodos: # Si no hay camino, igraph devuelve lista vacía
            return None, None, tiempo_total
            
        return camino_nodos, coste_total, tiempo_total
    except Exception as e:
        logging.error(f"Error durante la ejecución de Dijkstra: {e}")
        tiempo_total = time.time() - start_time
        return None, None, tiempo_total

# --- Funciones de Visualización con Folium ---

def crear_mapa_camino_corto(graph: ig.Graph, camino: List[int], coste: float, source: int, sink: int) -> Optional[folium.Map]:
    """Crea un mapa base con el camino más corto encontrado."""

    logging.info("Creando mapa base con el camino más corto...")
    try:
        start_coords = (graph.vs[source]['lat'], graph.vs[source]['lon'])
        # Usar un mapa base en escala de grises para que los colores resalten
        m = folium.Map(location=start_coords, zoom_start=6, tiles="CartoDB positron")
        
        # Marcadores mejorados para origen y destino
        icon_source = BeautifyIcon(icon='play', border_color='#2ca02c', text_color='#2ca02c', icon_shape='circle')
        icon_sink = BeautifyIcon(icon='stop', border_color='#d62728', text_color='#d62728', icon_shape='circle')
        folium.Marker(location=start_coords, popup=f"SOURCE: Nodo {source}", tooltip="Punto de Origen", icon=icon_source).add_to(m)
        folium.Marker(location=(graph.vs[sink]['lat'], graph.vs[sink]['lon']), popup=f"SINK: Nodo {sink}", tooltip="Punto de Destino", icon=icon_sink).add_to(m)
        
        # Capa para el camino más corto
        path_group = folium.FeatureGroup(name='Camino Más Corto (Dijkstra)', show=True)
        puntos = [(graph.vs[nid]['lat'], graph.vs[nid]['lon']) for nid in camino if graph.vs[nid]['lat'] is not None]
        
        folium.PolyLine(
            puntos, 
            color='#1f77b4',  # Un azul distintivo
            weight=5, 
            opacity=0.9, 
            tooltip=f"Camino más corto: {coste:.2f} km"
        ).add_to(path_group)
        
        path_group.add_to(m)
        return m
    except Exception as e:
        logging.error(f"No se pudo crear el mapa base. Error: {e}")
        return None

def agregar_caminos_random_al_mapa(m: folium.Map, graph: ig.Graph, num_caminos: int):
    """Genera caminos aleatorios (usando Dijkstra) y los agrega al mapa para contexto."""
    
    logging.info(f"Generando {num_caminos} caminos aleatorios para dar contexto a la red...")
    caminos_generados = []
    max_node_id = graph.vcount() - 1
    iterator = tqdm(range(num_caminos), desc="Generando caminos aleatorios")

    for _ in iterator:
        # Intentar hasta encontrar un par de nodos válidos con un camino entre ellos
        for _ in range(10): # Limitar intentos para evitar bucles infinitos
            u, v = random.randint(0, max_node_id), random.randint(0, max_node_id)
            if u == v or graph.vs[u]['lat'] is None or graph.vs[v]['lat'] is None: continue
            
            camino_nodos = graph.get_shortest_paths(u, to=v, weights='weight', output='vpath')
            
            if camino_nodos and camino_nodos[0]:
                camino = camino_nodos[0]
                if all(graph.vs[nid]['lat'] is not None for nid in camino):
                    # Calcular coste sumando los pesos de las aristas del camino
                    coste = sum(graph.es[graph.get_eid(camino[i], camino[i+1])]['weight'] for i in range(len(camino)-1))
                    if coste > 0:
                        caminos_generados.append({'path': camino, 'length_km': coste})
                        break # Camino válido encontrado

    if not caminos_generados:
        logging.warning("No se pudo generar ningún camino aleatorio válido con coordenadas completas.")
        return

    camino_mas_corto_muestra = min(caminos_generados, key=lambda x: x['length_km'])
    camino_mas_largo_muestra = max(caminos_generados, key=lambda x: x['length_km'])

    random_group = folium.FeatureGroup(name=f'{num_caminos} Caminos Aleatorios (Contexto)', show=False)

    for camino_info in caminos_generados:
        puntos = [(graph.vs[nid]['lat'], graph.vs[nid]['lon']) for nid in camino_info['path']]
        if camino_info == camino_mas_corto_muestra:
            color, weight, opacity, tooltip = '#d62728', 4, 1.0, f"Más corto de la muestra ({camino_info['length_km']:.2f} km)"
        elif camino_info == camino_mas_largo_muestra:
            color, weight, opacity, tooltip = '#2ca02c', 4, 1.0, f"Más largo de la muestra ({camino_info['length_km']:.2f} km)"
        else:
            color, weight, opacity, tooltip = '#555555', 1, 0.5, f"Camino de {camino_info['length_km']:.2f} km"
        
        folium.PolyLine(puntos, color=color, weight=weight, opacity=opacity, tooltip=tooltip).add_to(random_group)
        
    random_group.add_to(m)

# --- Bloque Principal de Ejecución ---
if __name__ == "__main__":
    # --- Configuración ---
    GRAFO_PKL_ENTRADA = 'grafo_igraph_paralelizado.pkl'
    MAPA_HTML_SALIDA = 'analisis_de_red_dijkstra.html'
    NUM_CAMINOS_RANDOM = 1000

    # 1. Cargar el grafo
    mi_grafo = cargar_grafo(GRAFO_PKL_ENTRADA)

    if mi_grafo:
        # 1.5. Asegurar que el grafo tiene pesos para Dijkstra
        preparar_pesos_geograficos(mi_grafo)

        # 2. Encontrar source/sink automáticos
        par_distante = encontrar_nodos_mas_distantes_aprox(mi_grafo)

        if par_distante:
            SOURCE_NODE, SINK_NODE = par_distante
            
            # 3. Calcular el camino más corto con Dijkstra
            camino_optimo, coste_total, tiempo_dijkstra = encontrar_camino_mas_corto_dijkstra(mi_grafo, SOURCE_NODE, SINK_NODE)
            
            if camino_optimo and coste_total is not None:
                # 4. Crear el mapa base con el camino más corto
                mapa_resultado = crear_mapa_camino_corto(mi_grafo, camino_optimo, coste_total, SOURCE_NODE, SINK_NODE)

                if mapa_resultado:
                    # 5. Agregar la capa de caminos aleatorios para contexto
                    agregar_caminos_random_al_mapa(mapa_resultado, mi_grafo, NUM_CAMINOS_RANDOM)
                    
                    folium.LayerControl().add_to(mapa_resultado)
                    mapa_resultado.save(MAPA_HTML_SALIDA)
                    
                    # 7. Imprimir resumen final en la consola
                    print("\n" + "="*60)
                    print("      ANÁLISIS DE RUTA ÓPTIMA (DIJKSTRA) COMPLETADO")
                    print("="*60)
                    print(f"Rango de análisis (automático): Nodos {SOURCE_NODE} a {SINK_NODE}")
                    print(f"Distancia del Camino Más Corto: {coste_total:.2f} km")
                    print(f"Número de saltos (nodos) en el camino: {len(camino_optimo)}")
                    print(f"Tiempo de cálculo (Dijkstra): {tiempo_dijkstra:.4f} segundos")
                    print(f"\nSe ha generado un mapa interactivo en: '{MAPA_HTML_SALIDA}'")
                    print("El mapa contiene las siguientes capas (usa el control de capas):")
                    print("  - [Visible] Camino Más Corto: La ruta óptima encontrada por Dijkstra.")
                    print("  - [Oculta]  Caminos Aleatorios: Una muestra para contexto de la red.")
                    print("      -> El más largo (distancia) de la muestra se muestra en VERDE.")
                    print("      -> El más corto (distancia) de la muestra se muestra en ROJO.")
                    print("="*60)
                else:
                    logging.error("No se pudo crear el objeto de mapa base. Abortando visualización.")
            else:
                logging.error(f"No se encontró un camino entre el nodo {SOURCE_NODE} y el {SINK_NODE}.")
        else:
            logging.error("No se pudo determinar el par de nodos de origen/destino. Abortando análisis.")
