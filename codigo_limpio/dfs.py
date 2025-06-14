# analisis_dfs_optimizado.py
"""
Script para análisis de red usando Búsqueda en Profundidad (DFS).

VERSIÓN OPTIMIZADA:
- Se ha re-implementado el DFS para usar la técnica de "parent pointers"
  en lugar de almacenar rutas completas en la pila. Esto resuelve el
  problema de 'MemoryError' en grafos muy grandes y profundos.
"""

import pickle
import time
import logging
import random
from collections import deque
from math import radians, sin, cos, sqrt, atan2
from itertools import combinations
import igraph as ig
from typing import Dict, Optional, Tuple, List

# --- Manejo de dependencias opcionales (sin cambios) ---
try:
    import folium
    from folium.plugins import BeautifyIcon
    FOLIUM_DISPONIBLE = True
except ImportError:
    FOLIUM_DISPONIBLE = False
try:
    from tqdm import tqdm
    TQDM_DISPONIBLE = True
except ImportError:
    TQDM_DISPONIBLE = False

# --- Configuración del Logging (sin cambios) ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler("analisis_dfs.log", mode='w', encoding='utf-8'),
                              logging.StreamHandler()])

# --- Funciones de Carga y Cálculo (sin cambios) ---

def cargar_grafo(grafo_path: str) -> Optional[ig.Graph]:
    logging.info(f"Cargando el grafo desde '{grafo_path}'...")
    start_time = time.time()
    try:
        with open(grafo_path, 'rb') as f: g = pickle.load(f)
        end_time = time.time()
        logging.info(f"Grafo cargado en {end_time - start_time:.2f} segundos. {g.summary()}")
        return g
    except Exception as e:
        logging.error(f"Error crítico al cargar el grafo: {e}"); return None

def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371; dlat, dlon = radians(lat2 - lat1), radians(lon2 - lon1)
    a = sin(dlat / 2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2)**2
    return R * 2 * atan2(sqrt(a), sqrt(1 - a))

def encontrar_nodos_mas_distantes_aprox(graph: ig.Graph) -> Optional[Tuple[int, int]]:
    logging.info("Buscando el par de nodos geográficamente más distantes (aproximación)...")
    extremos = {'min_lat': (float('inf'), None), 'max_lat': (float('-inf'), None),
                'min_lon': (float('inf'), None), 'max_lon': (float('-inf'), None)}
    for v in graph.vs:
        if v.index == 0 or v['lat'] is None: continue
        lat, lon = v['lat'], v['lon']
        if lat < extremos['min_lat'][0]: extremos['min_lat'] = (lat, v.index)
        if lat > extremos['max_lat'][0]: extremos['max_lat'] = (lat, v.index)
        if lon < extremos['min_lon'][0]: extremos['min_lon'] = (lon, v.index)
        if lon > extremos['max_lon'][0]: extremos['max_lon'] = (lon, v.index)
    nodos_candidatos_ids = {node_id for _, node_id in extremos.values() if node_id is not None}
    if len(nodos_candidatos_ids) < 2: return None
    max_dist, par_mas_distante = -1, None
    for u_id, v_id in combinations(nodos_candidatos_ids, 2):
        dist = haversine_distance(graph.vs[u_id]['lat'], graph.vs[u_id]['lon'], graph.vs[v_id]['lat'], graph.vs[v_id]['lon'])
        if dist > max_dist: max_dist, par_mas_distante = dist, (u_id, v_id)
    if par_mas_distante: logging.info(f"Par más distante encontrado: Nodos {par_mas_distante} (Distancia: {max_dist:.2f} km)")
    return par_mas_distante

# --- DFS OPTIMIZADO PARA MEMORIA ---

def dfs_encontrar_camino_optimizado(graph: ig.Graph, source: int, sink: int) -> Tuple[Optional[List[int]], float]:
    """
    Encuentra UNA ruta entre source y sink usando DFS iterativo con la técnica
    de "parent pointers" para evitar errores de memoria.
    """
    logging.info(f"Iniciando Búsqueda en Profundidad (DFS) optimizada de nodo {source} a {sink}.")
    start_time = time.time()
    
    # La pila solo almacena el nodo a visitar, no la ruta completa
    stack = [source]
    # El diccionario 'parents' guarda el rastro para reconstruir la ruta
    parents = {source: None}

    path_found = False
    while stack:
        current_node = stack.pop()

        if current_node == sink:
            path_found = True
            break # ¡Éxito! Salimos del bucle para reconstruir la ruta

        # Añadir vecinos a la pila si no han sido visitados (no tienen padre)
        for neighbor in reversed(graph.neighbors(current_node, mode='out')):
            if neighbor not in parents:
                parents[neighbor] = current_node
                stack.append(neighbor)
    
    end_time = time.time()
    
    # Reconstrucción de la ruta (solo si se encontró el destino)
    if path_found:
        logging.info(f"DFS encontró una ruta en {end_time - start_time:.4f} segundos.")
        path = []
        curr = sink
        while curr is not None:
            path.append(curr)
            curr = parents.get(curr) # Usamos .get() por seguridad
        path.reverse() # La ruta se construye de sink a source
        return path, end_time - start_time
    else:
        logging.warning(f"DFS no encontró una ruta entre {source} y {sink} después de {end_time - start_time:.2f} s.")
        return None, end_time - start_time


# --- Funciones de Visualización (sin cambios) ---

def crear_mapa_con_dfs(graph: ig.Graph, dfs_path: List[int], source: int, sink: int) -> Optional[folium.Map]:
    # ... (esta función es idéntica a la anterior)
    if not FOLIUM_DISPONIBLE: return None
    logging.info("Creando mapa base con la ruta DFS...")
    try:
        start_coords = (graph.vs[source]['lat'], graph.vs[source]['lon'])
        m = folium.Map(location=start_coords, zoom_start=6, tiles="CartoDB positron")
        icon_source = BeautifyIcon(icon='play', border_color='#2ca02c', text_color='#2ca02c', icon_shape='circle')
        icon_sink = BeautifyIcon(icon='stop', border_color='#d62728', text_color='#d62728', icon_shape='circle')
        folium.Marker(location=start_coords, popup=f"SOURCE: Nodo {source}", tooltip="Punto de Origen", icon=icon_source).add_to(m)
        folium.Marker(location=(graph.vs[sink]['lat'], graph.vs[sink]['lon']), popup=f"SINK: Nodo {sink}", tooltip="Punto de Destino", icon=icon_sink).add_to(m)
        dfs_group = folium.FeatureGroup(name='Ruta Encontrada (DFS)', show=True)
        puntos = [(graph.vs[nid]['lat'], graph.vs[nid]['lon']) for nid in dfs_path if graph.vs[nid]['lat'] is not None]
        folium.PolyLine(puntos, color='#ff7f0e', weight=5, opacity=0.9, tooltip=f"Ruta DFS ({len(dfs_path)} nodos)").add_to(dfs_group)
        dfs_group.add_to(m)
        return m
    except Exception as e:
        logging.error(f"No se pudo crear el mapa base. Error: {e}"); return None

def agregar_caminos_random_al_mapa(m: folium.Map, graph: ig.Graph, num_caminos: int):
    # ... (esta función es idéntica a la anterior)
    if not FOLIUM_DISPONIBLE or m is None: return
    logging.info(f"Generando {num_caminos} caminos aleatorios para dar contexto a la red...")
    caminos_generados, max_node_id = [], graph.vcount() - 1
    iterator = tqdm(range(num_caminos), desc="Generando caminos aleatorios") if TQDM_DISPONIBLE else range(num_caminos)
    for _ in iterator:
        while True:
            u, v = random.randint(1, max_node_id), random.randint(1, max_node_id)
            if u == v: continue
            camino_nodos = graph.get_shortest_paths(u, to=v, weights=None, output='vpath')
            if camino_nodos and camino_nodos[0]:
                camino = camino_nodos[0]
                if all(graph.vs[nid]['lat'] is not None for nid in camino):
                    dist_km = sum(haversine_distance(graph.vs[camino[i]]['lat'], graph.vs[camino[i]]['lon'],
                                                    graph.vs[camino[i+1]]['lat'], graph.vs[camino[i+1]]['lon'])
                                  for i in range(len(camino) - 1))
                    if dist_km > 0:
                        caminos_generados.append({'path': camino, 'length_km': dist_km}); break
    if not caminos_generados: return
    camino_mas_corto = min(caminos_generados, key=lambda x: x['length_km'])
    camino_mas_largo = max(caminos_generados, key=lambda x: x['length_km'])
    random_group = folium.FeatureGroup(name=f'{num_caminos} Caminos Aleatorios (Contexto)', show=False)
    for c in caminos_generados:
        if c == camino_mas_largo or c == camino_mas_corto: continue
        puntos = [(graph.vs[nid]['lat'], graph.vs[nid]['lon']) for nid in c['path']]
        folium.PolyLine(puntos, color='#555555', weight=1, opacity=0.5).add_to(random_group)
    p_corto = [(graph.vs[nid]['lat'], graph.vs[nid]['lon']) for nid in camino_mas_corto['path']]
    folium.PolyLine(p_corto, color='#d62728', weight=4, opacity=1.0, tooltip=f"Camino más corto ({camino_mas_corto['length_km']:.2f} km)").add_to(random_group)
    p_largo = [(graph.vs[nid]['lat'], graph.vs[nid]['lon']) for nid in camino_mas_largo['path']]
    folium.PolyLine(p_largo, color='#2ca02c', weight=4, opacity=1.0, tooltip=f"Camino más largo ({camino_mas_largo['length_km']:.2f} km)").add_to(random_group)
    random_group.add_to(m)


# --- Bloque Principal de Ejecución (actualizado para usar la función optimizada) ---
if __name__ == "__main__":
    GRAFO_PKL_ENTRADA = 'grafo_igraph_paralelizado.pkl'
    MAPA_HTML_SALIDA = 'analisis_dfs_optimizado.html'
    NUM_CAMINOS_RANDOM = 1000

    if not FOLIUM_DISPONIBLE: logging.error("Folium es necesario. Instálalo con: pip install folium")
    if not TQDM_DISPONIBLE: logging.warning("tqdm no está instalado. No habrá barra de progreso. Instálalo con: pip install tqdm")

    mi_grafo = cargar_grafo(GRAFO_PKL_ENTRADA)
    if mi_grafo:
        par_distante = encontrar_nodos_mas_distantes_aprox(mi_grafo)
        if par_distante:
            SOURCE_NODE, SINK_NODE = par_distante
            
            # --- Se llama a la nueva función optimizada ---
            dfs_path, tiempo_dfs = dfs_encontrar_camino_optimizado(mi_grafo, SOURCE_NODE, SINK_NODE)
            
            if dfs_path:
                mapa = crear_mapa_con_dfs(mi_grafo, dfs_path, SOURCE_NODE, SINK_NODE)
                if mapa:
                    agregar_caminos_random_al_mapa(mapa, mi_grafo, NUM_CAMINOS_RANDOM)
                    folium.LayerControl().add_to(mapa)
                    mapa.save(MAPA_HTML_SALIDA)
                    
                    print("\n" + "="*55)
                    print("      ANÁLISIS DE RUTA CON DFS (VERSIÓN OPTIMIZADA)")
                    print("="*55)
                    print(f"Rango de análisis (automático): Nodos {SOURCE_NODE} a {SINK_NODE}")
                    print(f"Ruta encontrada por DFS con {len(dfs_path)} nodos.")
                    print(f"Tiempo de cálculo (DFS): {tiempo_dfs:.4f} segundos.")
                    print("\nSe ha generado un mapa interactivo en:", f"'{MAPA_HTML_SALIDA}'")
                    print("El mapa contiene las mismas capas que el análisis anterior.")
                    print("="*55)
                else:
                    logging.error("No se pudo crear el objeto de mapa base.")
            else:
                logging.error("No se pudo encontrar una ruta con DFS entre los nodos seleccionados.")