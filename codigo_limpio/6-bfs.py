import igraph as ig
import folium
from folium.plugins import BeautifyIcon, AntPath
import pickle
import time
import logging
import os
import webbrowser
import random
from collections import deque

# --- Configuración ---
GRAFO_PICKLE_PATH = 'grafo_igraph_paralelizado.pkl'
MAPA_HTML_OUTPUT = 'mapa_camino_largo_con_arbol.html' # Nuevo nombre de archivo

# Nodo de inicio
NODO_INICIO = 9999427

# Define la cantidad máxima de nodos a explorar.
MAX_NODOS_A_EXPLORAR = 5000

# --- Configuración del Logging ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("analisis_bfs.log", mode='w', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

def visualizar_arbol_bfs(graph, start_node_id, max_nodes, output_filename):
    """
    Realiza una búsqueda BFS, resalta el camino más largo con un AntPath rojo
    y muestra el árbol de búsqueda completo con líneas estáticas.
    """
    logging.info(f"Iniciando BFS desde el nodo {start_node_id} con un límite de {max_nodes} nodos.")
    start_time = time.time()

    # 1. Realizar BFS para recolectar nodos, niveles y la estructura del árbol
    queue = deque([(start_node_id, 0)])
    visited = {start_node_id}
    levels = {start_node_id: 0}
    parent_map = {start_node_id: None}
    tree_edges = []
    
    while queue and len(visited) < max_nodes:
        current_node, current_level = queue.popleft()
        neighbors = graph.neighbors(current_node, mode="out")
        
        for neighbor in neighbors:
            if neighbor not in visited:
                if len(visited) >= max_nodes: break
                visited.add(neighbor)
                levels[neighbor] = current_level + 1
                parent_map[neighbor] = current_node
                tree_edges.append((current_node, neighbor))
                queue.append((neighbor, current_level + 1))
    
    logging.info(f"BFS completado. Se exploraron {len(visited)} nodos y {len(tree_edges)} aristas en {time.time() - start_time:.2f} segundos.")

    # 2. Encontrar el nodo más lejano en el árbol BFS
    farthest_node = max(visited, key=lambda node: levels[node])
    max_level = levels[farthest_node]
    logging.info(f"El camino más largo encontrado tiene {max_level} saltos y termina en el nodo {farthest_node}.")

    # 3. Preparar el mapa con Folium
    logging.info("Creando mapa con Folium...")
    lats = [graph.vs[n]['lat'] for n in visited]
    lons = [graph.vs[n]['lon'] for n in visited]
    map_center = [sum(lats) / len(lats), sum(lons) / len(lons)]

    m = folium.Map(location=map_center, zoom_start=10, tiles="CartoDB positron")

    # 4. Crear capas (FeatureGroups) para cada elemento visual
    nodos_explorados_group = folium.FeatureGroup(name=f"Nodos Explorados ({len(visited)})", show=True)
    aristas_estaticas_group = folium.FeatureGroup(name=f"Árbol BFS Estático ({len(tree_edges)} aristas)", show=True)
    camino_animado_group = folium.FeatureGroup(name="Camino Más Largo (Animado)", show=True)

    # 5. Añadir los nodos explorados al mapa
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#9467bd', '#8c564b', '#e377c2']
    for node_id in visited:
        lat, lon = graph.vs[node_id]['lat'], graph.vs[node_id]['lon']
        level = levels[node_id]
        popup_text = f"<b>Nodo:</b> {node_id}<br><b>Nivel:</b> {level}"
        
        folium.CircleMarker(
            location=[lat, lon], radius=3, color=colors[level % len(colors)],
            fill=True, fill_color=colors[level % len(colors)], fill_opacity=0.6, popup=popup_text
        ).add_to(nodos_explorados_group)

    # --- NUEVO: 6. Añadir las aristas estáticas del árbol BFS completo ---
    for parent, child in tree_edges:
        parent_coords = (graph.vs[parent]['lat'], graph.vs[parent]['lon'])
        child_coords = (graph.vs[child]['lat'], graph.vs[child]['lon'])
        
        folium.PolyLine(
            locations=[parent_coords, child_coords],
            color='#AAAAAA',  # Un color gris claro para que no domine
            weight=1,         # Línea delgada
            opacity=0.7
        ).add_to(aristas_estaticas_group)
        
    # 7. Reconstruir y añadir el camino más largo (rojo y animado)
    longest_path_coords = []
    curr = farthest_node
    while curr is not None:
        coords = (graph.vs[curr]['lat'], graph.vs[curr]['lon'])
        longest_path_coords.append(coords)
        curr = parent_map.get(curr)
    longest_path_coords.reverse()

    AntPath(
        locations=longest_path_coords,
        delay=1000, weight=5, color='#d62728', # Rojo y grueso
        pulse_color='#FFFFFF', dash_array=[25, 40]
    ).add_to(camino_animado_group)

    # 8. Añadir marcadores de INICIO y FIN sobre todas las capas
    start_lat, start_lon = graph.vs[start_node_id]['lat'], graph.vs[start_node_id]['lon']
    folium.Marker(
        location=[start_lat, start_lon], popup=f"<b>INICIO:</b> {start_node_id}",
        tooltip=f"INICIO: Nodo {start_node_id}",
        icon=BeautifyIcon(icon='play', icon_shape='circle', border_color='#2ca02c', text_color='#2ca02c', background_color='#FFF')
    ).add_to(m)

    end_lat, end_lon = graph.vs[farthest_node]['lat'], graph.vs[farthest_node]['lon']
    folium.Marker(
        location=[end_lat, end_lon], popup=f"<b>FIN:</b> {farthest_node}<br><b>Nivel:</b> {max_level}",
        tooltip=f"FIN: Nodo {farthest_node}",
        icon=BeautifyIcon(icon='stop', icon_shape='circle', border_color='#d62728', text_color='#d62728', background_color='#FFF')
    ).add_to(m)

    # 9. Añadir todas las capas y el control al mapa
    nodos_explorados_group.add_to(m)
    aristas_estaticas_group.add_to(m) # Añadimos la capa de aristas estáticas
    camino_animado_group.add_to(m)
    folium.LayerControl().add_to(m)

    # 10. Guardar y abrir el mapa
    m.save(output_filename)
    logging.info(f"Mapa guardado en '{output_filename}'.")
    webbrowser.open('file://' + os.path.realpath(output_filename))

if __name__ == "__main__":
    if not os.path.exists(GRAFO_PICKLE_PATH):
        logging.error(f"El archivo del grafo '{GRAFO_PICKLE_PATH}' no fue encontrado.")
    else:
        logging.info(f"Cargando el grafo desde '{GRAFO_PICKLE_PATH}'...")
        start_load_time = time.time()
        with open(GRAFO_PICKLE_PATH, 'rb') as f:
            g = pickle.load(f)
        logging.info(f"Grafo cargado en {time.time() - start_load_time:.2f} segundos.")

        if not (0 <= NODO_INICIO < g.vcount()):
            logging.error(f"El NODO_INICIO ({NODO_INICIO}) está fuera del rango.")
            NODO_INICIO = random.randint(0, g.vcount() - 1)
            logging.warning(f"Se ha seleccionado un nuevo nodo de inicio aleatorio: {NODO_INICIO}")

        visualizar_arbol_bfs(g, NODO_INICIO, MAX_NODOS_A_EXPLORAR, MAPA_HTML_OUTPUT)