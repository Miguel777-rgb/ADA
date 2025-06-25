import pickle
import time
import logging
import random
import igraph as ig
from typing import Optional, List, Tuple
import folium
from folium.plugins import BeautifyIcon
import colorsys

# --- Constantes y Configuración ---
SOURCE_NODE = 1
NUM_CAPAS_DESEADAS = 10 
# Reduce el camino de ~10 millones de nodos a este número antes de crear el mapa.
MAX_NODOS_PARA_MAPA = 50

# --- Configuración del Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler("analisis_dfs.log", mode='w', encoding='utf-8'),
                              logging.StreamHandler()])

# --- Funciones Auxiliares  ---
def cargar_grafo(grafo_path: str) -> Optional[ig.Graph]:
    """Carga un grafo desde un archivo pickle."""
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

def dfs_con_profundidad(graph: ig.Graph, source: int) -> Tuple[List[int], float, int, int]:
    """Realiza un recorrido DFS y devuelve el camino, el tiempo, y el nodo más profundo."""
    logging.info(f"Iniciando travesía DFS desde el nodo {source} y calculando profundidad.")
    start_time = time.time()
    stack = [(source, 0)]
    visitados = {source}
    nodos_explorados_en_orden = []
    max_profundidad, nodo_mas_profundo = 0, source
    while stack:
        current_node, current_depth = stack.pop()
        nodos_explorados_en_orden.append(current_node)
        if current_depth > max_profundidad:
            max_profundidad, nodo_mas_profundo = current_depth, current_node
        for neighbor in reversed(graph.neighbors(current_node, mode='out')):
            if neighbor not in visitados:
                visitados.add(neighbor)
                stack.append((neighbor, current_depth + 1))
    tiempo_total = time.time() - start_time
    logging.info(f"Travesía DFS completada. Nodos: {len(nodos_explorados_en_orden)}. Nodo más profundo: {nodo_mas_profundo} (prof: {max_profundidad}).")
    return nodos_explorados_en_orden, tiempo_total, nodo_mas_profundo, max_profundidad

def generar_color_gradiente_hls(paso_actual: int, total_pasos: int) -> str:
    """Genera un color en formato hexadecimal que va de azul (inicio) a rojo (final)."""
    if total_pasos <= 1: return "#FF0000"
    hue = (240/360) * (1 - (paso_actual / (total_pasos - 1)))
    r, g, b = colorsys.hls_to_rgb(hue, 0.5, 0.8)
    return f'#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}'


# --- FUNCIÓN DE VISUALIZACIÓN OPTIMIZADA CON MARCADORES DE CAPA ---
def crear_mapa_dfs_por_capas(
    graph: ig.Graph, 
    source_node: int,
    camino_a_dibujar: List[int], # Recibe la lista ya reducida
    nodo_mas_profundo: int,
    ultimo_nodo_real: int
) -> Optional[folium.Map]:
    """
    Crea un mapa estático usando un camino pre-reducido y añade marcadores de inicio/fin por capa.
    """
    logging.info(f"Creando mapa estático para dividirse en {NUM_CAPAS_DESEADAS} capas.")
    
    start_coords = (graph.vs[source_node]['lat'], graph.vs[source_node]['lon'])
    m = folium.Map(location=start_coords, zoom_start=3, tiles="CartoDB positron")
    
    total_pasos_visuales = len(camino_a_dibujar)
    
    # Capa de Vértices
    fg_vertices = folium.FeatureGroup(name=f"Vértices del Camino Visualizado ({total_pasos_visuales})", show=True)
    for node_id in camino_a_dibujar:
        coords = (graph.vs[node_id]['lat'], graph.vs[node_id]['lon'])
        folium.CircleMarker(
            location=coords, radius=3, color='#3186cc', fill=True, fill_color='#3186cc', fill_opacity=0.6,
            popup=f"Nodo {node_id}"
        ).add_to(fg_vertices)
    m.add_child(fg_vertices)

    # Lógica para dividir el camino en capas
    num_segmentos = total_pasos_visuales - 1
    segmentos_por_capa = max(1, num_segmentos // NUM_CAPAS_DESEADAS)
    
    for i in range(0, num_segmentos, segmentos_por_capa):
        start_paso, end_paso = i + 1, min(i + segmentos_por_capa, num_segmentos)
        
        fg_camino_segmentado = folium.FeatureGroup(name=f"Camino (Pasos {start_paso}-{end_paso})", show=(i == 0))
        
        # Dibuja las líneas de la capa
        for j in range(i, end_paso):
            coords = [(graph.vs[camino_a_dibujar[j]]['lat'], graph.vs[camino_a_dibujar[j]]['lon']), (graph.vs[camino_a_dibujar[j+1]]['lat'], graph.vs[camino_a_dibujar[j+1]]['lon'])]
            color_segmento = generar_color_gradiente_hls(j, total_pasos_visuales)
            folium.PolyLine(locations=coords, color=color_segmento, weight=2.5, opacity=1.0).add_to(fg_camino_segmentado)
        
        # --- NUEVO: AÑADIR MARCADORES DE INICIO Y FIN PARA ESTA CAPA ---
        start_node_layer = camino_a_dibujar[i]
        end_node_layer = camino_a_dibujar[end_paso]
        
        folium.CircleMarker(
            location=(graph.vs[start_node_layer]['lat'], graph.vs[start_node_layer]['lon']),
            radius=5, color='green', fill=True, fill_color='lightgreen',
            tooltip=f"Inicio Capa: Paso {start_paso}"
        ).add_to(fg_camino_segmentado)

        folium.CircleMarker(
            location=(graph.vs[end_node_layer]['lat'], graph.vs[end_node_layer]['lon']),
            radius=5, color='red', fill=True, fill_color='#ff7f7f', # Un rojo más claro
            tooltip=f"Fin Capa: Paso {end_paso}"
        ).add_to(fg_camino_segmentado)

        m.add_child(fg_camino_segmentado)

    # Marcadores Especiales (Generales)
    folium.Marker(location=start_coords, popup=f"INICIO DFS: Nodo {source_node}", tooltip="Punto de Inicio General",
                  icon=BeautifyIcon(icon='play', border_color='#2ca02c', text_color='#2ca02c')).add_to(m)
    end_real_coords = (graph.vs[ultimo_nodo_real]['lat'], graph.vs[ultimo_nodo_real]['lon'])
    folium.Marker(location=end_real_coords, popup=f"FIN DEL RECORRIDO: Nodo {ultimo_nodo_real}", tooltip="Último Nodo Visitado General",
                  icon=BeautifyIcon(icon='stop', border_color='#d62728', text_color='#d62728')).add_to(m)
    deepest_coords = (graph.vs[nodo_mas_profundo]['lat'], graph.vs[nodo_mas_profundo]['lon'])
    folium.Marker(location=deepest_coords, popup=f"NODO MÁS PROFUNDO: Nodo {nodo_mas_profundo}", tooltip="Nodo Más Profundo General",
                  icon=BeautifyIcon(icon='star', border_color='#800080', text_color='#800080', spin=True)).add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)
    return m

# --- Bloque Principal de Ejecución ---
if __name__ == "__main__":
    GRAFO_PKL_ENTRADA = 'grafo_igraph_paralelizado.pkl'
    MAPA_HTML_SALIDA = 'mapa_dfs_con_marcadores_de_capa.html'

    mi_grafo = cargar_grafo(GRAFO_PKL_ENTRADA)
    
    if mi_grafo and mi_grafo.vs:
        nodos_visitados_dfs, tiempo_dfs, nodo_profundo, prof_max = dfs_con_profundidad(mi_grafo, SOURCE_NODE)
        
        if nodos_visitados_dfs:
            logging.info(f"Reduciendo camino de {len(nodos_visitados_dfs)} nodos a un máximo de {MAX_NODOS_PARA_MAPA} para visualización.")
            if len(nodos_visitados_dfs) > MAX_NODOS_PARA_MAPA:
                step = len(nodos_visitados_dfs) // MAX_NODOS_PARA_MAPA
                camino_para_mapa = nodos_visitados_dfs[::step]
                if camino_para_mapa[-1] != nodos_visitados_dfs[-1]:
                    camino_para_mapa.append(nodos_visitados_dfs[-1])
            else:
                camino_para_mapa = nodos_visitados_dfs
            
            logging.info(f"El camino para el mapa ahora tiene {len(camino_para_mapa)} nodos. Creando mapa...")

            mapa = crear_mapa_dfs_por_capas(
                graph=mi_grafo, 
                source_node=SOURCE_NODE, 
                camino_a_dibujar=camino_para_mapa, 
                nodo_mas_profundo=nodo_profundo,
                ultimo_nodo_real=nodos_visitados_dfs[-1]
            )
            
            if mapa:
                mapa.save(MAPA_HTML_SALIDA)
                print("\n" + "="*60)
                print("         MAPA OPTIMIZADO CREADO CON ÉXITO")
                print("="*60)
                print(f"Mapa interactivo guardado en: '{MAPA_HTML_SALIDA}'")
                print("\nNovedades en el mapa:")
                print(f"  - El camino visual se ha dividido en ~{NUM_CAPAS_DESEADAS} capas.")
                print("  - CADA CAPA DE CAMINO AHORA TIENE SU PROPIO MARCADOR DE INICIO Y FIN.")
                print("  - Marcador especial (⭐) para el nodo más profundo.")
                print("="*60)
    else:
        logging.error("El grafo no se pudo cargar o está vacío.")
