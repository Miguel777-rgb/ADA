import pickle
import time
import logging
import random
import igraph as ig
from typing import Optional, List, Tuple
import folium
from folium.plugins import BeautifyIcon

# --- Configuración del Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
handlers=[logging.FileHandler("analisis_dfs.log", mode='w', encoding='utf-8'),
logging.StreamHandler()])

def cargar_grafo(grafo_path: str) -> Optional[ig.Graph]:
    logging.info(f"Cargando el grafo desde '{grafo_path}'...")
    start_time = time.time()
    try:
        with open(grafo_path, 'rb') as f:
            g = pickle.load(f)
        end_time = time.time()
        logging.info(f"Grafo cargado en {end_time - start_time:.2f} segundos. {g.summary()}")
        return g
    except FileNotFoundError:
        logging.error(f"Error crítico: El archivo de grafo '{grafo_path}' no fue encontrado.")
        return None
    except Exception as e:
        logging.error(f"Error crítico al cargar el grafo: {e}")
        return None

# --- ALGORITMO DFS ---
def dfs_exploracion_desde_un_punto(graph: ig.Graph, source: int) -> Tuple[List[int], float]:
    logging.info(f"Iniciando travesía DFS completa desde el nodo {source}.")
    start_time = time.time()
    stack = [source]
    visitados = {source}
    nodos_explorados_en_orden = []
    while stack:
        current_node = stack.pop()
        nodos_explorados_en_orden.append(current_node)
        for neighbor in reversed(graph.neighbors(current_node, mode='out')):
            if neighbor not in visitados:
                visitados.add(neighbor)
                stack.append(neighbor)
    end_time = time.time()
    logging.info(f"Travesía DFS completada en {end_time - start_time:.4f} segundos. Total de nodos visitados: {len(nodos_explorados_en_orden)}.")
    return nodos_explorados_en_orden, end_time - start_time

<<<<<<< HEAD
# --- Función de Visualización ---
=======
# --- Función de Visualización  ---
>>>>>>> f470e255f00c5659d582126782f9cf7abac44dd6
def crear_mapa_multicapa_dfs(
    graph: ig.Graph, 
    source_node: int,
    nodos_visitados: List[int], 
    max_puntos_linea: int = 5000,
    num_nodos_muestra: int = 1000
) -> Optional[folium.Map]:
    """
    Crea un mapa con dos capas consistentes:
    1. El camino completo de DFS (submuestreado).
    2. Una muestra aleatoria de nodos tomados DIRECTAMENTE del camino visualizado.
    """
    if not nodos_visitados:
        logging.warning("No hay nodos visitados para mostrar en el mapa.")
        return None

    logging.info("Creando mapa multicapa con visualización consistente.")
    try:
        start_coords = (graph.vs[source_node]['lat'], graph.vs[source_node]['lon'])
        m = folium.Map(location=start_coords, zoom_start=4, tiles="CartoDB positron")
        
        # --- 1. Definir el camino visual (submuestreado) ---
        # Esta lista de nodos será la FUENTE DE VERDAD para ambas capas.
        if len(nodos_visitados) > max_puntos_linea:
            step = len(nodos_visitados) // max_puntos_linea
            camino_visualizado = nodos_visitados[::step]
            if camino_visualizado[-1] != nodos_visitados[-1]:
                camino_visualizado.append(nodos_visitados[-1])
        else:
            camino_visualizado = nodos_visitados
        
        # --- CAPA 1: Camino Completo (Línea Naranja) ---
        fg_camino = folium.FeatureGroup(name="Camino DFS (visual)", show=True)
        puntos_del_camino = [(graph.vs[n]['lat'], graph.vs[n]['lon']) for n in camino_visualizado if graph.vs[n]['lat'] is not None]
        folium.PolyLine(
            puntos_del_camino, color="#ff7f0e", weight=2, opacity=0.8,
            tooltip=f"Camino DFS ({len(camino_visualizado)} de {len(nodos_visitados)} nodos)"
        ).add_to(fg_camino)
        m.add_child(fg_camino)

        # --- CAPA 2: Muestra de Nodos (Puntos Azules) ---
        # CORRECCIÓN CLAVE: La muestra se toma de 'camino_visualizado', no de 'nodos_visitados'.
        fg_puntos = folium.FeatureGroup(name=f"Muestra de Nodos (del camino visual)", show=True)
        end_node_id = nodos_visitados[-1]
        
        # Candidatos para la muestra son los nodos del camino visual, excluyendo inicio/fin.
        candidatos = [n for n in camino_visualizado if n != source_node and n != end_node_id]
        
        # Aseguramos no pedir más muestras de las disponibles.
        num_muestras_a_tomar = min(len(candidatos), num_nodos_muestra - 2)
        nodos_a_mostrar_aleatorios = random.sample(candidatos, num_muestras_a_tomar) if num_muestras_a_tomar > 0 else []

        # La muestra final siempre incluye inicio y fin, más los aleatorios.
        nodos_muestra_final = [source_node, end_node_id] + nodos_a_mostrar_aleatorios
        
        for node_id in nodos_muestra_final:
            coords = (graph.vs[node_id]['lat'], graph.vs[node_id]['lon'])
            if coords[0] is not None:
                folium.CircleMarker(
                    location=coords, radius=3.5, color='#1f77b4', fill=True, fill_color='#1f77b4',
                    fill_opacity=0.7, popup=f"Nodo de muestra: {node_id}"
                ).add_to(fg_puntos)
        m.add_child(fg_puntos)

        # --- MARCADORES PERMANENTES (Inicio y Fin Reales) ---
        folium.Marker(
            location=start_coords, popup=f"INICIO DFS: Nodo {source_node}", 
            tooltip="Punto de Inicio", icon=BeautifyIcon(icon='play', border_color='#2ca02c', text_color='#2ca02c'), z_index_offset=1000
        ).add_to(m)

        end_real_coords = (graph.vs[end_node_id]['lat'], graph.vs[end_node_id]['lon'])
        folium.Marker(
            location=end_real_coords, popup=f"FIN REAL DE LA TRAVESÍA: Nodo {end_node_id}", 
            tooltip="Fin de la Travesía", icon=BeautifyIcon(icon='stop', border_color='#d62728', text_color='#d62728'), z_index_offset=1000
        ).add_to(m)
        
        # --- CONTROL DE CAPAS ---
        folium.LayerControl().add_to(m)

        return m
    except Exception as e:
        logging.error(f"No se pudo crear el mapa. Error: {e}")
        return None

# --- Bloque Principal de Ejecución ---
if __name__ == "__main__":
    GRAFO_PKL_ENTRADA = 'grafo_igraph_paralelizado.pkl'
    MAPA_HTML_SALIDA = 'mapa_dfs_multicapa.html'

    mi_grafo = cargar_grafo(GRAFO_PKL_ENTRADA)
    
    if mi_grafo:
        try:
            SOURCE_NODE = random.randrange(mi_grafo.vcount())
            logging.info(f"Nodo de inicio de la travesía DFS seleccionado al azar: {SOURCE_NODE}")
        except ValueError:
            logging.error("El grafo está vacío y no se puede seleccionar un nodo de inicio.")
            exit()

        nodos_visitados_dfs, tiempo_dfs = dfs_exploracion_desde_un_punto(mi_grafo, SOURCE_NODE)
        
        if nodos_visitados_dfs:
            mapa = crear_mapa_multicapa_dfs(mi_grafo, SOURCE_NODE, nodos_visitados_dfs)
            
            if mapa:
                mapa.save(MAPA_HTML_SALIDA)
                
                print("\n" + "="*55)
                print("      ANÁLISIS DE TRAVESÍA DFS CON MÚLTIPLES CAPAS")
                print("="*55)
                print(f"Travesía iniciada desde el nodo (aleatorio): {SOURCE_NODE}")
                print(f"Tiempo total de la travesía: {tiempo_dfs:.4f} segundos.")
                print(f"Total de nodos visitados en el componente: {len(nodos_visitados_dfs)}")
                
                print(f"\nMapa interactivo guardado en: '{MAPA_HTML_SALIDA}'")
                print(" -> El mapa contiene DOS capas consistentes que puedes activar/desactivar:")
                print("    1. 'Camino DFS': La línea que muestra la forma de la travesía.")
                print("    2. 'Muestra de Nodos': Puntos aleatorios tomados DEL CAMINO VISIBLE.")
                print("="*55)
            else:
                logging.error("No se pudo crear el objeto de mapa base.")
        else:
            logging.info("La travesía DFS no visitó ningún nodo.")
