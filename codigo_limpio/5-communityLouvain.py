import pickle
import time
import logging
import random
from collections import Counter, defaultdict
import igraph as ig
from typing import Optional, List, Dict, Tuple
import folium
from folium.plugins import MarkerCluster
import matplotlib
import matplotlib.colors as colors
from tqdm import tqdm

# --- Configuración del Logging  ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
handlers=[logging.FileHandler("analisis_comunidades_louvain_manual.log", mode='w', encoding='utf-8'),
logging.StreamHandler()])

# --- UMBRALES DE VISUALIZACIÓN  ---
UMBRAL_MAX_VISUALIZACION_ARISTAS = 3000
UMBRAL_MAX_NODOS_A_DIBUJAR = 3000

# --- Función de Carga de Grafo  ---
def cargar_grafo(grafo_path: str) -> Optional[ig.Graph]:
    logging.info(f"Cargando el grafo desde '{grafo_path}'..."); start_time = time.time()
    try:
        with open(grafo_path, 'rb') as f: g = pickle.load(f)
        end_time = time.time(); logging.info(f"Grafo cargado en {end_time - start_time:.2f} segundos. {g.summary()}")
        return g
    except Exception as e:
        logging.error(f"Error crítico al cargar el grafo: {e}"); return None

# --- INICIO DE LA IMPLEMENTACIÓN MANUAL DE LOUVAIN ---

def _calcular_delta_modularity(node: int, target_community: int, node_degree: float,
                               links_to_target_community: float, total_community_degree: float,
                               total_edge_weight: float) -> float:
    if total_edge_weight == 0: return 0
    term1 = links_to_target_community / total_edge_weight
    term2 = (total_community_degree * node_degree) / (2 * (total_edge_weight ** 2))
    return term1 - term2

def _louvain_phase1(graph: ig.Graph, total_edge_weight: float) -> Tuple[Dict[int, int], bool]:
    """
    Fase 1: Optimización de modularidad local.
    Devuelve la partición y un booleano que indica si hubo cambios.
    """
    communities = {v.index: v.index for v in graph.vs}
    
    # 1. Obtenemos la lista de todas las fuerzas de los nodos.
    all_strengths = graph.strength(weights='weight' if 'weight' in graph.es.attributes() else None)
    # 2. Creamos el diccionario mapeando el índice del nodo a su fuerza.
    node_degrees = {i: strength for i, strength in enumerate(all_strengths)}
    # -----------------------

    community_degrees = node_degrees.copy()
    
    improved = False
    nodes = list(range(graph.vcount()))
    random.shuffle(nodes)
    
    for node in tqdm(nodes, desc="  Fase 1: Optimizando modularidad"):
        current_community = communities[node]
        best_community = current_community
        max_gain = 0.0

        links_to_communities = defaultdict(float)
        for neighbor in graph.neighbors(node, mode='all'):
            edge_id = graph.get_eid(node, neighbor, directed=False, error=False)
            if edge_id != -1:
                weight = graph.es[edge_id]['weight'] if 'weight' in graph.es.attributes() else 1.0
                links_to_communities[communities[neighbor]] += weight
        
        for community, links_weight in links_to_communities.items():
            if community == current_community: continue
            gain = _calcular_delta_modularity(
                node, community, node_degrees[node], links_weight,
                community_degrees[community], total_edge_weight
            )
            if gain > max_gain:
                max_gain = gain
                best_community = community

        if max_gain > 0:
            community_degrees[current_community] -= node_degrees[node]
            community_degrees[best_community] += node_degrees[node]
            communities[node] = best_community
            improved = True
            
    return communities, improved

def _louvain_phase2(graph: ig.Graph, communities: Dict[int, int]) -> ig.Graph:
    """
    Fase 2: Agregación de comunidades. Construye un nuevo grafo.
    """
    new_nodes = list(set(communities.values()))
    new_node_map = {old_id: new_id for new_id, old_id in enumerate(new_nodes)}
    
    new_graph = ig.Graph(directed=False)
    new_graph.add_vertices(len(new_nodes))
    
    edge_weights = defaultdict(float)
    
    for edge in graph.es:
        source, target = edge.tuple
        c_source = communities[source]
        c_target = communities[target]
        weight = edge['weight'] if 'weight' in graph.es.attributes() else 1.0
        
        if c_source != c_target:
            new_source = new_node_map[c_source]
            new_target = new_node_map[c_target]
            
            # Ordenar para evitar duplicados en grafo no dirigido
            if new_source < new_target:
                edge_weights[(new_source, new_target)] += weight
            else:
                edge_weights[(new_target, new_source)] += weight

    new_graph.add_edges(edge_weights.keys())
    new_graph.es['weight'] = list(edge_weights.values())
    
    return new_graph

def detectar_comunidades_louvain_manual(graph: ig.Graph) -> Dict[int, List[int]]:
    """
    Implementación manual completa del algoritmo de Louvain.
    """
    logging.info("Iniciando detección de comunidades con implementación MANUAL de Louvain...")
    start_time = time.time()
    
    # Crucial: El algoritmo de Louvain trabaja sobre grafos no dirigidos.
    # Esta conversión es necesaria y soluciona el `ValueError`.
    if graph.is_directed():
        logging.info("El grafo es dirigido. Convirtiendo a no dirigido para el análisis de comunidades.")
        # Usamos `as_undirected` que es más seguro que modificar el grafo in-place.
        current_graph = graph.as_undirected(mode='collapse', combine_edges=dict(weight="sum"))
    else:
        current_graph = graph.copy()

    # La membresía inicial mapea cada nodo del grafo ORIGINAL a su comunidad actual.
    membership = {v.index: v.index for v in graph.vs}
    
    pass_num = 1
    while True:
        logging.info(f"--- Louvain - Pass {pass_num} ---")
        
        total_edge_weight = sum(current_graph.es['weight']) if 'weight' in current_graph.es.attributes() else current_graph.ecount()
        if total_edge_weight == 0:
            logging.warning("El grafo no tiene aristas. Deteniendo el algoritmo.")
            break
            
        communities, improved = _louvain_phase1(current_graph, total_edge_weight)
        
        if not improved:
            logging.info("No hubo más mejoras de modularidad. Convergencia alcanzada.")
            break
        
        # Mapear la nueva partición a la membresía global.
        # Si el nodo 5 va a la com. 10, y en la siguiente pasada
        # la com. 10 va a la super-com. 2, el nodo 5 debe pertenecer a la super-com. 2.
        partition_map = communities
        for node_idx in range(graph.vcount()):
             membership[node_idx] = partition_map.get(membership[node_idx], membership[node_idx])

        logging.info("  Fase 2: Agregando comunidades para la siguiente pasada...")
        current_graph = _louvain_phase2(current_graph, communities)
        
        if current_graph.vcount() == 0 or current_graph.ecount() == 0:
            logging.info("El grafo agregado ya no tiene nodos o aristas. Deteniendo.")
            break
            
        pass_num += 1

    final_comunidades = defaultdict(list)
    for node, comm in membership.items():
        final_comunidades[comm].append(node)
        
    end_time = time.time()
    logging.info(f"Algoritmo manual de Louvain completado en {end_time - start_time:.2f} s. Se encontraron {len(final_comunidades)} comunidades.")
    return dict(final_comunidades)

# El resto del código (análisis, colores, visualización y bloque main).
# ... (pegar el resto del código desde aquí)
# --- Función de Análisis  ---
def analizar_y_seleccionar_comunidades(comunidades_dict: Dict[int, List[int]], num_random: int = 20) -> Dict[str, List[int]]:
    logging.info("Analizando y seleccionando comunidades para visualización...");
    if not comunidades_dict or len(comunidades_dict) < 3: logging.warning("No hay suficientes comunidades para un análisis detallado."); return {}
    comunidades_con_tamaño = [(cid, len(miembros)) for cid, miembros in comunidades_dict.items()]
    id_grande, tamaño_grande = max(comunidades_con_tamaño, key=lambda item: item[1])
    com_mayores_a_uno = [c for c in comunidades_con_tamaño if c[1] > 1]
    id_pequeña, tamaño_pequeña = min(com_mayores_a_uno, key=lambda item: item[1]) if com_mayores_a_uno else min(comunidades_con_tamaño, key=lambda item: item[1])
    ids_extremos = {id_grande, id_pequeña}
    posibles_ids_random = [cid for cid, size in comunidades_con_tamaño if cid not in ids_extremos and 5 < size < UMBRAL_MAX_VISUALIZACION_ARISTAS]
    ids_random = random.sample(posibles_ids_random, min(num_random, len(posibles_ids_random))) if posibles_ids_random else []
    seleccion = {'grande': [id_grande], 'pequena': [id_pequeña], 'random': ids_random}
    logging.info(f"Comunidad más grande (ID {id_grande}): {tamaño_grande} miembros.")
    logging.info(f"Comunidad más pequeña (>1 miembro) (ID {id_pequeña}): {tamaño_pequeña} miembros.")
    logging.info(f"Se seleccionaron {len(ids_random)} comunidades aleatorias (tamaño < {UMBRAL_MAX_VISUALIZACION_ARISTAS}).")
    return seleccion

# --- Función de Colores  ---
def crear_mapa_de_colores(tamaño_min: int, tamaño_max: int):
    colormap = matplotlib.colormaps.get_cmap('coolwarm')
    normalizador = colors.LogNorm(vmin=max(1, tamaño_min), vmax=tamaño_max)
    return lambda tamaño: colors.to_hex(colormap(normalizador(tamaño)))

# --- Función de Visualización  ---
def visualizar_comunidades(graph: ig.Graph, comunidades_dict: Dict[int, List[int]], seleccion: Dict[str, List[int]], output_filename: str):
    logging.info(f"Creando mapa de visualización en '{output_filename}'...")
    coords_validas = [(v['lat'], v['lon']) for v in graph.vs if v['lat'] is not None and v.index != 0]
    if not coords_validas: logging.error("No hay nodos con coordenadas para visualizar."); return
    avg_lat = sum(c[0] for c in coords_validas) / len(coords_validas); avg_lon = sum(c[1] for c in coords_validas) / len(coords_validas)
    m = folium.Map(location=[avg_lat, avg_lon], zoom_start=2, tiles="CartoDB positron")
    tamaños = {cid: len(miembros) for cid, miembros in comunidades_dict.items() if miembros}
    if not tamaños: logging.error("Las comunidades están vacías, no se puede generar el mapa."); return
    mapa_color = crear_mapa_de_colores(min(tamaños.values()), max(tamaños.values()))
    ids_a_visualizar = seleccion['grande'] + seleccion['pequena'] + seleccion['random']
    iterator = tqdm(ids_a_visualizar, desc="Creando capas de comunidades")
    
    for com_id in iterator:
        miembros_originales = comunidades_dict.get(com_id)
        if not miembros_originales: continue
        
        tamaño_original = len(miembros_originales)
        color = mapa_color(tamaño_original)
        show_layer = com_id in seleccion['grande'] or com_id in seleccion['pequena']
        nombre_capa = f"Comunidad {com_id} ({tamaño_original} miembros)"
        if com_id in seleccion['grande']: nombre_capa = f"Comunidad Más Grande ({tamaño_original})"
        if com_id in seleccion['pequena']: nombre_capa = f"Comunidad Más Pequeña ({tamaño_original})"
        
        container = MarkerCluster(name=nombre_capa, show=show_layer) if tamaño_original > 200 else folium.FeatureGroup(name=nombre_capa, show=show_layer)
        container.add_to(m)

        nodos_a_dibujar = miembros_originales
        if tamaño_original > UMBRAL_MAX_NODOS_A_DIBUJAR:
            logging.warning(f"Comunidad {com_id} ({tamaño_original} nodos) excede umbral. Mostrando muestra de {UMBRAL_MAX_NODOS_A_DIBUJAR}.")
            nodos_a_dibujar = random.sample(miembros_originales, UMBRAL_MAX_NODOS_A_DIBUJAR)

        nodos_visibles_con_coords = []
        for nodo_id in nodos_a_dibujar:
            v = graph.vs[nodo_id]
            if v['lat'] is not None and v['lon'] is not None:
                nodos_visibles_con_coords.append(nodo_id)
                folium.CircleMarker(location=(v['lat'], v['lon']), radius=4, color=color, fill=True, fill_color=color, fill_opacity=0.7, tooltip=f"Nodo {v.index} (Com. {com_id})").add_to(container)
        
        if tamaño_original <= UMBRAL_MAX_VISUALIZACION_ARISTAS:
            if len(nodos_visibles_con_coords) > 1:
                subgrafo_comunidad = graph.subgraph(nodos_visibles_con_coords)
                num_aristas_internas = len(subgrafo_comunidad.es)
                logging.info(f"Comunidad {com_id} (tamaño {tamaño_original}): Intentando dibujar {num_aristas_internas} aristas internas.")

                for arista in subgrafo_comunidad.es:
                    id_origen = nodos_visibles_con_coords[arista.source]; id_destino = nodos_visibles_con_coords[arista.target]
                    v_origen = graph.vs[id_origen]; v_destino = graph.vs[id_destino]
                    folium.PolyLine(locations=[(v_origen['lat'], v_origen['lon']), (v_destino['lat'], v_destino['lon'])], color=color, weight=1.5, opacity=0.5).add_to(container)
        else:
            logging.warning(f"Omitiendo dibujo de aristas para comunidad {com_id} (tamaño: {tamaño_original} > {UMBRAL_MAX_VISUALIZACION_ARISTAS}).")

    folium.LayerControl(collapsed=False).add_to(m)
    logging.info("Guardando el mapa en el archivo HTML...")
    m.save(output_filename)
    logging.info(f"Mapa guardado correctamente en '{output_filename}'.")


# --- Bloque Principal ---
if __name__ == "__main__":
    GRAFO_PKL_ENTRADA = 'grafo_igraph_paralelizado.pkl'
    MAPA_HTML_SALIDA = 'analisis_comunidades_louvain_manual.html'
    NUM_COMUNIDADES_RANDOM = 50

    mi_grafo = cargar_grafo(GRAFO_PKL_ENTRADA)
    if mi_grafo:
        coms_dict = detectar_comunidades_louvain_manual(mi_grafo)
        if coms_dict:
            comunidades_seleccionadas = analizar_y_seleccionar_comunidades(coms_dict, NUM_COMUNIDADES_RANDOM)
            if comunidades_seleccionadas:
                visualizar_comunidades(mi_grafo, coms_dict, comunidades_seleccionadas, MAPA_HTML_SALIDA)
                
                print("\n" + "="*60)
                print(" ANÁLISIS DE COMUNIDADES (LOUVAIN - IMPLEMENTACIÓN MANUAL)")
                print("="*60)
                print(f"Se encontraron un total de {len(coms_dict)} comunidades.")
                print("\nSe ha generado un mapa interactivo en:", f"'{MAPA_HTML_SALIDA}'")
                print("\nOPTIMIZACIONES DE VISUALIZACIÓN:")
                print(f"  - Muestreo de Nodos: Para comunidades con > {UMBRAL_MAX_NODOS_A_DIBUJAR} miembros,")
                print("    solo se muestra una muestra aleatoria para no colapsar el navegador.")
                print(f"  - Omisión de Aristas: Los caminos solo se dibujan para comunidades")
                print(f"    con <= {UMBRAL_MAX_VISUALIZACION_ARISTAS} miembros.")
                print("\nEl color de cada comunidad indica su tamaño original:")
                print("  - Colores fríos (azul): Comunidades pequeñas.")
                print("  - Colores cálidos (rojo): Comunidades grandes.")
                print("\nPara más detalles, revisa el archivo de log: 'analisis_comunidades_louvain_manual.log'")
                print("="*60)