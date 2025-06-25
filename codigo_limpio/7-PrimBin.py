import pickle
import logging
import igraph as ig
from typing import Optional, Tuple, List, Dict
import polars as pl
import graphistry
import random
import time
import webbrowser
from tqdm import tqdm
from math import radians, sin, cos, sqrt, atan2
from collections import defaultdict

# --- PARÁMETROS DE CONFIGURACIÓN ---
MAX_NODOS_PARA_ANALISIS = 200000 
MAX_ARISTAS_MST_PARA_GRAPHISTRY = 75000
GRAPHISTRY_USERNAME = "MiquelAngFl"
GRAPHISTRY_PASSWORD = "sxtXy3ZG.DAhdsg"

# --- Configuración del Logging ---
logging.basicConfig(
    level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("analisis_prim_louvain_graphistry.log", mode='w', encoding='utf-8'), logging.StreamHandler()]
)

# --- Implementación Manual del Heap Binario (Min-Heap) ---
class MinHeap:
    def __init__(self): self.heap, self.position_map = [], {}
    def is_empty(self) -> bool: return len(self.heap) == 0
    def _parent(self, i: int) -> int: return (i - 1) // 2
    def _left_child(self, i: int) -> int: return 2 * i + 1
    def _right_child(self, i: int) -> int: return 2 * i + 2
    def _swap(self, i: int, j: int):
        self.position_map[self.heap[i][1]], self.position_map[self.heap[j][1]] = j, i
        self.heap[i], self.heap[j] = self.heap[j], self.heap[i]
    def _bubble_up(self, i: int):
        parent_idx = self._parent(i)
        while i > 0 and self.heap[i][0] < self.heap[parent_idx][0]:
            self._swap(i, parent_idx); i = parent_idx; parent_idx = self._parent(i)
    def _bubble_down(self, i: int):
        min_index = i
        while True:
            left_idx, right_idx = self._left_child(i), self._right_child(i)
            if left_idx < len(self.heap) and self.heap[left_idx][0] < self.heap[min_index][0]: min_index = left_idx
            if right_idx < len(self.heap) and self.heap[right_idx][0] < self.heap[min_index][0]: min_index = right_idx
            if i == min_index: break
            self._swap(i, min_index); i = min_index
    def push(self, weight: float, node_id: int):
        self.heap.append((weight, node_id)); self.position_map[node_id] = len(self.heap) - 1
        self._bubble_up(len(self.heap) - 1)
    def pop(self) -> Tuple[float, int]:
        if self.is_empty(): raise IndexError("pop from an empty heap")
        min_element = self.heap[0]; last_element = self.heap.pop(); del self.position_map[min_element[1]]
        if not self.is_empty():
            self.heap[0] = last_element; self.position_map[last_element[1]] = 0
            self._bubble_down(0)
        return min_element
    def update(self, node_id: int, new_weight: float):
        if node_id not in self.position_map: return
        idx = self.position_map[node_id]
        if new_weight < self.heap[idx][0]:
            self.heap[idx] = (new_weight, node_id); self._bubble_up(idx)

# --- Funciones de Carga y Preparación ---
def cargar_grafo(grafo_path: str) -> Optional[ig.Graph]:
    logging.info(f"Cargando el grafo desde '{grafo_path}'...")
    start_time = time.time();
    try:
        with open(grafo_path, 'rb') as f: g = pickle.load(f)
        logging.info(f"Grafo cargado en {time.time() - start_time:.2f} segundos."); return g
    except Exception as e:
        logging.error(f"Error al cargar el grafo: {e}"); return None

def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371; dlat, dlon = radians(lat2 - lat1), radians(lon2 - lon1)
    a = sin(dlat / 2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2)**2
    a = min(1.0, a); c = 2 * atan2(sqrt(a), sqrt(1 - a)); return R * c

def preparar_pesos_geograficos(graph: ig.Graph):
    if 'weight' in graph.es.attributes(): return
    logging.info("Calculando pesos geográficos (Haversine)...")
    pesos = []
    for edge in tqdm(graph.es, desc="Calculando pesos"):
        source_v = graph.vs[edge.source]; target_v = graph.vs[edge.target]
        if source_v['lat'] is not None and target_v['lat'] is not None:
            pesos.append(haversine_distance(source_v['lat'], source_v['lon'], target_v['lat'], target_v['lon']))
        else:
            pesos.append(float('inf'))
    graph.es['weight'] = pesos
    logging.info("Pesos geográficos asignados.")

def encontrar_nodo_inicio_valido(graph: ig.Graph, nodes_in_partition: List[int]) -> Optional[int]:
    random.shuffle(nodes_in_partition)
    for node_id in nodes_in_partition:
        v = graph.vs[node_id]
        if v['lat'] is not None and graph.degree(node_id) > 0:
            return node_id
    return None

def _calcular_delta_modularity(node: int, target_community: int, node_degree: float,
                               links_to_target_community: float, total_community_degree: float,
                               total_edge_weight: float) -> float:
    if total_edge_weight == 0: return 0
    term1 = links_to_target_community / total_edge_weight
    term2 = (total_community_degree * node_degree) / (2 * (total_edge_weight ** 2))
    return term1 - term2


def _louvain_phase1(graph: ig.Graph, total_edge_weight: float) -> Tuple[Dict[int, int], bool]:
    communities = {v.index: v.index for v in graph.vs}
    all_strengths = graph.strength(weights='weight' if 'weight' in graph.es.attributes() else None)
    node_degrees = {i: strength for i, strength in enumerate(all_strengths)}
    community_degrees = node_degrees.copy()
    improved = False
    nodes = list(range(graph.vcount()))
    random.shuffle(nodes)
    for node in tqdm(nodes, desc="  Louvain Fase 1", leave=False):
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
            gain = _calcular_delta_modularity(node, community, node_degrees[node], links_weight, community_degrees[community], total_edge_weight)
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
    new_nodes = list(set(communities.values()))
    new_node_map = {old_id: new_id for new_id, old_id in enumerate(new_nodes)}
    new_graph = ig.Graph(directed=False)
    new_graph.add_vertices(len(new_nodes))
    edge_weights = defaultdict(float)
    for edge in graph.es:
        source, target = edge.tuple
        c_source, c_target = communities[source], communities[target]
        weight = edge['weight'] if 'weight' in graph.es.attributes() else 1.0
        if c_source != c_target:
            new_source, new_target = new_node_map[c_source], new_node_map[c_target]
            edge_weights[tuple(sorted((new_source, new_target)))] += weight
    new_graph.add_edges(edge_weights.keys())
    new_graph.es['weight'] = list(edge_weights.values())
    return new_graph

def detectar_comunidades_louvain_manual(graph: ig.Graph) -> Dict[int, List[int]]:
    logging.info("Iniciando detección de comunidades con Louvain manual...")
    start_time = time.time()
    current_graph = graph.as_undirected(mode='collapse', combine_edges=dict(weight="sum")) if graph.is_directed() else graph.copy()
    membership = {v.index: v.index for v in graph.vs}
    pass_num = 1
    while True:
        logging.info(f"--- Louvain - Pass {pass_num} ---")
        total_edge_weight = sum(current_graph.es['weight']) if 'weight' in current_graph.es.attributes() else current_graph.ecount()
        if total_edge_weight == 0: break
        communities, improved = _louvain_phase1(current_graph, total_edge_weight)
        if not improved: break
        partition_map = communities
        for node_idx in range(graph.vcount()):
             membership[node_idx] = partition_map.get(membership[node_idx], membership[node_idx])
        current_graph = _louvain_phase2(current_graph, communities)
        if current_graph.vcount() <= 1 or current_graph.ecount() == 0: break
        pass_num += 1
    final_comunidades = defaultdict(list)
    for node, comm in membership.items(): final_comunidades[comm].append(node)
    logging.info(f"Louvain completado en {time.time() - start_time:.2f} s. Se encontraron {len(final_comunidades)} comunidades.")
    return dict(final_comunidades)

def prim_mst_forest_manual(graph: ig.Graph) -> Tuple[List[Tuple[int, int]], float]:
    logging.info("Calculando el Bosque de Expansión Mínima (Prim manual) por comunidad (Louvain)...")
    start_time = time.time()
    
    # Paso 1: Detectar comunidades con Louvain
    comunidades_dict = detectar_comunidades_louvain_manual(graph)

    parent, in_mst = {v.index: None for v in graph.vs}, {v.index: False for v in graph.vs}
    forest_edges, total_forest_weight = [], 0
    
    logging.info(f"Grafo tiene {len(comunidades_dict)} comunidades. Calculando un MST para cada una.")
    # Itera sobre las comunidades en lugar de los componentes
    for community_nodes in tqdm(comunidades_dict.values(), desc="Procesando Comunidades para MST"):
        if len(community_nodes) <= 1: continue
        
        start_node = encontrar_nodo_inicio_valido(graph, community_nodes)
        if start_node is None: 
            logging.warning(f"No se encontró un nodo de inicio válido para una comunidad de tamaño {len(community_nodes)}")
            continue
            
        # El algoritmo de Prim se aplica al subgrafo inducido por la comunidad
        key = {node: float('inf') for node in community_nodes}
        key[start_node] = 0
        pq = MinHeap()
        for node in community_nodes: pq.push(key[node], node)
        
        while not pq.is_empty():
            weight, u = pq.pop()
            if in_mst[u]: continue
            
            in_mst[u] = True
            if weight != float('inf'): total_forest_weight += weight
            if parent[u] is not None: forest_edges.append((parent[u], u))
            
            for neighbor_v in graph.neighbors(u, mode='all'):
                # Nos aseguramos que el vecino pertenezca a la misma comunidad
                if neighbor_v in key: 
                    edge = graph.es[graph.get_eid(u, neighbor_v, directed=False, error=False)]
                    if edge is None: continue # Arista podría no existir en grafo original si es dirigido
                    
                    edge_weight = edge['weight']
                    if not in_mst[neighbor_v] and edge_weight < key[neighbor_v]:
                        key[neighbor_v] = edge_weight
                        parent[neighbor_v] = u
                        pq.update(neighbor_v, edge_weight)

    logging.info(f"Algoritmo de Prim sobre comunidades completado en {time.time() - start_time:.4f} s.")
    return forest_edges, total_forest_weight

# --- Función de Visualización del MST con Graphistry ---
def visualizar_mst_con_graphistry(graph: ig.Graph, mst_edges: List[Tuple[int, int]]):
    if not mst_edges:
        logging.error("La lista de aristas del MST está vacía."); return

    logging.info("Preparando el MST para la visualización en Graphistry...")
    aristas_a_dibujar = mst_edges
    if len(mst_edges) > MAX_ARISTAS_MST_PARA_GRAPHISTRY:
        logging.warning(f"El MSF ({len(mst_edges)} aristas) supera el límite de {MAX_ARISTAS_MST_PARA_GRAPHISTRY}. Tomando muestra.")
        aristas_a_dibujar = random.sample(mst_edges, MAX_ARISTAS_MST_PARA_GRAPHISTRY)
    
    df_edges_pl = pl.DataFrame(aristas_a_dibujar, schema={'src_orig': pl.Int64, 'dst_orig': pl.Int64}, orient="row")
    
    nodos_unicos = set(df_edges_pl['src_orig'].to_list() + df_edges_pl['dst_orig'].to_list())
    logging.info(f"La muestra del MSF contiene {len(nodos_unicos)} nodos únicos.")

    nodos_data = {
        'node_orig': list(nodos_unicos),
        'lat': [graph.vs[i]['lat'] for i in nodos_unicos],
        'lon': [graph.vs[i]['lon'] for i in nodos_unicos],
        'degree_original': [graph.degree(i) for i in nodos_unicos]
    }
    df_nodes_pl = pl.DataFrame(nodos_data)
    id_map = {original_id: new_id for new_id, original_id in enumerate(nodos_unicos)}
    
    src_remapped = df_edges_pl['src_orig'].replace(id_map)
    dst_remapped = df_edges_pl['dst_orig'].replace(id_map)
    
    df_edges_pl = df_edges_pl.with_columns([
        src_remapped.alias('src'),
        dst_remapped.alias('dst')
    ]).drop(['src_orig', 'dst_orig'])
    
    node_remapped = df_nodes_pl['node_orig'].replace(id_map)
    df_nodes_pl = df_nodes_pl.with_columns(
        node_remapped.alias('node')
    )
    
    try:
        logging.info("Convirtiendo a Pandas y generando la visualización...")
        df_nodes_pd = df_nodes_pl.to_pandas()
        df_edges_pd = df_edges_pl.to_pandas()
        
        g_viz = graphistry.bind(source='src', destination='dst', node='node').plot(df_edges_pd, df_nodes_pd)
        if isinstance(g_viz, str): url = g_viz
        else: url = g_viz.url

        logging.info(f"Visualización del MSF creada. Ábrela en tu navegador: {url}")
        print(f"\n--> URL de la visualización del MSF (por comunidades): {url}\n")
        webbrowser.open(url)
    except Exception as e:
        logging.error(f"Ocurrió un error al generar la visualización de Graphistry: {e}")

if __name__ == "__main__":
    GRAFO_PKL_ENTRADA = 'grafo_igraph_paralelizado.pkl'
    
    try:
        graphistry.register(api=3, username=GRAPHISTRY_USERNAME, password=GRAPHISTRY_PASSWORD)
    except Exception as e:
        logging.error(f"No se pudo registrar en Graphistry: {e}"); exit()

    grafo_completo = cargar_grafo(GRAFO_PKL_ENTRADA)
    if not grafo_completo: exit()

    if grafo_completo.vcount() > MAX_NODOS_PARA_ANALISIS:
        logging.warning(f"El grafo completo ({grafo_completo.vcount()} nodos) es demasiado grande. Creando subgrafo de muestra...")
        nodos_muestra_ids = random.sample(range(grafo_completo.vcount()), MAX_NODOS_PARA_ANALISIS)
        grafo_para_analisis = grafo_completo.subgraph(nodos_muestra_ids)
        logging.info(f"Subgrafo creado: {grafo_para_analisis.summary()}")
    else:
        grafo_para_analisis = grafo_completo

    preparar_pesos_geograficos(grafo_para_analisis)

    if grafo_para_analisis.is_directed():
        grafo_no_dirigido = grafo_para_analisis.as_undirected(mode='collapse', combine_edges=dict(weight="min"))
    else:
        grafo_no_dirigido = grafo_para_analisis
    
    mst_aristas, _ = prim_mst_forest_manual(grafo_no_dirigido)
    
    if mst_aristas:
        print("\n" + "="*70)
        print("  VISUALIZACIÓN DEL MSF (POR COMUNIDADES LOUVAIN) CON GRAPHISTRY")
        print("="*70)
        # Se pasa el grafo completo para poder buscar los atributos originales de los nodos
        visualizar_mst_con_graphistry(grafo_completo, mst_aristas)
        print("="*70)
    else:
        logging.error("No se pudo calcular el MSF basado en comunidades para el subgrafo.")