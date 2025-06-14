# analisis_comunidades_final_v2.py
"""
Script final para la detección y visualización de comunidades en una red a gran escala.
VERSIÓN 2: Corrige el NameError 'matplotlib' is not defined importando el
paquete completo.
"""

import pickle
import time
import logging
import random
from collections import Counter
import igraph as ig
from typing import Optional, List, Dict

# --- Manejo de dependencias opcionales (CORREGIDO) ---
try:
    import folium
    from folium.plugins import MarkerCluster
    # CORRECCIÓN: Importar el paquete principal para tener acceso a .colormaps
    import matplotlib
    import matplotlib.colors as colors
    LIBS_DISPONIBLES = True
except ImportError:
    LIBS_DISPONIBLES = False

try:
    from tqdm import tqdm
    TQDM_DISPONIBLE = True
except ImportError:
    TQDM_DISPONIBLE = False

# --- Configuración del Logging (sin cambios) ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler("analisis_comunidades_final.log", mode='w', encoding='utf-8'),
                              logging.StreamHandler()])

# --- Función de Carga de Grafo (sin cambios) ---
def cargar_grafo(grafo_path: str) -> Optional[ig.Graph]:
    logging.info(f"Cargando el grafo desde '{grafo_path}'..."); start_time = time.time()
    try:
        with open(grafo_path, 'rb') as f: g = pickle.load(f)
        end_time = time.time(); logging.info(f"Grafo cargado en {end_time - start_time:.2f} segundos. {g.summary()}")
        return g
    except Exception as e:
        logging.error(f"Error crítico al cargar el grafo: {e}"); return None

# --- Implementación Propia del LPA (sin cambios) ---
def detectar_comunidades_lpa(graph: ig.Graph, max_iter: int = 20) -> Dict[int, List[int]]:
    logging.info("Iniciando detección de comunidades con algoritmo propio (LPA)..."); start_time = time.time()
    labels = {v.index: v.index for v in graph.vs}; num_nodos = graph.vcount()
    for i in range(max_iter):
        logging.info(f"LPA - Iteración {i + 1}/{max_iter}..."); changes_count = 0
        nodes_to_process = list(range(1, num_nodos)); random.shuffle(nodes_to_process)
        iterator = tqdm(nodes_to_process, desc=f"Iteración {i+1}") if TQDM_DISPONIBLE else nodes_to_process
        for node_id in iterator:
            neighbors = graph.neighbors(node_id, mode='all')
            if not neighbors: continue
            label_counts = Counter([labels[n] for n in neighbors])
            max_freq = max(label_counts.values())
            most_frequent_labels = [label for label, count in label_counts.items() if count == max_freq]
            new_label = random.choice(most_frequent_labels)
            if labels[node_id] != new_label: labels[node_id] = new_label; changes_count += 1
        logging.info(f"Fin de la iteración {i + 1}. Hubo {changes_count} cambios de etiqueta.")
        if changes_count == 0: logging.info("Convergencia alcanzada."); break
    if i == max_iter - 1: logging.warning("Se alcanzó el máx. de iteraciones.")
    comunidades = {};
    for node, label in labels.items():
        if node == 0: continue
        if label not in comunidades: comunidades[label] = []
        comunidades[label].append(node)
    end_time = time.time(); logging.info(f"LPA completado en {end_time - start_time:.2f} s. Se encontraron {len(comunidades)} comunidades.")
    return comunidades

# --- Funciones de Análisis (sin cambios) ---
def analizar_y_seleccionar_comunidades(comunidades_dict: Dict[int, List[int]], num_random: int = 20) -> Dict[str, List[int]]:
    logging.info("Analizando y seleccionando comunidades para visualización...")
    if len(comunidades_dict) < 3: logging.warning("No hay suficientes comunidades."); return {}
    comunidades_con_tamaño = [(cid, len(miembros)) for cid, miembros in comunidades_dict.items()]
    id_grande, tamaño_grande = max(comunidades_con_tamaño, key=lambda item: item[1])
    com_mayores_a_uno = [c for c in comunidades_con_tamaño if c[1] > 1]
    id_pequeña, tamaño_pequeña = min(com_mayores_a_uno, key=lambda item: item[1]) if com_mayores_a_uno else min(comunidades_con_tamaño, key=lambda item: item[1])
    ids_extremos = {id_grande, id_pequeña}
    posibles_ids_random = [cid for cid, size in comunidades_con_tamaño if cid not in ids_extremos and 5 < size < 1000]
    ids_random = random.sample(posibles_ids_random, min(num_random, len(posibles_ids_random))) if posibles_ids_random else []
    seleccion = {'grande': [id_grande], 'pequena': [id_pequeña], 'random': ids_random}
    logging.info(f"Comunidad más grande (ID {id_grande}): {tamaño_grande} miembros.")
    logging.info(f"Comunidad más pequeña (>1 miembro) (ID {id_pequeña}): {tamaño_pequeña} miembros.")
    logging.info(f"Se seleccionaron {len(ids_random)} comunidades aleatorias.")
    return seleccion

# --- Función de Colores (CORREGIDA) ---
def crear_mapa_de_colores(tamaño_min: int, tamaño_max: int):
    """Crea una función que mapea un tamaño a un color de un degradado."""
    # CORRECCIÓN: Usar matplotlib.colormaps en lugar de cm
    colormap = matplotlib.colormaps.get_cmap('coolwarm')
    normalizador = colors.LogNorm(vmin=max(1, tamaño_min), vmax=tamaño_max)
    return lambda tamaño: colors.to_hex(colormap(normalizador(tamaño)))

# --- Función de Visualización (sin cambios en la lógica, solo depende de la corrección anterior) ---
def visualizar_comunidades(graph: ig.Graph, comunidades_dict: Dict[int, List[int]], seleccion: Dict[str, List[int]], output_filename: str):
    if not LIBS_DISPONIBLES: return
    logging.info(f"Creando mapa de visualización en '{output_filename}'...")
    coords_validas = [(v['lat'], v['lon']) for v in graph.vs if v['lat'] is not None and v.index != 0]
    if not coords_validas: logging.error("No hay nodos con coordenadas."); return
    avg_lat = sum(c[0] for c in coords_validas) / len(coords_validas)
    avg_lon = sum(c[1] for c in coords_validas) / len(coords_validas)
    m = folium.Map(location=[avg_lat, avg_lon], zoom_start=5, tiles="CartoDB positron")
    tamaños = {cid: len(miembros) for cid, miembros in comunidades_dict.items()}
    mapa_color = crear_mapa_de_colores(min(tamaños.values()), max(tamaños.values()))
    ids_a_visualizar = seleccion['grande'] + seleccion['pequena'] + seleccion['random']
    iterator = tqdm(ids_a_visualizar, desc="Creando capas de comunidades") if TQDM_DISPONIBLE else ids_a_visualizar
    for com_id in iterator:
        miembros = comunidades_dict[com_id]
        tamaño = len(miembros)
        color = mapa_color(tamaño)
        show_layer = com_id in seleccion['grande'] or com_id in seleccion['pequena']
        nombre_capa = f"Comunidad {com_id} ({tamaño} miembros)"
        if com_id in seleccion['grande']: nombre_capa = f"Comunidad Más Grande ({tamaño})"
        if com_id in seleccion['pequena']: nombre_capa = f"Comunidad Más Pequeña ({tamaño})"
        if tamaño > 200: container = MarkerCluster(name=nombre_capa, show=show_layer)
        else: container = folium.FeatureGroup(name=nombre_capa, show=show_layer)
        container.add_to(m)
        for nodo_id in miembros:
            v = graph.vs[nodo_id]
            if v['lat'] is not None:
                folium.CircleMarker(location=(v['lat'], v['lon']), radius=4, color=color, fill=True,
                                    fill_color=color, fill_opacity=0.7,
                                    tooltip=f"Nodo {nodo_id} (Comunidad {com_id})"
                                   ).add_to(container)
    folium.LayerControl(collapsed=False).add_to(m)
    m.save(output_filename)
    logging.info(f"Mapa guardado correctamente en '{output_filename}'.")

# --- Bloque Principal (sin cambios) ---
if __name__ == "__main__":
    GRAFO_PKL_ENTRADA = 'grafo_igraph_paralelizado.pkl'
    MAPA_HTML_SALIDA = 'analisis_comunidades_final.html'
    NUM_COMUNIDADES_RANDOM = 20
    MAX_ITER_LPA = 1
    if not LIBS_DISPONIBLES: logging.error("Librerías no encontradas: pip install folium matplotlib tqdm")
    mi_grafo = cargar_grafo(GRAFO_PKL_ENTRADA)
    if mi_grafo:
        coms_dict = detectar_comunidades_lpa(mi_grafo, max_iter=MAX_ITER_LPA)
        if coms_dict:
            comunidades_seleccionadas = analizar_y_seleccionar_comunidades(coms_dict, NUM_COMUNIDADES_RANDOM)
            if comunidades_seleccionadas:
                visualizar_comunidades(mi_grafo, coms_dict, comunidades_seleccionadas, MAPA_HTML_SALIDA)
                print("\n" + "="*60)
                print("      ANÁLISIS DE COMUNIDADES (LPA PROPIO - V2 CORREGIDA)")
                print("="*60)
                print(f"Se encontraron un total de {len(coms_dict)} comunidades.")
                print("\nSe ha generado un mapa interactivo en:", f"'{MAPA_HTML_SALIDA}'")
                print("El mapa visualiza las siguientes comunidades (usa el control de capas):")
                print("  - La comunidad más grande.")
                print("  - La comunidad más pequeña (con más de 1 miembro).")
                print(f"  - Una muestra de {len(comunidades_seleccionadas.get('random', []))} comunidades de tamaño intermedio.")
                print("\nEl color de cada comunidad indica su tamaño:")
                print("  - Colores fríos (azul): Comunidades pequeñas.")
                print("  - Colores cálidos (rojo): Comunidades grandes.")
                print("="*60)