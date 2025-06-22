import pickle
import time
import logging
import random
from collections import Counter
import igraph as ig
from typing import Optional, List, Dict
import folium
from folium.plugins import MarkerCluster
import matplotlib
import matplotlib.colors as colors
from tqdm import tqdm


# --- Configuración del Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
handlers=[logging.FileHandler("analisis_comunidades_lpa_final.log", mode='w', encoding='utf-8'),
logging.StreamHandler()])

# --- UMBRALES DE VISUALIZACIÓN ---
# Si una comunidad tiene más nodos que este umbral, no dibujaremos sus aristas.
UMBRAL_MAX_VISUALIZACION_ARISTAS = 2000
# Si una comunidad tiene más nodos que este umbral, solo dibujaremos una muestra aleatoria de ellos.
UMBRAL_MAX_NODOS_A_DIBUJAR = 5000

# --- Función de Carga de Grafo ---
def cargar_grafo(grafo_path: str) -> Optional[ig.Graph]:
    logging.info(f"Cargando el grafo desde '{grafo_path}'..."); start_time = time.time()
    try:
        with open(grafo_path, 'rb') as f: g = pickle.load(f)
        end_time = time.time(); logging.info(f"Grafo cargado en {end_time - start_time:.2f} segundos. {g.summary()}")
        return g
    except Exception as e:
        logging.error(f"Error crítico al cargar el grafo: {e}"); return None

# --- Implementación del LPA ---
def detectar_comunidades_lpa(graph: ig.Graph, max_iter: int = 10) -> Dict[int, List[int]]:
    logging.info(f"Iniciando detección de comunidades con algoritmo propio (LPA) con {max_iter} iteraciones...")
    start_time = time.time()
    
    labels = {v.index: v.index for v in graph.vs}
    
    for i in range(max_iter):
        logging.info(f"LPA - Iteración {i + 1}/{max_iter}...");
        changes_count = 0
        
        nodes_to_process = list(range(graph.vcount())); random.shuffle(nodes_to_process)
        iterator = tqdm(nodes_to_process, desc=f"Iteración {i+1}")
        
        for node_id in iterator:
            neighbors = graph.neighbors(node_id, mode='all')
            if not neighbors: continue
            
            label_counts = Counter(labels[n] for n in neighbors)
            max_freq = max(label_counts.values())
            
            most_frequent_labels = [label for label, count in label_counts.items() if count == max_freq]
            new_label = random.choice(most_frequent_labels)
            
            if labels[node_id] != new_label:
                labels[node_id] = new_label
                changes_count += 1
                
        logging.info(f"Fin de la iteración {i + 1}. Hubo {changes_count} cambios de etiqueta.")
        if changes_count == 0:
            logging.info("Convergencia alcanzada antes del máximo de iteraciones."); break
            
    if i == max_iter - 1 and changes_count > 0:
        logging.warning("Se alcanzó el máximo de iteraciones sin convergencia completa.")

    comunidades = {};
    start_node = 1 if graph.vs[0].attributes().get('name') == 'dummy_root' else 0
    for node, label in labels.items():
        if node < start_node: continue
        if label not in comunidades: comunidades[label] = []
        comunidades[label].append(node)
        
    end_time = time.time()
    logging.info(f"LPA completado en {end_time - start_time:.2f} s. Se encontraron {len(comunidades)} comunidades.")
    return comunidades

# --- Función de Análisis ---
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

# --- Función de Colores ---
def crear_mapa_de_colores(tamaño_min: int, tamaño_max: int):
    colormap = matplotlib.colormaps.get_cmap('coolwarm')
    normalizador = colors.LogNorm(vmin=max(1, tamaño_min), vmax=tamaño_max)
    return lambda tamaño: colors.to_hex(colormap(normalizador(tamaño)))

# --- Función de Visualización (con todas las optimizaciones) ---
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

        # Muestreo de nodos para comunidades gigantes
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
        
        # Dibujo de aristas solo para comunidades pequeñas
        if tamaño_original <= UMBRAL_MAX_VISUALIZACION_ARISTAS:
            if len(nodos_visibles_con_coords) > 1:
                subgrafo_comunidad = graph.subgraph(nodos_visibles_con_coords)
                
                # Log de diagnóstico
                num_aristas_internas = len(subgrafo_comunidad.es)
                logging.info(f"Comunidad {com_id} (tamaño {tamaño_original}): "
                             f"Intentando dibujar conexiones. Se encontraron {num_aristas_internas} aristas internas.")

                for arista in subgrafo_comunidad.es:
                    id_origen = nodos_visibles_con_coords[arista.source]; id_destino = nodos_visibles_con_coords[arista.target]
                    v_origen = graph.vs[id_origen]; v_destino = graph.vs[id_destino]
                    folium.PolyLine(locations=[(v_origen['lat'], v_origen['lon']), (v_destino['lat'], v_destino['lon'])], color=color, weight=1.5, opacity=0.5).add_to(container)
        else:
            logging.warning(f"Omitiendo dibujo de aristas para comunidad {com_id} (tamaño: {tamaño_original} > {UMBRAL_MAX_VISUALIZACION_ARISTAS}).")

    folium.LayerControl(collapsed=False).add_to(m)
    logging.info("Guardando el mapa en el archivo HTML. Este paso puede tardar...")
    m.save(output_filename)
    logging.info(f"Mapa guardado correctamente en '{output_filename}'.")

# --- Bloque Principal ---
if __name__ == "__main__":
    GRAFO_PKL_ENTRADA = 'grafo_igraph_paralelizado.pkl'
    MAPA_HTML_SALIDA = 'analisis_comunidades_lpa_final.html'
    NUM_COMUNIDADES_RANDOM = 20
    MAX_ITER_LPA = 10

    mi_grafo = cargar_grafo(GRAFO_PKL_ENTRADA)
    if mi_grafo:
        coms_dict = detectar_comunidades_lpa(mi_grafo, max_iter=MAX_ITER_LPA)
        if coms_dict:
            comunidades_seleccionadas = analizar_y_seleccionar_comunidades(coms_dict, NUM_COMUNIDADES_RANDOM)
            if comunidades_seleccionadas:
                visualizar_comunidades(mi_grafo, coms_dict, comunidades_seleccionadas, MAPA_HTML_SALIDA)
                
                print("\n" + "="*60)
                print(" ANÁLISIS DE COMUNIDADES (LPA - VERSIÓN FINAL OPTIMIZADA)")
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
                print("\nPara más detalles, revisa el archivo de log: 'analisis_comunidades_lpa_final.log'")
                print("="*60)