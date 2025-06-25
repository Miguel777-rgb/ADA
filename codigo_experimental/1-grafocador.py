import igraph as ig
import polars as pl  # Usamos Polars en lugar de Pandas
import graphistry
import random
import logging as log
import os
import sys

# --- Configuración del Logging ---
log.basicConfig(
    level=log.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        log.FileHandler("grafo_grafico_graphistry_polars.log", mode='w', encoding='utf-8'),
        log.StreamHandler()
    ]
)


def graficar_con_graphistry(path_pkl, num_nodos_muestra=15000):

    log.info(f"Intentando cargar el grafo desde: '{path_pkl}' para visualizar con Graphistry.")
    try:
        g = ig.Graph.Read(path_pkl, format="pickle")
        total_nodos = g.vcount()
        total_aristas = g.ecount()
        log.info(f"Grafo cargado correctamente con {total_nodos} nodos y {total_aristas} aristas.")
    except FileNotFoundError:
        log.error(f"Error: No se encontró el archivo pickle en '{path_pkl}'.")
        return
    except Exception as e:
        log.error(f"Error al cargar el grafo desde '{path_pkl}': {e}")
        return

    if total_nodos == 0:
        log.warning("El grafo está vacío, no se puede graficar.")
        return

    # --- Muestreo de Nodos ---
    if num_nodos_muestra >= total_nodos:
        indices_muestra = list(range(total_nodos))
        log.info(f"Graficando todos los {total_nodos} nodos.")
    else:
        indices_muestra = sorted(random.sample(range(total_nodos), num_nodos_muestra))
        log.info(f"Seleccionando aleatoriamente {num_nodos_muestra} nodos para la muestra.")

    subg = g.subgraph(indices_muestra)
    log.info(f"Subgrafo creado con {subg.vcount()} nodos y {subg.ecount()} aristas.")

    # --- Preparación de DataFrames de Polars ---
    
    # 1. Crear el DataFrame de Nodos con Polars
    log.info("Creando DataFrame de Polars para los nodos con atributos (lat, lon)...")
    node_data = []
    mapa_idx_subg_a_original = {subg.vs[i].index: indices_muestra[i] for i in range(subg.vcount())}

    for v in subg.vs:
        try:
            original_id = mapa_idx_subg_a_original[v.index]
            node_data.append({
                'node_id': original_id + 1,
                'lat': v['lat'],
                'lon': v['lon']
            })
        except KeyError:
            log.warning(f"Nodo con índice original {original_id+1} no tiene atributos 'lat' o 'lon'. Se omitirán.")
            
    if not node_data:
        log.error("No se pudieron extraer datos de ningún nodo. Abortando visualización.")
        return
        
    nodes_df = pl.DataFrame(node_data)
    log.info(f"DataFrame de Polars para nodos creado con {nodes_df.height} filas.")

    # 2. Crear el DataFrame de Aristas con Polars
    log.info("Creando DataFrame de Polars para las aristas...")
    edges = subg.get_edgelist()
    edge_list = []
    for src_subg_idx, dst_subg_idx in edges:
        original_src_id = mapa_idx_subg_a_original[src_subg_idx] + 1
        original_dst_id = mapa_idx_subg_a_original[dst_subg_idx] + 1
        edge_list.append({'source': original_src_id, 'destination': original_dst_id})
    
    edges_df = pl.DataFrame(edge_list)
    log.info(f"DataFrame de Polars para aristas creado con {edges_df.height} filas.")
    
    # --- Visualización con Graphistry ---
    log.info("Generando la visualización con Graphistry...")
    
    node_id_col = 'node_id'
    
    g_viz = graphistry.edges(edges_df, 'source', 'destination').nodes(nodes_df, node_id_col)
    
    g_viz = g_viz.encode_point_x('lon').encode_point_y('lat')
    
    try:
        url = g_viz.plot(render=False, plot_title=f'Muestra de {subg.vcount()} Nodos del Grafo (Polars)')
        log.info("Visualización generada exitosamente.")
        log.info(f"Ábrela en tu navegador: {url}")
    except Exception as e:
        log.error(f"Error al generar la gráfica de Graphistry: {e}")

if __name__ == "__main__":
    # --- Autenticación de Graphistry usando Variables de Entorno ---
    try:
        graphistry.register(api=3, username="MiquelAngFl", password="sxtXy3ZG.DAhdsg")
        log.info("Autenticación con Graphistry mediante variables de entorno fue exitosa.")
    except Exception as e:
        log.error(f"Fallo en la autenticación con Graphistry: {e}")
        log.error("Verifica que tus credenciales en las variables de entorno sean correctas.")
        sys.exit(1)

    # --- Ejecución del Proceso de Gráfica ---
    grafo_archivo = "grafo_igraph_paralelizado.pkl"
    num_nodos_a_mostrar = 20000 
    
    graficar_con_graphistry(grafo_archivo, num_nodos_a_mostrar)