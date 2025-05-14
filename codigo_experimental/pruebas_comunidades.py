import igraph as ig
import plotly.graph_objects as go
import random
import logging as log
import numpy as np
import plotly.express as px
import time

log.basicConfig(
    level=log.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        log.FileHandler("grafo_comunidades.log", mode='w', encoding='utf-8'),
        log.StreamHandler()
    ]
)

def cargar_grafo(path_pkl):
    """Carga el grafo desde un archivo pickle."""
    try:
        g = ig.Graph.Read(path_pkl, format="pickle")
        log.info(f"Grafo cargado correctamente con {g.vcount()} nodos y {g.ecount()} aristas.")
        return g
    except FileNotFoundError:
        log.error(f"Error: No se encontró el archivo pickle en '{path_pkl}'.")
        return None
    except Exception as e:
        log.error(f"Error al cargar el grafo desde '{path_pkl}': {e}")
        return None

def analisis_eda_basico(grafo):
    """Realiza un análisis exploratorio básico del grafo."""
    if grafo is None:
        return

    num_nodos = grafo.vcount()
    num_aristas = grafo.ecount()
    es_dirigido = grafo.is_directed()
    densidad = grafo.density()

    log.info("\n--- Estadísticas Básicas del Grafo ---")
    log.info(f"Número de nodos: {num_nodos}")
    log.info(f"Número de aristas: {num_aristas}")
    log.info(f"¿Es dirigido?: {es_dirigido}")
    log.info(f"Densidad: {densidad:.4f}")

def visualizar_comunidades_interactivo(grafo, comunidades, num_nodos_muestra=1000):
    """Visualiza las comunidades detectadas en una muestra del grafo con colores."""
    if grafo is None or comunidades is None:
        log.warning("Grafo o comunidades no proporcionados para la visualización.")
        return

    total_nodos = grafo.vcount()
    if total_nodos <= num_nodos_muestra:
        nodos_muestra_indices = list(range(total_nodos))
        log.info(f"Visualizando comunidades en todos los {total_nodos} nodos.")
    else:
        nodos_muestra_indices = random.sample(range(total_nodos), num_nodos_muestra)
        log.info(f"Visualizando comunidades en una muestra de {num_nodos_muestra} de {total_nodos} nodos.")

    subgrafo_original = grafo.subgraph(nodos_muestra_indices)
    comunidades_subgrafo = [comunidades[i] for i in nodos_muestra_indices]

    # Crear un grafo no dirigido solo con aristas mutuas para el layout
    nodos_para_grafo_comunidades = list(range(len(subgrafo_original.vs)))
    edges_para_grafo_comunidades = []
    for i in range(len(subgrafo_original.vs)):
        for j in range(i + 1, len(subgrafo_original.vs)):
            if subgrafo_original.are_adjacent(i, j) and subgrafo_original.are_adjacent(j, i):
                edges_para_grafo_comunidades.append((i, j))

    grafo_comunidades = ig.Graph(n=len(subgrafo_original.vs), edges=edges_para_grafo_comunidades, directed=False)
    try:
        l = grafo_comunidades.layout("fr")
        xs = [l[i][0] for i in range(len(grafo_comunidades.vs))]
        ys = [l[i][1] for i in range(len(grafo_comunidades.vs))]
        zs = [random.random() * 0.1 for _ in range(len(grafo_comunidades.vs))] # Pequeña variación en Z
    except Exception as e:
        log.warning(f"Error al calcular layout para comunidades: {e}. Usando coordenadas aleatorias.")
        xs = [random.random() for _ in range(len(grafo_comunidades.vs))]
        ys = [random.random() for _ in range(len(grafo_comunidades.vs))]
        zs = [random.random() * 0.1 for _ in range(len(grafo_comunidades.vs))]

    edge_x, edge_y, edge_z = [], [], []
    for edge in grafo_comunidades.es:
        source, target = edge.tuple
        edge_x += [xs[source], xs[target], None]
        edge_y += [ys[source], ys[target], None]
        edge_z += [zs[source], zs[target], None]
    
    num_comunidades = len(set(comunidades_subgrafo))
    # Asegura que haya suficientes colores para todas las comunidades
    base_colors = px.colors.qualitative.Set2
    if num_comunidades > len(base_colors):
        # Repite la lista de colores si hay más comunidades que colores
        community_colors = (base_colors * ((num_comunidades // len(base_colors)) + 1))[:num_comunidades]
    else:
        community_colors = base_colors[:num_comunidades]
    color_map = {community: community_colors[i] for i, community in enumerate(sorted(list(set(comunidades_subgrafo))))}
    node_colors = [color_map[c] for c in comunidades_subgrafo]

    fig = go.Figure(data=[go.Scatter3d(
        x=edge_x, y=edge_y, z=edge_z,
        mode='lines',
        line=dict(color='lightgray', width=0.5),
        hoverinfo='none'
    ),
    go.Scatter3d(
        x=xs, y=ys, z=zs,
        mode='markers',
        marker=dict(
            size=6,
            color=node_colors,
            opacity=0.8
        ),
        text=[f'Nodo {nodos_muestra_indices[i] + 1}, Comunidad {comunidades_subgrafo[i]}' for i in range(len(subgrafo_original.vs))],
        hoverinfo='text'
    )])

    fig.update_layout(
        title=f'Visualización de Comunidades (Solo Aristas Mutuas) en una Muestra del Grafo ({len(subgrafo_original.vs)} Nodos)',
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
            xaxis=dict(showbackground=False),
            yaxis=dict(showbackground=False),
            zaxis=dict(showbackground=False)
        ),
        margin=dict(l=0, r=0, b=0, t=40)
    )
    fig.write_html("grafo_comunidades.html")
    log.info(f"La visualización de las comunidades se ha guardado en grafo_comunidades.html")

if __name__ == "__main__":
    grafo_archivo = "grafo_igraph_paralelizado10M.pkl"
    grafo = cargar_grafo(grafo_archivo)

    if grafo:
        analisis_eda_basico(grafo)

        try:
            start_time_mutual = time.time()
            edges_para_comunidades = []
            for edge in grafo.es:
                source = edge.source
                target = edge.target
                if grafo.are_adjacent(target, source):
                    # Añadir la arista como no dirigida (solo una vez para evitar duplicados)
                    if tuple(sorted((source, target))) not in edges_para_comunidades and source != target:
                        edges_para_comunidades.append(tuple(sorted((source, target))))

            grafo_no_dirigido_mutual = ig.Graph(n=grafo.vcount(), edges=list(set(edges_para_comunidades)), directed=False)
            log.info(f"Tiempo para crear grafo de aristas mutuas: {time.time() - start_time_mutual:.2f} segundos")

            start_time_comunidades = time.time()
            if hasattr(grafo_no_dirigido_mutual, "community_fastgreedy"):
                dendrogram = grafo_no_dirigido_mutual.community_fastgreedy()
                comunidades_detectadas = dendrogram.as_clustering().membership
                log.info(f"Tiempo para detección de comunidades: {time.time() - start_time_comunidades:.2f} segundos")
                visualizar_comunidades_interactivo(grafo, comunidades_detectadas, num_nodos_muestra=2000)
            else:
                log.warning("El método community_fastgreedy no está disponible en esta versión de igraph o para este tipo de grafo.")

        except AttributeError:
            log.warning("No se encontraron métodos de detección de comunidades en el grafo.")
        except Exception as e:
            log.error(f"Error al visualizar comunidades: {e}")
            log.info(f"Tiempo para detección de comunidades: {time.time() - start_time_comunidades:.2f} segundos")

            visualizar_comunidades_interactivo(grafo, comunidades_detectadas, num_nodos_muestra=2000)

        except AttributeError:
            log.warning("No se encontraron métodos de detección de comunidades en el grafo.")
        except Exception as e:
            log.error(f"Error al visualizar comunidades: {e}")