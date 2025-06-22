import igraph as ig
import plotly.graph_objects as go
import random
import pickle
import logging as log

log.basicConfig(
    level=log.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        log.FileHandler("grafo_grafico_muestra.log", mode='w', encoding='utf-8'),
        log.StreamHandler()
    ]
)

def graficar_grafo_3d_muestra(path_pkl, num_nodos_muestra=15000):
    """
    Carga un grafo desde un archivo pickle y grafica una muestra aleatoria de nodos en 3D usando Plotly.
    Los ejes X, Y y Z corresponden a Latitud, Longitud e Índice del Nodo (base 1).

    Args:
        path_pkl (str): Ruta al archivo pickle del grafo de igraph.
        num_nodos_muestra (int): Número de nodos aleatorios a mostrar.
    """
    log.info(f"Intentando cargar el grafo desde: '{path_pkl}' para una muestra 3D de {num_nodos_muestra} nodos.")
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

    if num_nodos_muestra >= total_nodos:
        nodos_muestra = list(range(total_nodos))
        log.info(f"El número de nodos a mostrar ({num_nodos_muestra}) es mayor o igual al total de nodos ({total_nodos}). Graficando todos los nodos.")
    else:
        nodos_muestra_indices = random.sample(range(total_nodos), num_nodos_muestra)
        nodos_muestra = sorted(nodos_muestra_indices)
        log.info(f"Seleccionando aleatoriamente {num_nodos_muestra} nodos para la muestra 3D.")

    subg = g.subgraph(nodos_muestra)
    log.info(f"Subgrafo creado con {subg.vcount()} nodos para la muestra 3D.")

    # Obtener coordenadas de latitud, longitud e índice para la muestra
    xs, ys, zs = [], [], []
    nodos_para_grafico = []
    ubicaciones_encontradas = 0
    original_indices_subgrafo = sorted([g.vs[i].index for i in nodos_muestra]) # Obtener los índices originales

    for i, original_index in enumerate(original_indices_subgrafo):
        try:
            v = subg.vs[i]
            lat = v['lat']
            lon = v['lon']
            xs.append(lat)  # Eje X: Latitud
            ys.append(lon)  # Eje Y: Longitud
            zs.append(original_index + 1)  # Eje Z: Índice del nodo original (base 1)
            nodos_para_grafico.append(original_index)
            ubicaciones_encontradas += 1
        except KeyError:
            log.warning(f"Nodo con índice original {original_index + 1} no tiene atributos 'lat' o 'lon'. Se omitirá del gráfico 3D.")

    if not xs or not ys or not zs:
        log.warning("No se encontraron coordenadas de ubicación para ningún nodo en la muestra. No se puede graficar en 3D.")
        return

    log.info(f"Se encontraron coordenadas de ubicación para {ubicaciones_encontradas} de {subg.vcount()} nodos en la muestra 3D.")

    # Extraer información de las aristas para el subgrafo
    edges = subg.get_edgelist()
    edge_x, edge_y, edge_z = [], [], []
    for src_subg_index, dst_subg_index in edges:
        try:
            edge_x += [xs[src_subg_index], xs[dst_subg_index], None]
            edge_y += [ys[src_subg_index], ys[dst_subg_index], None]
            edge_z += [zs[src_subg_index], zs[dst_subg_index], None]
        except IndexError:
            log.warning(f"Error de índice al procesar arista ({src_subg_index}, {dst_subg_index}) en el subgrafo.")

    # Crear la figura con Plotly
    fig = go.Figure()

    # Trazar las aristas
    fig.add_trace(go.Scatter3d(
        x=edge_x, y=edge_y, z=edge_z,
        mode='lines',
        line=dict(color='lightgray', width=0.3),
        hoverinfo='none'
    ))

    # Trazar los nodos
    fig.add_trace(go.Scatter3d(
        x=xs, y=ys, z=zs,
        mode='markers',
        marker=dict(
            size=3,
            color='blue',
            opacity=0.6
        ),
        hoverinfo='text',
        text=[f'Nodo {original_index + 1}' for original_index in nodos_para_grafico]
    ))

    fig.update_layout(
        title=f'Muestra de {subg.vcount()} Nodos del Grafo ({total_nodos} Nodos Totales) en 3D (Latitud, Longitud, Índice)',
        showlegend=False,
        scene=dict(
            xaxis_title='Latitud',
            yaxis_title='Longitud',
            zaxis_title='Índice',
            xaxis=dict(showbackground=False),
            yaxis=dict(showbackground=False),
            zaxis=dict(showbackground=False)
        ),
        margin=dict(l=0, r=0, b=0, t=40)
    )

    log.info(f"Mostrando la gráfica de la muestra de {subg.vcount()} nodos en 3D (Latitud, Longitud, Índice).")
    fig.show()

def graficar_grafo_mapa_2d_muestra(path_pkl, num_nodos_muestra=15000):
    """
    Carga un grafo desde un archivo pickle y grafica una muestra aleatoria de nodos en 2D estilo mapa.
    utilizando la latitud como coordenada X y la longitud como coordenada Y.

    Args:
        path_pkl (str): Ruta al archivo pickle del grafo de igraph.
        num_nodos_muestra (int): Número de nodos aleatorios a mostrar.
    """
    log.info(f"Intentando cargar el grafo desde: '{path_pkl}' para una muestra 2D de {num_nodos_muestra} nodos.")
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

    if num_nodos_muestra >= total_nodos:
        nodos_muestra = list(range(total_nodos))
        log.info(f"El número de nodos a mostrar ({num_nodos_muestra}) es mayor o igual al total de nodos ({total_nodos}). Graficando todos los nodos en 2D.")
    else:
        nodos_muestra_indices = random.sample(range(total_nodos), num_nodos_muestra)
        nodos_muestra = sorted(nodos_muestra_indices)
        log.info(f"Seleccionando aleatoriamente {num_nodos_muestra} nodos para la muestra 2D.")

    subg = g.subgraph(nodos_muestra)
    log.info(f"Subgrafo creado con {subg.vcount()} nodos para la muestra 2D.")

    # Obtener coordenadas de latitud y longitud (invertidas) para la muestra
    longitudes, latitudes = [], []
    nodos_para_grafico = []
    ubicaciones_encontradas = 0
    original_indices_subgrafo = sorted([g.vs[i].index for i in nodos_muestra])

    for i, original_index in enumerate(original_indices_subgrafo):
        try:
            v = subg.vs[i]
            lat = v['lat']
            lon = v['lon']
            longitudes.append(lat)  # Ahora longitudes almacena latitud (eje X)
            latitudes.append(lon)   # Ahora latitudes almacena longitud (eje Y)
            nodos_para_grafico.append(original_index)
            ubicaciones_encontradas += 1
        except KeyError:
            log.warning(f"Nodo con índice original {original_index + 1} no tiene atributos 'lat' o 'lon'. Se omitirá del gráfico 2D.")

    if not longitudes or not latitudes:
        log.warning("No se encontraron coordenadas de ubicación para ningún nodo en la muestra 2D. No se puede graficar en 2D.")
        return

    log.info(f"Se encontraron coordenadas de ubicación para {ubicaciones_encontradas} de {subg.vcount()} nodos en la muestra 2D.")

    # Extraer información de las aristas para los nodos con ubicación en el subgrafo
    edges = subg.get_edgelist()
    edge_x, edge_y = [], []
    for src_subg_index, dst_subg_index in edges:
        edge_x += [longitudes[src_subg_index], longitudes[dst_subg_index], None]
        edge_y += [latitudes[src_subg_index], latitudes[dst_subg_index], None]

    # Crear la figura con Plotly
    fig = go.Figure()

    # Trazar las aristas
    fig.add_trace(go.Scatter(
        x=edge_x, y=edge_y,
        mode='lines',
        line=dict(color='lightgray', width=0.1),
        hoverinfo='none'
    ))

    # Trazar los nodos
    fig.add_trace(go.Scatter(
        x=longitudes, y=latitudes,  # 'longitudes' ahora tiene latitud, 'latitudes' ahora tiene longitud
        mode='markers',
        marker=dict(
            size=1.5,
            color='blue',
            opacity=0.4
        ),
        hoverinfo='text',
        text=[f'Nodo {original_index + 1}' for original_index in nodos_para_grafico]
    ))

    fig.update_layout(
        title=f'Muestra de {subg.vcount()} Nodos del Grafo ({total_nodos} Nodos Totales) en 2D (Latitud, Longitud)',
        showlegend=False,
        xaxis_title='Latitud',
        yaxis_title='Longitud',
        margin=dict(l=0, r=0, b=0, t=40),
        yaxis=dict(scaleanchor="x", scaleratio=1)
    )

    log.info(f"Mostrando la gráfica de la muestra de {subg.vcount()} nodos en 2D (Latitud, Longitud).")
    fig.show()

if __name__ == "__main__":
    grafo_archivo = "grafo_igraph_paralelizado.pkl"
    num_nodos_a_mostrar = 15000
    graficar_grafo_3d_muestra(grafo_archivo, num_nodos_a_mostrar)
    graficar_grafo_mapa_2d_muestra(grafo_archivo, num_nodos_a_mostrar)