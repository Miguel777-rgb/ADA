import igraph as ig
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import random
import logging as log
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
    try:
        import pickle
        with open(path_pkl, "rb") as f:
            g = pickle.load(f)
        log.info(f"Grafo cargado correctamente con {g.vcount()} nodos y {g.ecount()} aristas.")
        return g
    except Exception as e:
        log.error(f"Error al cargar el grafo: {e}")
        return None

def histograma_grados(grafo, top_n=20):
    grados = grafo.degree()
    in_grados = grafo.indegree() if grafo.is_directed() else grados
    out_grados = grafo.outdegree() if grafo.is_directed() else grados

    # Histograma de grados
    fig = px.histogram(grados, nbins=50, title="Histograma de grado total")
    fig.write_html("histograma_grado_total.html")
    log.info("Histograma de grado total guardado en histograma_grado_total.html")

    # Top-N nodos por grado
    top_idx = np.argsort(grados)[-top_n:][::-1]
    top_grados = [grados[i] for i in top_idx]
    fig2 = px.bar(x=top_idx, y=top_grados, title=f"Top-{top_n} nodos por grado total")
    fig2.write_html("top_nodos_grado.html")
    log.info(f"Top-{top_n} nodos por grado guardado en top_nodos_grado.html")

    # In/Out si es dirigido
    if grafo.is_directed():
        fig3 = px.histogram(in_grados, nbins=50, title="Histograma de grado de entrada")
        fig3.write_html("histograma_in_grado.html")
        fig4 = px.histogram(out_grados, nbins=50, title="Histograma de grado de salida")
        fig4.write_html("histograma_out_grado.html")
        log.info("Histogramas de in/out grado guardados.")

def top_nodos_importancia(grafo, n=20):
    # PageRank
    pr = grafo.pagerank()
    top_idx = np.argsort(pr)[-n:][::-1]
    log.info("Top nodos por PageRank:")
    for i in top_idx:
        log.info(f"Nodo {i+1}: PageRank={pr[i]:.6f}, Grado={grafo.degree(i)}")

    # Centralidad de intermediación (betweenness)
    log.info("Calculando betweenness (esto puede tardar)...")
    t0 = time.time()
    try:
        bt = grafo.betweenness(cutoff=100)  # cutoff para acelerar en grafos grandes
    except Exception:
        bt = grafo.betweenness()
    log.info(f"Betweenness calculado en {time.time()-t0:.2f}s")
    top_idx_bt = np.argsort(bt)[-n:][::-1]
    log.info("Top nodos por betweenness:")
    for i in top_idx_bt:
        log.info(f"Nodo {i+1}: Betweenness={bt[i]:.2f}")

def resumen_comunidades(comunidades):
    sizes = np.bincount(comunidades)
    log.info(f"Comunidades detectadas: {len(sizes)}")
    log.info(f"Tamaño de comunidades (top 10): {np.sort(sizes)[::-1][:10]}")

def visualizar_subgrafo_importante(grafo, comunidades, top_n=200):
    # Selecciona la comunidad más grande
    from collections import Counter
    comunidad_mayor = Counter(comunidades).most_common(1)[0][0]
    nodos_comunidad = [i for i, c in enumerate(comunidades) if c == comunidad_mayor]
    if len(nodos_comunidad) > top_n:
        nodos_muestra = random.sample(nodos_comunidad, top_n)
    else:
        nodos_muestra = nodos_comunidad
    subgrafo = grafo.subgraph(nodos_muestra)
    colores = px.colors.qualitative.Set2
    color_map = [colores[comunidades[i] % len(colores)] for i in nodos_muestra]
    layout = subgrafo.layout("fr")
    xs, ys = zip(*layout.coords)
    edge_x, edge_y = [], []
    for e in subgrafo.es:
        s, t = e.tuple
        edge_x += [xs[s], xs[t], None]
        edge_y += [ys[s], ys[t], None]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=edge_x, y=edge_y, mode='lines', line=dict(color='gray', width=0.5), hoverinfo='none'))
    fig.add_trace(go.Scatter(x=xs, y=ys, mode='markers', marker=dict(size=8, color=color_map, opacity=0.8),
                             text=[f'Nodo {nodos_muestra[i]+1}, Comunidad {comunidades[nodos_muestra[i]]}' for i in range(len(nodos_muestra))],
                             hoverinfo='text'))
    fig.update_layout(title=f"Subgrafo de la comunidad más grande ({len(nodos_muestra)} nodos)",
                      xaxis=dict(showgrid=False, zeroline=False), yaxis=dict(showgrid=False, zeroline=False))
    fig.write_html("subgrafo_comunidad_mayor.html")
    log.info("Visualización de la comunidad más grande guardada en subgrafo_comunidad_mayor.html")

if __name__ == "__main__":
    grafo_archivo = "grafo_igraph_paralelizado.pkl"
    grafo = cargar_grafo(grafo_archivo)
    if grafo:
        analisis_eda_basico = lambda g: None  # Opcional: puedes agregar tu función EDA aquí
        analisis_eda_basico(grafo)
        histograma_grados(grafo, top_n=30)
        top_nodos_importancia(grafo, n=20)

        # Comunidades (FastGreedy sobre grafo no dirigido de aristas mutuas)
        log.info("Detectando comunidades (FastGreedy sobre aristas mutuas)...")
        edges_mutuas = []
        for e in grafo.es:
            s, t = e.source, e.target
            if grafo.are_adjacent(t, s) and s != t:
                if (t, s) not in edges_mutuas:
                    edges_mutuas.append((s, t))
        grafo_mutuo = ig.Graph(n=grafo.vcount(), edges=edges_mutuas, directed=False)
        dendrogram = grafo_mutuo.community_fastgreedy()
        comunidades = dendrogram.as_clustering().membership
        resumen_comunidades(comunidades)
        visualizar_subgrafo_importante(grafo, comunidades, top_n=200)