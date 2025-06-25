import igraph as ig
import graphistry
import random
import pickle
import logging as log
import pandas as pd

graphistry.register(api=3, username='MiquelAngFl', password='sxtXy3ZG.DAhdsg')
log.basicConfig(
    level=log.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        log.FileHandler("grafo_graphistry.log", mode='w', encoding='utf-8'),
        log.StreamHandler()
    ]
)

# Configura tu API Key de Graphistry (o usa login automático si ya estás autenticado)
graphistry.register(api=3, protocol="https", server="hub.graphistry.com")
# graphistry.register(api=3, username='TU_USUARIO', password='TU_CONTRASEÑA') # solo si quieres autenticación explícita

def graficar_con_graphistry(path_pkl: str, num_nodos_muestra: int = 15000):
    try:
        with open(path_pkl, 'rb') as f:
            g = pickle.load(f)
        log.info(f"Grafo cargado correctamente: {g.vcount()} nodos y {g.ecount()} aristas.")
    except Exception as e:
        log.error(f"Error al cargar el grafo: {e}")
        return

    # Limitar muestra
    if g.vcount() == 0:
        log.warning("Grafo vacío.")
        return

    nodos_muestra = list(range(g.vcount()))
    if num_nodos_muestra < g.vcount():
        nodos_muestra = random.sample(nodos_muestra, num_nodos_muestra)
        log.info(f"Mostrando una muestra aleatoria de {num_nodos_muestra} nodos.")

    subg = g.subgraph(nodos_muestra)

    # Crear DataFrames para Graphistry
    edges = subg.get_edgelist()
    df_edges = pd.DataFrame(edges, columns=['source', 'target'])

    df_nodes = pd.DataFrame({
        'node': [v.index for v in subg.vs],
        'label': [f"Nodo {v.index}" for v in subg.vs]
    })

    # Añadir lat/lon si existen (opcional)
    if 'lat' in subg.vs.attributes() and 'lon' in subg.vs.attributes():
        df_nodes['lat'] = subg.vs['lat']
        df_nodes['lon'] = subg.vs['lon']

        # Mostrar grafo en Graphistry
    plotter = graphistry.bind(source='source', destination='target', node='node')
    plot = plotter.edges(df_edges).nodes(df_nodes).plot()
    log.info(f"Visualización generada: {plot}")

if __name__ == "__main__":
    grafo_path = "grafo_igraph_paralelizado.pkl"
    num_muestra = 500_000
    graficar_con_graphistry(grafo_path, num_muestra)
