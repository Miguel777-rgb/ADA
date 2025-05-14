'''
from pyspark import SparkContext
print(SparkContext()._jvm.org.apache.hadoop.util.VersionInfo.getVersion())

import igraph as ig
print(ig.__version__)
'''
import igraph as ig

# Crear un grafo simple no dirigido
g = ig.Graph.Famous("Zachary")

try:
    dendrogram = g.community_fastgreedy()
    communities = dendrogram.as_clustering().membership
    print("Comunidades detectadas:", communities)
except AttributeError:
    print("Error: El método community_fastgreedy no se encontró.")
else:
    print("La instalación parece correcta.")