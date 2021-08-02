import numpy as np
import communities as c
from communities.algorithms import louvain_method
from communities.visualization import louvain_animation

def rmse(score_label):
    n = 0
    error = 0
    for s in score_label:
        error += (s[1] - s[0]) ** 2
        n += 1
    return np.sqrt(error/n)

def I_similarity(i, j):
    union=0
    sum = 0
    for s in i:
        if s in j:
            union = union+1#交集
            intersection = len(i)+len(j)-union #并集
            sum += float(union/intersection)
    return sum*2/(i*(i-1))

def U_homogeneity(u, v):
    union=0
    sum = 0
    for s in u:
        if s in v:
            union = union+1#交集
            intersection = len(u)+len(v)-union #并集
            sum += float(union/intersection)
    return sum*2/(u*(v-1))

adj_matrix = [...]
communities, _ = louvain_method(adj_matrix)#louvain_method(adj_matrix : numpy.ndarray, n : int = None) -> list
draw_communities(adj_matrix, communities)#draw_communities(adj_matrix : numpy.ndarray, communities : list, dark : bool = False, filename : str = None, seed : int = 1)

#animation: louvain_animation(adj_matrix : numpy.ndarray, frames : list, dark : bool = False, duration : int = 15, filename : str = None, dpi : int = None, seed : int = 2)
#https://cloud.tencent.com/developer/article/1801337