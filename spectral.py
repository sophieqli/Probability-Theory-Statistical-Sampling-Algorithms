import math
import torch
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

#Exploration of spectral theory: random walks, graph theory...etc

def lap(G):
    #unnormalized laplacian
    A = nx.to_numpy_array(G)
    D = np.diag(A.sum(axis = 1))
    return D - A

def eig_decomp_sorted(X):
    eigenvals, eigenvecs = torch.linalg.eigh(X)
    idx = torch.argsort(eigenvals, descending = True)
    eigenvals = eigenvals[idx]
    eigenvecs = eigenvecs[:, idx]
    return eigenvals, eigenvecs
def k_means(U, k): 
    #centroids
    n = U.shape[0]
    init_ind = torch.randperm(n)[:k]
    C = U[init_ind] #shld be k, k
    print("centroids init ")
    print(C)
    its = 5000
    C_tmp = {} #keys = clusters, values = list of points
    for i in range(its):
        for clus in range(k): C_tmp[clus] = []
        for pt in range(n):
            dists_pt = torch.sum((U[pt] - C)**2, dim = 1)
            C_tmp[torch.argmin(dists_pt).item()].append(pt)
        
        #update centroids
        for clus in range(k):
            if C_tmp[clus] == []: continue 
            tens_list = [U[pt] for pt in C_tmp[clus] ]
            C[clus] = torch.stack(tens_list).mean(dim=0)
    print("centroids points are, final ")
    print(C)
    print("groups are ")
    print(C_tmp)
    return C_tmp, C

def unnorm_cluster(G, k):
    L = lap(G)
    print("laplacian is ")
    print(L)
    vals, vecs = eig_decomp_sorted(torch.tensor(L))
    #get first k eigenvectors
    U = vecs[:, :k]
    print("U is ")
    print(U)
    n = U.shape[0]
    assert n >= k, "not possible "
    return k_means(U,k )

def norm_cluster(G, k):
    A = nx.to_numpy_array(G)
    D = torch.tensor(np.diag(A.sum(axis = 1)))
    L = lap(G)

    #compute L_sym 
    L_sym = torch.sqrt(torch.linalg.inv(D)) @ L @ torch.sqrt(torch.linalg.inv(D))
    vals, vecs = eig_decomp_sorted(torch.tensor(L_sym))
    #get first k eigenvectors
    U = vecs[:, :k]

    Unorm = U / torch.sqrt(torch.sum(U**2, axis = 0)) 
    return k_means(Unorm,k)

def sym_norm_trans_mat(G):
    A = nx.to_numpy_array(G)
    print("A is ")
    print(A)
    D = np.diag(A.sum(axis=1)) #diagonal degree matrix
    D_inv_sqrt = np.diag(1.0 / np.sqrt(np.diag(D)))
    print("D_inv_sqrt")
    P = D_inv_sqrt @ A @ D_inv_sqrt
    return torch.tensor(P, dtype=torch.float32)

def printGinfo(G, neighs = [0]):
    print("Nodes:", G.nodes())
    print("Edges:", G.edges())
    for i in neighs:
        print("neighbors of node ", i,": ", list(G.neighbors(i)))
    print("Degrees:", dict(G.degree()))
    nx.draw(G, with_labels=True, node_size=100)
    plt.show()

#Make plots                                                                                          
def make_plots(G, Clus):
  #up to 4 clusters (according to color map)
  node_colors = {}                                                                                     
  for cluster_id, nodes in Clus.items():                                                               
      for node in nodes:                                                                               
          node_colors[node] = cluster_id                                                               
                                                                                                        
  color_map = ['red', 'blue', 'green', 'purple']                                                       
  colors = [color_map[node_colors[n]] for n in G.nodes()]                                              
                                                                                                        
  # drawing graph                                                                                      
  pos = nx.spring_layout(G, seed=42)                                                                   
  plt.figure(figsize=(8, 6))                                                                           
  nx.draw_networkx_nodes(G, pos, node_color=colors, node_size=700, alpha=0.9)                          
  nx.draw_networkx_edges(G, pos, width=1.5, alpha=0.7)                                                 
  nx.draw_networkx_labels(G, pos, font_color='white', font_weight='bold')                              
  from matplotlib.patches import Patch                                                                 
  legend_elements = [Patch(facecolor=color_map[i], label=f'Cluster {i}') for i in range(len(color_map))]
  plt.legend(handles=legend_elements)                                                                  
  plt.title("Spectral Clustering of Graph Nodes")                                                      
  plt.axis('off')                                                                                      
  plt.tight_layout()                                                                                   
  plt.show()

make_plots(G, Clus)

import random

#simplified example, comparing diff clustering algos 
def make_clustered_graph():
    G = nx.Graph()
    num_clusters = 3
    nodes_per_cluster = 10
    total_nodes = num_clusters * nodes_per_cluster

    #put nodes in clusters
    cluster_nodes = []
    for i in range(num_clusters):
        nodes = list(range(i * nodes_per_cluster, (i + 1) * nodes_per_cluster))
        cluster_nodes.append(nodes)
        G.add_nodes_from(nodes)
        # dense internal connections (80% of edges)
        for u in nodes:
            for v in nodes:
                if u < v and random.random() < 0.8: G.add_edge(u, v)

    #sparse connections between clusters
    for i in range(num_clusters):
        for j in range(i + 1, num_clusters):
            u = random.choice(cluster_nodes[i])
            v = random.choice(cluster_nodes[j])
            G.add_edge(u, v)

    return G


#undirected graph
G = nx.grid_2d_graph(5, 5)
G = nx.convert_node_labels_to_integers(G)
#printGinfo(G)
Clus, C = unnorm_cluster(G, 4)
print("Clus")
print(Clus)
print("C")
print(C)


