#originally, unzipped colab file 
#from google.colab import files
#uploaded = files.upload()
#!tar -xvzf facebook.tar.gz

from spectral.py import *

import os
import networkx as nx
import numpy as np
import pandas as pd

def load_ego_graph(ego_id, data_dir='facebook'):
    edge_file = os.path.join(data_dir, f"{ego_id}.edges")
    G = nx.read_edgelist(edge_file, nodetype=int)

    #ego node
    ego_node = int(ego_id)
    G.add_node(ego_node)
    G.add_edges_from((ego_node, n) for n in G.nodes if n != ego_node)

    #load features
    feat_file = os.path.join(data_dir, f"{ego_id}.feat")
    egofeat_file = os.path.join(data_dir, f"{ego_id}.egofeat")
    featnames_file = os.path.join(data_dir, f"{ego_id}.featnames")

    X = np.loadtxt(feat_file)
    node_ids = X[:, 0].astype(int)
    features = X[:, 1:]

    #add ego feats
    ego_feat = np.loadtxt(egofeat_file).reshape(1, -1)
    features = np.vstack([features, ego_feat])
    node_ids = np.append(node_ids, ego_node)

    with open(featnames_file) as f:
        feat_names = [line.strip() for line in f]

    df = pd.DataFrame(features, index=node_ids, columns=feat_names)
    return G, df

#get one graph 
G, features = load_ego_graph("107")
print(f"Graph summary:\n"
      f"Nodes: {G.number_of_nodes()}\n"
      f"Edges: {G.number_of_edges()}\n"
      f"Is connected: {nx.is_connected(G)}\n"
      f"Average degree: {sum(dict(G.degree()).values()) / G.number_of_nodes():.2f}")

#Stage 1 (Naive): Do PCA on "features" df
X_np = features.to_numpy()
X_tensor = torch.tensor(X_np, dtype=torch.float32)
X_tensor -= torch.mean(X_tensor, dim = 0) #center at 0

Cov = X_tensor.T @ X_tensor
eigvals, eigvecs = eig_decomp_sorted(Cov)
U = eigvecs.T #rows as eigenvecs

#Want to explain 80% of variance 
total_variance = torch.sum(eigvals)
cumulative_variance = torch.cumsum(eigvals, dim=0)
cumulative_variance
cumulative_variance = cumulative_variance / total_variance

idx_80 = 0
for i in range(len(cumulative_variance)):
  if cumulative_variance[i] >= 0.80:
    idx_80 = i
    break
print(idx_80+1, " components explain 80% variance ")

coords_80 = (U[:idx_80, ] @ X_tensor.T) #coords, with axes = principal components explaining 80% variance


