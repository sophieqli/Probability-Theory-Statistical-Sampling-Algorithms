#originally, unzipped colab file 
#from google.colab import files
#uploaded = files.upload()
#!tar -xvzf facebook.tar.gz

#dataset link: https://snap.stanford.edu/data/egonets-Facebook.html

from spectral.py import *

import os
import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
cumulative_variance = cumulative_variance / total_variance

idx_80 = 0
for i in range(len(cumulative_variance)):
  if cumulative_variance[i] >= 0.80:
    idx_80 = i
    break
print(idx_80+1, " components explain 80% variance ")

coords_80 = (U[:idx_80, ] @ X_tensor.T) #coords, with axes = principal components explaining 80% variance

def PCAplots(coords, cumulative_variance):
    #variance explained vs components 
    plt.plot(cumulative_variance.numpy())
    plt.xlabel("Number of Principal Components")
    plt.ylabel("Cumulative Variance Explained")
    plt.title("Scree Plot")
    plt.grid(True)
    plt.show()

    #projection onto first two components (definitely shows clustering)
    plt.scatter(coords_80[0], coords_80[1])
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("Projection onto First 2 Principal Components")
    plt.show()


    from mpl_toolkits.mplot3d import Axes3D  #3D plotting
    #first 3 principal components (again, clusteing is visible)
    pc1 = coords_80[0]
    pc2 = coords_80[1]
    pc3 = coords_80[2]

    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(pc1, pc2, pc4, c='blue', alpha=0.7)

    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_zlabel('PC3')
    ax.set_title('3D PCA Projection')

    plt.show()


#How well do the first i components approximate the original data?
#error metric using frobenius norm

errors = []
num_components = range(1, U.shape[0] + 1)

for idx in num_components:
    coords = (U[:idx, ] @ X_tensor.T)
    X_reconstructed = (U[:idx, :].T @ coords).T
    reconstruction_error = torch.norm(X_tensor - X_reconstructed) / torch.norm(X_tensor)
    errors.append(reconstruction_error.item())

#it's alright, not great

#Stage 2: try PCA using SVD (more numerically stable)
