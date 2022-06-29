import os
import numpy as np
import torch
import csv
import random
from scipy.spatial import distance
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.manifold import TSNE
import dgl
import torch.nn.functional as F

def normalize_graph(A):
    eps = 2.2204e-16
    deg_inv_sqrt = (A.sum(dim=-1).clamp(min=0.)+eps).pow(-0.5)
    if A.size()[0] != A.size()[1]:
        A =  deg_inv_sqrt.unsqueeze(-1) * (deg_inv_sqrt.unsqueeze(-1) * A)
    else:
        A = deg_inv_sqrt.unsqueeze(-1) * A * deg_inv_sqrt.unsqueeze(-2)
    return A


def plot_embedding(data, label, title = None):
    # label = np.ones(205)
    tsne = TSNE(n_components=2, init='pca', random_state=0)
    data = tsne.fit_transform(tsne.fit_transform(data[0].cpu().numpy()))
    x_min, x_max = np.min(data, 0), np.max(data, 0)
    data = (data - x_min) / (x_max - x_min)

    fig = plt.figure()
    ax = plt.subplot(111)
    for i in range(data.shape[0]):
        plt.text(data[i, 0], data[i, 1], str(label[i]),
                 color=plt.cm.Set1(label[i] / 10.),
                 fontdict={'weight': 'bold', 'size': 9})
    plt.xticks([])
    plt.yticks([])
    # plt.title(title)
    # plt.show()
    fig.savefig("Test_1.png", dpi=300, bbox_inches='tight')
    return fig
# BFS1(seed=0)

def torch2dgl(graph):
    N = graph.shape[0]
    graph_sp = graph.to_sparse()
    edges_src = graph_sp.indices()[0]
    edges_dst = graph_sp.indices()[1]
    edge_features = graph_sp.values()
    graph_dgl = dgl.graph((edges_src, edges_dst), num_nodes=N)
    graph_dgl.edata['weight'] = edge_features
    return graph_dgl



def nor_graph(ori_graph, topk = 5, w = 0.5):
    N = ori_graph.size()[0]
    I_input = torch.eye(N)
    top_adj = torch.topk(ori_graph, k=topk, dim=1, sorted=False, largest=True)
    maxadj = top_adj.values[:, -1].view(N, 1).repeat(1, N)
    maxlist = top_adj[1]
    maxadj = (ori_graph >= maxadj) + 0
    ori_graph = ori_graph * maxadj
    S = normalize_graph(ori_graph)

    graph = S*(1-w)  + I_input*w
    return graph