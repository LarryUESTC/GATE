import os
import numpy as np
import torch
import dgl
import csv
import random
from scipy.spatial import distance
from sklearn.linear_model import RidgeClassifier
from sklearn.feature_selection import RFE
from nilearn import connectome
from sklearn.model_selection import KFold, StratifiedKFold, StratifiedShuffleSplit

def feature_selection(matrix, labels, train_ind, fnum):
    """
        matrix       : feature matrix (num_subjects x num_features)
        labels       : ground truth labels (num_subjects x 1)
        train_ind    : indices of the training samples
        fnum         : size of the feature vector after feature selection

    return:
        x_data      : feature matrix of lower dimension (num_subjects x fnum)
    """

    estimator = RidgeClassifier()
    selector = RFE(estimator, n_features_to_select=fnum, step=100)

    featureX = matrix[train_ind, :]
    featureY = labels[train_ind]
    selector = selector.fit(featureX, featureY.ravel())
    x_data = selector.transform(matrix)

    print("Number of labeled samples %d" % len(train_ind))
    print("Number of features selected %d" % x_data.shape[1])

    return x_data

def import_data(dir_csv, args = None):
    data_all_list = []
    data_name = ['subject_id', 'old_id', 'asd_label', 'sex', 'age', 'site','feature']
    for i in data_name:
        data_all_list.append([])
    n = -1
    with open(dir_csv, 'rt') as f:
        cr = csv.reader(f)
        for row in cr:
            if n == -1:
                n += 1
                continue
            for i in range(len(data_name)-1):
                data_all_list[i].append(row[i])
            data_all_list[-1].append(row[len(data_name)-1:])
            n+=1
    return np.array({'n': n, 'data_feature':np.array(data_all_list[-1]).astype(np.float32), 'data_label':np.array(data_all_list[2]).astype(np.int),'data_sex':np.array(data_all_list[3]).astype(np.int),'data_age':np.array(data_all_list[4]).astype(np.float32)}, dtype=object)


def train_test(data_label, label_rate):
    train_index = []
    test_index = []
    for j in range(data_label.max().item() + 1):
        index = list(0, len(data_label) - 1)[(data_label == j)]
        x_list0 = random.sample(list(index), int(len(index) * label_rate))
        for x in x_list0:
            train_index.append(int(x))
    for c in range(len(data_label)):
        if int(c) not in train_index:
            test_index.append(int(c))
    return train_index, test_index

def train_test2(data_label, label_rate, data_feature_4000):

    kf = StratifiedShuffleSplit(n_splits=5, train_size=label_rate)
    train_list = []
    test_list = []
    for train_index, test_index in kf.split(data_feature_4000, data_label):
        train_list.append(list(train_index))
        test_list.append(list(test_index))

    return train_list, test_list

def prepocess_data_new(args, dataset):
    random.seed(args.seed)
    data_dir = args.data_dir
    label_rate = str(int(args.label_rate*100))
    seed = str(int(args.seed))
    dir_root_np = os.path.join(data_dir, dataset, seed, label_rate)
    dir_np = os.path.join(data_dir, dataset+'.npy')

    data_all_list = np.load(dir_np, allow_pickle=True).tolist()
    t = data_all_list['t']
    data_feature = torch.from_numpy(data_all_list['data_feature_list'])
    data_feature_st = torch.from_numpy(data_all_list['data_feature_list_st'])
    data_label = torch.from_numpy(data_all_list['data_label']).long()
    graph = torch.from_numpy(data_all_list['data_graph_list'])

    if not os.path.exists(os.path.join(dir_root_np, 'train' + '.npy')):
        if not os.path.exists(dir_root_np):
            os.makedirs(dir_root_np)
        try:
            train_index, test_index = train_test2(data_label, args.label_rate, data_feature[0])
        except:
            train_index, test_index = train_test2(data_label, args.label_rate, data_feature)
        np.save(os.path.join(dir_root_np, 'train' + '.npy'), train_index, allow_pickle=True)
        np.save(os.path.join(dir_root_np, 'test' + '.npy'), test_index, allow_pickle=True)
    train_index = np.load(os.path.join(dir_root_np, 'train' + '.npy'), allow_pickle=True).tolist()
    test_index = np.load(os.path.join(dir_root_np, 'test' + '.npy'), allow_pickle=True).tolist()

    return t, data_feature.size()[0], data_feature, data_feature_st, graph, data_label, train_index, test_index

def normalize_graph(A):
    eps = 2.2204e-10
    deg_inv_sqrt = (A.sum(dim=-1).clamp(min=0.)+eps).pow(-0.5)
    if A.size()[0] != A.size()[1]:
        A =  deg_inv_sqrt.unsqueeze(-1) * (deg_inv_sqrt.unsqueeze(-1) * A)
    else:
        A = deg_inv_sqrt.unsqueeze(-1) * A * deg_inv_sqrt.unsqueeze(-2)
    return A.clamp(max=1.0)

def normalize_graph_half(A):
    eps = torch.tensor([2.2204e-16]).half()
    deg_inv_sqrt = (A.sum(dim=-1).clamp(min=0.)+eps).pow(-0.5)
    if A.size()[0] != A.size()[1]:
        A =  deg_inv_sqrt.unsqueeze(-1) * (deg_inv_sqrt.unsqueeze(-1) * A)
    else:
        A = deg_inv_sqrt.unsqueeze(-1) * A * deg_inv_sqrt.unsqueeze(-2)
    return A

def create_graph(data_sex, data_age, data_feature):
    num_nodes = data_feature.size()[0]
    graph = torch.zeros((num_nodes, num_nodes))

    distv = distance.pdist(data_feature.cpu().numpy(), metric='correlation')
    # Convert to a square symmetric distance matrix
    dist = distance.squareform(distv)
    sigma = np.mean(dist)
    # Get affinity from similarity matrix
    sparse_graph = np.exp(- dist ** 2 / (2 * sigma ** 2))
    sparse_graph = torch.from_numpy(sparse_graph.astype(np.float32))

    for k in range(num_nodes):
        for j in range(k + 1, num_nodes):
            if data_sex[k] == data_sex[j]:
                graph[k, j] += 1
                graph[j, k] += 1
            val = abs(float(data_age[k]) - float(data_age[j]))
            if val < 1:
                graph[k, j] += 1
                graph[j, k] += 1
    graph = normalize_graph(graph)
    I_input = torch.eye(num_nodes)

    ori_graph = sparse_graph*graph
    topk = 5
    N = ori_graph.size()[0]
    maxadj = torch.topk(ori_graph, k=topk, dim=1, sorted=False, largest=True).values[:, -1].view(N, 1).repeat(1, N)
    maxadj = (ori_graph >= maxadj) + 0
    ori_graph = ori_graph * maxadj
    S = normalize_graph(ori_graph)
    graph = S + I_input
    return graph

def create_new_graph(data_feature):
    num_nodes = data_feature.size()[0]
    graph = torch.zeros((num_nodes, num_nodes))

    distv = distance.pdist(data_feature.cpu().numpy(), metric='correlation')
    # Convert to a square symmetric distance matrix
    dist = distance.squareform(distv)
    sigma = np.mean(dist)
    # Get affinity from similarity matrix
    sparse_graph = np.exp(- dist ** 2 / (2 * sigma ** 2))
    sparse_graph = torch.from_numpy(sparse_graph.astype(np.float32))

    return sparse_graph

def drop_feature(x, drop_prob):
    drop_mask = torch.empty(
        (x.size(1),),
        dtype=torch.float32,
        device=x.device).uniform_(0, 1) < drop_prob
    x = x.clone()
    x[:, drop_mask] = 0

    return x

def mask_edge(graph, mask_prob):
    E = graph.number_of_edges()

    mask_rates = torch.FloatTensor(np.ones(E) * mask_prob)
    masks = torch.bernoulli(1 - mask_rates)
    mask_idx = masks.nonzero().squeeze(1)
    return mask_idx

def RA(graph, x, feat_drop_rate, edge_mask_rate):
    n_node = graph.number_of_nodes()

    edge_mask = mask_edge(graph, edge_mask_rate)
    feat = drop_feature(x, feat_drop_rate)

    ng = dgl.graph([])
    ng.add_nodes(n_node)
    src = graph.edges()[0]
    dst = graph.edges()[1]

    nsrc = src[edge_mask]
    ndst = dst[edge_mask]
    ng.add_edges(nsrc, ndst)

    return ng, feat

