import scipy.io as sio
import os
from nilearn.connectome import ConnectivityMeasure
import numpy as np
from sklearn.linear_model import RidgeClassifier
from sklearn.feature_selection import RFE
from sklearn.decomposition import PCA
from scipy.spatial import distance
import random
import ABIDEParser as Reader
dataname = 'ABIDE'
rootpath= r"your_dir\ABIDE_pcp\cpac\filt_noglobal"
dim = 512
windowsize =[10,20,30,40,50]
strite = 15

subject_IDs = Reader.get_ids()
graph_feat = Reader.create_affinity_graph_from_scores(['SEX', 'SITE_ID'], subject_IDs)

labels = Reader.get_subject_score(subject_IDs, score='DX_GROUP')
num_nodes = subject_IDs.size
Time = []

y = np.zeros([num_nodes, 1])

for i in range(num_nodes):
    y[i] = int(labels[subject_IDs[i]])-1

for name in subject_IDs:
    name_path = os.path.join(rootpath, name)
    feature = []
    for time_dir in os.listdir(name_path):
        if time_dir[-3:] == ".1D":
            feature_dir = os.path.join(name_path, time_dir)
            i = 0
            for line in open(feature_dir, "r"):
                if i == 0:
                    i += 1
                    continue
                temp = line[:-1].split('\t')
                feature.append([float(x) for x in temp])
    Time.append(np.array(feature))


from nilearn import connectome
conn_measure = connectome.ConnectivityMeasure(kind='correlation')

random_list = []
random_list_np = np.array([])
random_list_selected = []

mask = (np.triu_indices(110)[0], np.triu_indices(110)[1]+1)
t = 15

window_point_Time = []
for size in windowsize:
    point_Time = []
    for x in Time:
        time = x.shape[0]
        window_point = []
        for i in range(t):
            k = windowsize[-1] / 2 + strite * i
            if k + windowsize[-1] / 2 >= (time - 1):
                k = random.randint(windowsize[-1] / 2, time - windowsize[-1] / 2 - 1)
            begin = int(k - size / 2)
            end = int(k + size / 2)
            window_point.append([begin, k , end])
        point_Time.append(window_point)

    window_point_Time.append(point_Time)

for i in range(t*5):
    all_feature = []
    j = 0
    w_p = i % t
    e_p = int(i / t)
    for x in Time:
        begin = window_point_Time[e_p][j][w_p][0]
        end =   window_point_Time[e_p][j][w_p][-1]
        ppc = conn_measure.fit_transform([x[begin:end]])[0]
        # ppc = np.corrcoef(x[k:k+windowsize].T)
        feature_i = ppc[mask]
        # print(np.where(np.isnan(feature_i)))
        temp = feature_i.astype(np.float32)
        all_feature.append(temp)
        j+=1

    random_list.append(all_feature)
    try:
        random_list_np = np.concatenate((random_list_np, all_feature), 0)
    except:
        random_list_np = all_feature

A_selected_list = []
A_selected_mean = []
pca = PCA(n_components=dim)
pca.fit(random_list_np)
for feature in random_list:
    x_selected = pca.transform(feature)
    random_list_selected.append(x_selected)

    distv = distance.pdist(x_selected, metric='correlation')
    dist = distance.squareform(distv)
    sigma = np.mean(dist)
    graph = np.exp(- dist ** 2 / (2 * sigma ** 2))
    final_graph = graph_feat * graph
    A_selected_list.append(final_graph)

data_all_list = np.array({'t': t,
                'data_feature_list':np.array(random_list),
                'data_feature_list_st':np.array(random_list_selected),
                'data_label': y,
                'data_graph_list':np.array(A_selected_list).astype(np.float32),
                },
             dtype=object)
np.save(r'your_dir/Ori_'+dataname+ '_MA'  +'.npy', data_all_list, allow_pickle=True)

print("DONE")