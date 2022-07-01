#!/usr/bin/env python
# -*- coding: UTF-8 -*-
#This program is based on Population-GCN (https://github.com/parisots/population-gcn) and CCA-SSG (https://github.com/hengruizhang98/CCA-SSG)
import torch
import warnings
from colorama import init, Fore, Back, Style
import argparse
from tool_data import prepocess_data_new
from torch.nn.functional import cosine_similarity
import tool_graph
import random
import numpy as np
from statistics import mean, stdev
from model import GATE, LogReg
from sklearn.metrics import precision_score, accuracy_score, f1_score, recall_score, roc_auc_score
from sql_writer import WriteToDatabase, get_primary_key_and_value, get_columns
import socket, getpass, os
warnings.filterwarnings("ignore")


def my_train(data_feature_list, data_label, graph_list,
             train_index, test_index, args, fold_num, nb_t,
             config, writer, data_feature_N, graph_gl_list_list_N, nb_t_N):
    device = data_label.device
    nb_dim = data_feature_list.size(-1)
    N = data_feature_list.size(1)
    nb_classes = int(data_label.max() + 1)
    lastdim = config['lastdim']
    model = GATE(nb_dim, lastdim * config['plus'], lastdim, config['lastdim2'], 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr1'], weight_decay=config['weight_decay1'])
    model = model.to(device)

    acc_list = []
    auc_list = []
    precision_list = []
    recall_list = []
    f1_list = []

    acc_list_FT = []
    auc_list_FT = []
    precision_list_FT = []
    recall_list_FT = []
    f1_list_FT = []
    choose = [0, nb_t_N * 1, nb_t_N * 3, nb_t_N * 4]

    I_target = torch.tensor(np.eye(N)).to(device)

    for epoch in range(0, config['epoch_all1'] + 1):
        model.train()
        optimizer.zero_grad()
        k = random.randint(0, nb_t_N - 1)
        id_2_N = choose[random.randint(0, 3)] + k
        id_1 = random.randint(0, nb_t - 2)
        id_2 = id_1 + 1
        feat1_ori = data_feature_list[id_1]
        feat2_ori = data_feature_list[id_2]
        feat2_ori_N = data_feature_N[id_2_N]
        graph2_ori_N = graph_gl_list_list_N[-1][id_2_N]
        graph1_ori = graph_list[0][id_1]
        graph2_ori = graph_list[-1][id_2]

        if config['SAMA']:
            feat2_ori = random.sample([feat2_ori, feat2_ori_N], 1)[0]
            graph2_ori = random.sample([graph2_ori_N, graph2_ori], 1)[0]

        graph1, feat1 = tool_data.RA(graph1_ori.cpu(), feat1_ori, config['random_aug_feature'] / 10,
                                   config['random_aug_edge'] / 10).add_self_loop()
        graph2, feat2 = tool_data.RA(graph2_ori.cpu(), feat2_ori, config['random_aug_feature'] / 10,
                                   config['random_aug_edge'] / 10).add_self_loop()
        graph1 = graph1.to(device)
        graph2 = graph2.to(device)
        feat1 = feat1.to(device)
        feat2 = feat2.to(device)

        embeding_a, embeding_b, p1, p2 = model(graph1, feat1, graph2, feat2)

        c1 = torch.mm(embeding_a.T, embeding_a)/ N
        c2 = torch.mm(embeding_b.T, embeding_b)/ N

        loss_c1 = (I_target - c1).pow(2).mean() + torch.diag(c1).mean()
        loss_c2 = (I_target - c2).pow(2).mean() + torch.diag(c2).mean()
        loss = 1 - config['alpha'] * cosine_similarity(embeding_a, embeding_b.detach(), dim=-1).mean() + config['beta'] * (
                    loss_c1 + loss_c2)

        loss.backward()
        optimizer.step()

        #################################
        if epoch % 50 == 0 and epoch != 0:
            model.eval()
            embeds = 0
            test_graph_list = graph_list[0]
            for X_test, A_test in zip(data_feature_list, test_graph_list):
                A_test = A_test.to(device)
                A_test = A_test.remove_self_loop().add_self_loop()
                embeds += model.get_embedding(A_test, X_test)
            embeds = embeds / nb_t
            train_embs = embeds[train_index]
            test_embs = embeds[test_index]
            train_labels = data_label[train_index]
            ''' Linear Evaluation '''
            logreg = LogReg(train_embs.shape[1], nb_classes)
            opt = torch.optim.Adam(logreg.parameters(), lr=0.01, weight_decay=0e-5)

            logreg = logreg.to(device)
            loss_fn = torch.nn.CrossEntropyLoss()

            for epoch2 in range(100):
                logreg.train()
                opt.zero_grad()
                logits = logreg(train_embs)
                loss2 = loss_fn(logits, train_labels.squeeze())
                loss2.backward()
                opt.step()

            logreg.eval()
            with torch.no_grad():
                test_logits = logreg(test_embs)
                pred = torch.argmax(test_logits, dim=1)

                accs = accuracy_score(data_label[test_index].cpu(), pred.cpu())
                precision = precision_score(data_label[test_index].cpu(), pred.cpu())
                recall = recall_score(data_label[test_index].cpu(), pred.cpu())
                f1 = f1_score(data_label[test_index].cpu(), pred.cpu())
                try:
                    auc = roc_auc_score(data_label[test_index].cpu(), pred.cpu())
                except:
                    auc = f1 * 0

                acc_list.append(accs)
                auc_list.append(auc)
                precision_list.append(precision)
                recall_list.append(recall)
                f1_list.append(f1)
                writer.write({'epoch': epoch, "fold": fold_num, "seed": args.seed,
                              "alpha": config['alpha'],
                              "beta": config['beta'],
                              "lastdim": config['lastdim'],
                              "plus": config['plus'],
                              "lastdim2": config['lastdim2'],
                              "epoch_all1": config['epoch_all1'],
                              "epoch_all2": config['epoch_all2'],
                              "knn": config['knn'],
                              "lr1": config['lr1'],
                              "lr2": config['lr2'],
                              "weight_decay1": config['weight_decay1'],
                              "weight_decay2": config['weight_decay2'],
                              "random_aug_feature": config['random_aug_feature'],
                              "random_aug_edge": config['random_aug_edge'],
                              "SAMA": config['SAMA'],
                              },
                             {"train_acc": accs,
                              "train_precision": precision,
                              "train_recall": recall,
                              "train_f1": f1,
                              "train_auc": auc})

        #################################
        if epoch == config['epoch_all1']:
            opt = torch.optim.Adam(model.parameters(), lr=config['lr2'], weight_decay=config['weight_decay2'])
            loss_fn = torch.nn.CrossEntropyLoss()
            train_labels = data_label[train_index]
            test_labels = data_label[test_index]
            for epoch_FT in range(config['epoch_all2'] + 1):
                model.train()
                opt.zero_grad()
                embeds = 0
                test_graph_list = graph_list[0]
                for X_test, A_test in zip(data_feature_list, test_graph_list):
                    A_test = A_test.to(device)
                    A_test = A_test.remove_self_loop().add_self_loop()
                    embeds += model.finetune(A_test, X_test)
                pred = embeds / nb_t
                loss2 = loss_fn(pred[train_index], train_labels)
                loss2.backward()
                opt.step()

                if epoch_FT % 20 == 0 and epoch_FT != 0:
                    with torch.no_grad():
                        embeds = 0
                        test_graph_list = graph_list[0]
                        for X_test, A_test in zip(data_feature_list, test_graph_list):
                            A_test = A_test.to(device)
                            A_test = A_test.remove_self_loop().add_self_loop()
                            embeds += model.finetune(A_test, X_test)
                        embeds = embeds / nb_t
                        pred = torch.argmax(embeds, dim=1)[test_index]

                        accs_FT = accuracy_score(data_label[test_index].cpu(), pred.cpu())
                        precision_FT = precision_score(data_label[test_index].cpu(), pred.cpu())
                        recall_FT = recall_score(data_label[test_index].cpu(), pred.cpu())
                        f1_FT = f1_score(data_label[test_index].cpu(), pred.cpu())
                        try:
                            auc_FT = roc_auc_score(data_label[test_index].cpu(), pred.cpu())
                        except:
                            auc_FT = f1 * 0
                        model.eval()
                        acc_list_FT.append(accs_FT)
                        auc_list_FT.append(auc_FT)
                        precision_list_FT.append(precision_FT)
                        recall_list_FT.append(recall_FT)
                        f1_list_FT.append(f1_FT)
                        writer.write({'epoch': config['epoch_all1'] + epoch_FT, "fold": fold_num, "seed": args.seed,
                                      "alpha": config['alpha'],
                                      "beta": config['beta'],
                                      "lastdim": config['lastdim'],
                                      "plus": config['plus'],
                                      "lastdim2": config['lastdim2'],
                                      "epoch_all1": config['epoch_all1'],
                                      "epoch_all2": config['epoch_all2'],
                                      "knn": config['knn'],
                                      "lr1": config['lr1'],
                                      "lr2": config['lr2'],
                                      "weight_decay1": config['weight_decay1'],
                                      "weight_decay2": config['weight_decay2'],
                                      "random_aug_feature": config['random_aug_feature'],
                                      "random_aug_edge": config['random_aug_edge'],
                                      "SAMA": config['SAMA'],
                                      },
                                     {"test_acc": accs_FT,
                                      "test_precision": precision_FT,
                                      "test_recall": recall_FT,
                                      "test_f1": f1_FT,
                                      "test_auc": auc_FT})

    torch.save(model.state_dict(), args.save_dir + '/' + args.dataset + '_FOLD_' + str(fold_num) + '.pth')

    return acc_list, auc_list, precision_list, recall_list, f1_list, \
           acc_list_FT, auc_list_FT, precision_list_FT, recall_list_FT, f1_list_FT


def my_test(data_feature_list, data_label, graph_list,
            train_index, test_index, args, fold_num, nb_t,
            config, writer, data_feature_N, graph_gl_list_list_N, nb_t_N):
    device = data_label.device
    nb_dim = data_feature_list.size(-1)
    lastdim = config['lastdim']
    model = GATE(nb_dim, lastdim * config['plus'], lastdim, config['lastdim2'], 1, False)
    if os.path.exists(args.save_dir + '/' + args.dataset + '_FOLD_' + str(fold_num) + '.pth'):
        load_params = torch.load(args.save_dir + '/' + args.dataset + '_FOLD_' + str(fold_num) + '.pth')
        model_params = model.state_dict()
        same_parsms = {k: v for k, v in load_params.items() if k in model_params.keys()}
        model_params.update(same_parsms)
        model.load_state_dict(model_params)
    model = model.to(device)
    acc_list = []
    auc_list = []
    precision_list = []
    recall_list = []
    f1_list = []
    acc_list_FT = []
    auc_list_FT = []
    precision_list_FT = []
    recall_list_FT = []
    f1_list_FT = []

    with torch.no_grad():
        embeds = 0
        test_graph_list = graph_list[0]
        for X_test, A_test in zip(data_feature_list, test_graph_list):
            A_test = A_test.to(device)
            A_test = A_test.remove_self_loop().add_self_loop()
            embeds += model.finetune(A_test, X_test)
        embeds = embeds / nb_t
        pred = torch.argmax(embeds, dim=1)[test_index]

        accs_FT = accuracy_score(data_label[test_index].cpu(), pred.cpu())
        precision_FT = precision_score(data_label[test_index].cpu(), pred.cpu())
        recall_FT = recall_score(data_label[test_index].cpu(), pred.cpu())
        f1_FT = f1_score(data_label[test_index].cpu(), pred.cpu())
        try:
            auc_FT = roc_auc_score(data_label[test_index].cpu(), pred.cpu())
        except:
            auc_FT = f1_FT * 0
        model.eval()
        acc_list_FT.append(accs_FT)
        auc_list_FT.append(auc_FT)
        precision_list_FT.append(precision_FT)
        recall_list_FT.append(recall_FT)
        f1_list_FT.append(f1_FT)
    string_2 = Fore.GREEN + "accs: {:.3f},auc: {:.3f},pre: {:.3f},recall: {:.3f},f1: {:.3f}".format(
        accs_FT, auc_FT, precision_FT, recall_FT, f1_FT)
    print(string_2)
    return acc_list, auc_list, precision_list, recall_list, f1_list, \
           acc_list_FT, auc_list_FT, precision_list_FT, recall_list_FT, f1_list_FT


def main_A(config, checkpoint_dir=None):
    host_name = socket.gethostname()
    TABLE_NAME = 'TMI_GATE'
    PRIMARY_KEY, PRIMARY_VALUE = get_primary_key_and_value(
        {
            'alpha': ["double precision", None],
            'beta': ["double precision", None],
            'lastdim': ['integer', None],
            "plus": ['integer', None],
            "lastdim2": ['integer', None],
            'epoch_all1': ['integer', None],
            'epoch_all2': ['integer', None],
            'knn': ['integer', None],
            'lr1': ["double precision", None],
            'lr2': ["double precision", None],
            'weight_decay1': ["double precision", None],
            'weight_decay2': ["double precision", None],
            'random_aug_feature': ['integer', None],
            'random_aug_edge': ['integer', None],
            "SAMA": ['bool', None],
            "seed": ["integer", None],
            "dataset": ["text", args.dataset],
            "label": ["double precision", args.label_rate],
            "epoch": ["integer", None],
            "fold": ["integer", None],
            "model_name": ["text", host_name + os.path.split(__file__)[-1][:-3]]
        }
    )

    REFRESH = False
    OVERWRITE = True

    test_val_metrics = {
        "acc": None,
        "auc": None,
        "precision": None,
        "recall": None,
        "f1": None,
    }
    train_val_metrics = {
        "acc": None,
        "auc": None,
        "precision": None,
        "recall": None,
        "f1": None,
    }
    writer = WriteToDatabase({'host': "xxx.xxx.xxx.xxx", "port": "xxxxx",
                              "database": "xxxxx", "user": "xxxxx", "password": "xxxxx"},
                             TABLE_NAME,
                             PRIMARY_KEY,
                             get_columns(train_val_metrics, test_val_metrics),
                             PRIMARY_VALUE,
                             PRIMARY_VALUE,
                             REFRESH,
                             OVERWRITE)
    writer.init()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    nb_t, nb_nodes, data_feature, data_feature_st, graph_st, data_label, train_index_list, test_index_list = prepocess_data_new(args, args.dataset+'_SA')
    nb_t_N, nb_nodes_N, data_feature_N, data_feature_st_N, graph_st_N, data_label_N, train_index_list_N, test_index_list_N = prepocess_data_new(args, args.dataset+'_MA')
    data_label = data_label.squeeze().to(device)

    graph_list_list = []
    graph_gl_list_list = []
    graph_list_list_N = []
    graph_gl_list_list_N = []
    for b in range(1, config['knn'] + 1):
        graph_list = [tool_graph.nor_graph(graph_st[k], topk=b, w=0.5) for k in range(nb_t)]
        graph_gl_list = [tool_graph.torch2dgl(A) for A in graph_list]
        graph_list_list.append(graph_list)
        graph_gl_list_list.append(graph_gl_list)

    for b in range(1, config['knn'] + 1):
        graph_list_N = [tool_graph.nor_graph(graph_st_N[k], topk=b, w=0.5) for k in range(nb_t_N * 5)]
        graph_gl_list_N = [tool_graph.torch2dgl(A) for A in graph_list_N]
        graph_list_list_N.append(graph_list_N)
        graph_gl_list_list_N.append(graph_gl_list_N)

    data_feature_list = data_feature.to(device)
    data_feature_N = data_feature_N.to(device)

    acc_list_seed = []
    auc_list_seed = []
    precision_list_seed = []
    recall_list_seed = []
    f1_list_seed = []

    acc_list_seed_FT = []
    auc_list_seed_FT = []
    precision_list_seed_FT = []
    recall_list_seed_FT = []
    f1_list_seed_FT = []

    for seed in range(2021, 2024):
        args.seed = seed
        torch.manual_seed(seed)
        random.seed(seed)
        fold_num = 0
        acc_list = []
        acc_current = 0
        acc_list_mlp = []
        auc_list_mlp = []
        precision_list_mlp = []
        recall_list_mlp = []
        f1_list_mlp = []

        acc_list_mlp_FT = []
        auc_list_mlp_FT = []
        precision_list_mlp_FT = []
        recall_list_mlp_FT = []
        f1_list_mlp_FT = []

        for train_index, test_index in zip(train_index_list, test_index_list):
            fold_num += 1
            if args.train == True:
                acc_list, auc_list, precision_list, recall_list, f1_list, acc_list_FT, auc_list_FT, precision_list_FT, recall_list_FT, f1_list_FT \
                    = my_train(data_feature_list, data_label, graph_gl_list_list, train_index, test_index, args,
                               fold_num, nb_t, config, writer,
                               data_feature_N, graph_gl_list_list_N, nb_t_N)
                acc_list_mlp.append(mean(acc_list[-2:]))
                auc_list_mlp.append(mean(auc_list[-2:]))
                precision_list_mlp.append(mean(precision_list[-2:]))
                recall_list_mlp.append(mean(recall_list[-2:]))
                f1_list_mlp.append(mean(f1_list[-2:]))

                acc_list_mlp_FT.append(acc_list_FT[-1])
                auc_list_mlp_FT.append(auc_list_FT[-1])
                precision_list_mlp_FT.append(precision_list_FT[-1])
                recall_list_mlp_FT.append(recall_list_FT[-1])
                f1_list_mlp_FT.append(f1_list_FT[-1])
                print("=" * 10 + str(fold_num) + "_End" + "=" * 10)
            else:
                acc_list, auc_list, precision_list, recall_list, f1_list, acc_list_FT, auc_list_FT, precision_list_FT, recall_list_FT, f1_list_FT \
                    = my_test(data_feature_list, data_label, graph_gl_list_list, train_index, test_index, args,
                              fold_num, nb_t, config, writer,
                              data_feature_N, graph_gl_list_list_N, nb_t_N)

                acc_list_mlp.append(acc_list_FT[-1])
                auc_list_mlp.append(auc_list_FT[-1])
                precision_list_mlp.append(precision_list_FT[-1])
                recall_list_mlp.append(recall_list_FT[-1])
                f1_list_mlp.append(f1_list_FT[-1])

                acc_list_mlp_FT.append(acc_list_FT[-1])
                auc_list_mlp_FT.append(auc_list_FT[-1])
                precision_list_mlp_FT.append(precision_list_FT[-1])
                recall_list_mlp_FT.append(recall_list_FT[-1])
                f1_list_mlp_FT.append(f1_list_FT[-1])
                print("=" * 10 + str(fold_num) + " FOLD End" + "=" * 10)



        string_end = Fore.BLUE + "accs: {:.2f}, std: {:.2f}".format(mean(acc_list_mlp_FT) * 100, stdev(acc_list_mlp_FT) * 100)
        print(string_end)
        writer.write({'epoch': -1, "fold": -1, "seed": args.seed,
                      "alpha": config['alpha'],
                      "beta": config['beta'],
                      "lastdim": config['lastdim'],
                      "plus": config['plus'],
                      "lastdim2": config['lastdim2'],
                      "epoch_all1": config['epoch_all1'],
                      "epoch_all2": config['epoch_all2'],
                      "knn": config['knn'],
                      "lr1": config['lr1'],
                      "lr2": config['lr2'],
                      "weight_decay1": config['weight_decay1'],
                      "weight_decay2": config['weight_decay2'],
                      "random_aug_feature": config['random_aug_feature'],
                      "random_aug_edge": config['random_aug_edge'],
                      "SAMA": config['SAMA'],
                      },
                     {"test_acc": mean(acc_list_mlp_FT),
                      "test_precision": mean(precision_list_mlp_FT),
                      "test_recall": mean(recall_list_mlp_FT),
                      "test_f1": mean(f1_list_mlp_FT),
                      "test_auc": mean(auc_list_mlp_FT),
                      "train_acc": mean(acc_list_mlp),
                      "train_precision": mean(precision_list_mlp),
                      "train_recall": mean(recall_list_mlp),
                      "train_f1": mean(f1_list_mlp),
                      "train_auc": mean(auc_list_mlp)
                      },
                     )

        acc_list_seed.append(mean(acc_list_mlp))
        auc_list_seed.append(mean(precision_list_mlp))
        precision_list_seed.append(mean(recall_list_mlp))
        recall_list_seed.append(mean(f1_list_mlp))
        f1_list_seed.append(mean(auc_list_mlp))

        acc_list_seed_FT.append(mean(acc_list_mlp_FT))
        auc_list_seed_FT.append(mean(precision_list_mlp_FT))
        precision_list_seed_FT.append(mean(recall_list_mlp_FT))
        recall_list_seed_FT.append(mean(f1_list_mlp_FT))
        f1_list_seed_FT.append(mean(auc_list_mlp_FT))

    writer.write({'epoch': -2, "fold": -2, "seed": -1,
                  "alpha": config['alpha'],
                  "beta": config['beta'],
                  "lastdim": config['lastdim'],
                  "plus": config['plus'],
                  "lastdim2": config['lastdim2'],
                  "epoch_all1": config['epoch_all1'],
                  "epoch_all2": config['epoch_all2'],
                  "knn": config['knn'],
                  "lr1": config['lr1'],
                  "lr2": config['lr2'],
                  "weight_decay1": config['weight_decay1'],
                  "weight_decay2": config['weight_decay2'],
                  "random_aug_feature": config['random_aug_feature'],
                  "random_aug_edge": config['random_aug_edge'],
                  "SAMA": config['SAMA'],
                  },
                 {"train_acc": mean(acc_list_seed),
                  "train_precision": mean(auc_list_seed),
                  "train_recall": mean(precision_list_seed),
                  "train_f1": mean(recall_list_seed),
                  "train_auc": mean(f1_list_seed),
                  "test_acc": mean(acc_list_seed_FT),
                  "test_precision": mean(precision_list_seed_FT),
                  "test_recall": mean(recall_list_seed_FT),
                  "test_f1": mean(f1_list_seed_FT),
                  "test_auc": mean(auc_list_seed_FT)
                  })


def main(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = "6"

    config = {
        "alpha": 1,
        "beta": 0.2,
        'lastdim': 256,
        'plus': 1,
        'lastdim2': 128,
        'epoch_all1': 400,
        'epoch_all2': 100,
        'knn': 5,
        'lr1': 0.001,
        'lr2': 0.0001,
        'weight_decay1': 1e-5,
        'weight_decay2': 1e-5,
        'random_aug_feature': 3,
        'random_aug_edge': 3,
        'SAMA': True,
    }
    main_A(config)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', type=int, default=2022)
    parser.add_argument('--res_dir', type=str, default='./result/')
    parser.add_argument('--data_dir', type=str, default='./data/')
    parser.add_argument('--dataset', type=str, default='Ori_ABIDE', choices=['Ori_FTD',
                                                                             'Ori_ABIDE',
                                                                             ])

    parser.add_argument('--save_dir', type=str, default='./modelsave/')
    parser.add_argument('--train', type=bool, default=False)
    parser.add_argument('--label_rate', type=float, default=0.2)

    args = parser.parse_args()
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    main(args)
