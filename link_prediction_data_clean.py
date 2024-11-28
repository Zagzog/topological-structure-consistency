# -*- coding:utf-8 -*-
"""
created by zazo
2022.8.10
"""
import os
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import networkx as nx
from matplotlib.colors import LinearSegmentedColormap
import random
from scipy import optimize
from scipy.optimize import curve_fit


def power_law(x, a, b):
    return a * x ** b

# 按测试集大小来汇总数据，共6个网络，每个网络一个文件
def clean_base_probesize(graph, link_type):
    store = []
    probe_size = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45]
    for ps in probe_size:
        if link_type == 'deg':
            #path = 'F:/论文/lp-new/link-prediction/' + graph + '/alpha2/' + link_type + '/probe_size' + str(ps)
            if graph == 'sampled SmaGri' or graph == 'sampled Kohonen' or graph == 'sampled Delicious':
                path = 'F:/论文/lp-new/link-prediction/' + graph + '/alpha8/' + link_type + '/probe_size' + str(ps)
            else:
                path = 'F:/论文/lp-new/link-prediction/' + graph + '/alpha8/' + link_type + '/probe_size' + str(ps)
        elif link_type == 'dis':
            #path = 'F:/论文/lp-new/link-prediction/' + graph + '/alpha-2/' + link_type + '/probe_size' + str(ps)
            if graph == 'sampled SmaGri' or graph == 'sampled Kohonen' or graph == 'sampled Delicious':
                path = 'F:/论文/lp-new/link-prediction/' + graph + '/alpha-8/' + link_type + '/probe_size' + str(ps)
            else:
                path = 'F:/论文/lp-new/link-prediction/' + graph + '/alpha-8/' + link_type + '/probe_size' + str(ps)
        else:
            path = 'F:/论文/lp-new/link-prediction/' + graph + '/' + link_type + '/probe_size' + str(ps)
        f = open(path + '/average_auc.txt', 'r')
        lines = f.readlines()
        f.close()
        store.append(lines[0])
    path1 = 'F:/论文/lp-new/link-prediction/results/probe_size/' + graph
    folder = os.path.exists(path1)
    if not folder:
        os.makedirs(path1)
    f = open(path1 + '/' + link_type + '_auc.txt', 'w')
    for i in store:
        f.write(str(i) + '\n')
    f.close()


def clean_base_probesize_errorbar(graph, link_type):
    store = []
    probe_size = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45]
    for ps in probe_size:
        store1 = []
        if link_type == 'deg':
            # path = 'F:/论文/lp-new/link-prediction/' + graph + '/alpha2/' + link_type + '/probe_size' + str(ps)
            if graph == 'sampled SmaGri' or graph == 'sampled Kohonen' or graph == 'sampled Delicious':
                path = 'F:/论文/lp-new/link-prediction/' + graph + '/alpha8/' + link_type + '/probe_size' + str(ps)
            else:
                path = 'F:/论文/lp-new/link-prediction/' + graph + '/alpha8/' + link_type + '/probe_size' + str(ps)
        elif link_type == 'dis':
            # path = 'F:/论文/lp-new/link-prediction/' + graph + '/alpha-2/' + link_type + '/probe_size' + str(ps)
            if graph == 'sampled SmaGri' or graph == 'sampled Kohonen' or graph == 'sampled Delicious':
                path = 'F:/论文/lp-new/link-prediction/' + graph + '/alpha-8/' + link_type + '/probe_size' + str(ps)
            else:
                path = 'F:/论文/lp-new/link-prediction/' + graph + '/alpha-8/' + link_type + '/probe_size' + str(ps)
        else:
            path = 'F:/论文/lp-new/link-prediction/' + graph + '/' + link_type + '/probe_size' + str(ps)
        f = open(path + '/auc.txt', 'r')
        lines = f.readlines()
        f.close()
        for line in lines:
            item = line.strip('\n')
            store1.append(float(item))

        store.append(np.std(store1))

    path1 = 'F:/论文/lp-new/link-prediction/results/probe_size/' + graph
    folder = os.path.exists(path1)
    if not folder:
        os.makedirs(path1)

    f = open(path1 + '/' + link_type + '_auc_errorbar.txt', 'w')
    for i in store:
        f.write(str(i) + '\n')
    f.close()


def plot_probsize():
    graph_type = ['new FW1', 'new FW2', 'new FW3',  'sampled SmaGri', 'sampled Kohonen', 'sampled Delicious']
    link_type1 = ['deg', 'dis', 'cn', 'pa', 'lr', 'rwr_ver2']
    x = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45]
    lab_list = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)']

    color_list = ['#525252', '#F24040', '#1A70DE', '#38AD6B', '#B078DE', '#CC9900']
    point_type = ['s', 'o', '^', 'v', 'D', '<']
    subplot_loc = [(2, 3, 1), (2, 3, 2), (2, 3, 3), (2, 3, 4), (2, 3, 5), (2, 3, 6)]
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    font = {
        'family': 'simhei',
        'weight': 'bold',
        'size': 10,
    }
    font2 = {
        'family': 'simhei',
        'weight': 'bold',
        'size': 20,
    }
    a1 = plt.figure(figsize=(15, 8))
    for i in range(len(graph_type)):
        plt.subplot(subplot_loc[i][0], subplot_loc[i][1], subplot_loc[i][2])
        for j in range(len(link_type1)):
            y = []
            yerror = []
            path = 'F:/论文/lp-new/link-prediction/results/probe_size/' + graph_type[i] + '/' + link_type1[j]
            f = open(path + '_auc.txt', 'r')
            lines = f.readlines()
            f.close()
            for line in lines:
                item = line.strip('\n')
                y.append(float(item))

            f = open(path + '_auc_errorbar.txt', 'r')
            lines = f.readlines()
            f.close()
            for line in lines:
                item = line.strip('\n')
                yerror.append(float(item))

            plt.plot(x, y, color=color_list[j], marker=point_type[j], linestyle='-')
            plt.xlim(0, 0.5)
            #plt.ylim(0.25, 1)
            if graph_type[i] == 'new FW2':
                plt.ylim(0.45, 1)
            elif graph_type[i] == 'new FW3':
                plt.ylim(0.45, 1)
            elif graph_type[i] == 'sampled SmaGri' or graph_type[i] == 'new FW1':
                plt.ylim(0.35, 0.95)
            elif graph_type[i] == 'sampled Delicious':
                plt.ylim(0.25, 1)
            else:
                plt.ylim(0.3, 0.9)
            plt.xticks([0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45])
            plt.tick_params(labelsize=11)

            plt.xlabel('the size of probe set', font2, labelpad=8)
            plt.ylabel('AUC', font2, labelpad=10)
            plt.errorbar(x, y, color=color_list[j], marker=point_type[j], linestyle='-', yerr=yerror, capsize=3)
            plt.title(lab_list[i], font2, x=-0.17)

    plt.subplots_adjust(wspace=0.3, hspace=0.3, left=0.07, right=0.98, bottom=0.06, top=0.95)
    plt.show()


def plot_alphachange(type):
    graph_type = ['new FW1', 'new FW2', 'new FW3', 'sampled SmaGri', 'sampled Kohonen', 'sampled Delicious']
    probe_list = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45]
    alpha_list = [8, 6, 4, 2, 0, -2, -4, -6, -8]
    lab_list = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)']
    subplot_loc = [(2, 3, 1), (2, 3, 2), (2, 3, 3), (2, 3, 4), (2, 3, 5), (2, 3, 6)]
    # color2 = LinearSegmentedColormap.from_list("", ['#2D2DE0', '#FFFFFF', '#DE2727'])
    color2 = LinearSegmentedColormap.from_list("", ['#4791C5', '#FFFFFF', '#DE6C5B'])
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    font = {
        'family': 'simhei',
        'weight': 'bold',
        'size': 10,
    }
    font2 = {
        'family': 'simhei',
        'weight': 'bold',
        'size': 20,
    }

    a1 = plt.figure(figsize=(15, 8))
    for i in range(len(graph_type)):
        plt.subplot(subplot_loc[i][0], subplot_loc[i][1], subplot_loc[i][2])
        data = []
        for j in range(len(alpha_list)):
            store = []
            for k in range(len(probe_list)):
                path = 'F:/论文/lp-new/link-prediction/' + graph_type[i] + '/alpha' + str(alpha_list[j]) + '/' + type + \
                       '/probe_size' + str(probe_list[k])
                f = open(path + '/average_auc.txt', 'r')
                lines = f.readlines()
                f.close()
                store.append(float(lines[0]))
            data.append(store)
        data = pd.DataFrame(data)
        if type == "dis":
            sns.heatmap(data, robust=True, vmin=0.2, vmax=0.8, cmap=color2, annot_kws={"front_size": 11})
        else:
            sns.heatmap(data, robust=True, cmap=color2, annot_kws={"front_size": 15})
        plt.xticks([i+0.5 for i in range(9)], [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45])
        plt.yticks([i+0.5 for i in range(9)], [8, 6, 4, 2, 0, -2, -4, -6, -8], rotation='horizontal')
        plt.xlabel('The size of probe set', font2, labelpad=8)
        plt.ylabel(r'$\alpha$', font2, labelpad=10)
        plt.tick_params(labelsize=11)
        plt.title(lab_list[i], font2, x=-0.17)

    plt.subplots_adjust(wspace=0.2, hspace=0.3, left=0.05, right=0.98, bottom=0.06, top=0.95)
    plt.show()


def gini_coef(wealths):
    cum_wealths = np.cumsum(sorted(np.append(wealths, 0)))
    sum_wealths = cum_wealths[-1]
    xarray = np.array(range(0, len(cum_wealths))) / (len(cum_wealths)-1)
    yarray = cum_wealths / sum_wealths
    B = np.trapz(yarray, x=xarray)
    A = 0.5 - B
    return A / (A+B)


def plot_graph():
    graph_type = ['new FW1', 'new FW2', 'new FW3', 'sampled SmaGri', 'sampled Kohonen', 'sampled Delicious']
    font = {
        'family': 'simhei',
        'weight': 'bold',
        'size': 30,
    }
    font2 = {
        'family': 'simhei',
        'weight': 'bold',
        'size': 20,
    }
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    lab_list = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)']
    subplot_loc = [(2, 3, 1), (2, 3, 2), (2, 3, 3), (2, 3, 4), (2, 3, 5), (2, 3, 6)]

    for i in range(len(graph_type)):
        graph_name = graph_type[i]
        G = nx.DiGraph()
        f = open('F:/论文/link prediction/distance_summary/distance_summary/6个数据集/' + graph_name + '.txt', 'r')
        lines = f.readlines()
        f.close()
        for line in lines:
            item = line.strip('\n').split(' ')
            if item[0] != item[1]:
                G.add_edge(item[0], item[1])
        N = nx.number_of_nodes(G)
        M = nx.number_of_edges(G)
        d_dict = dict(nx.all_pairs_dijkstra_path_length(G))
        d = 0
        count = 0
        for key1 in d_dict:
            for key2 in d_dict[key1]:
                if key1 != key2:
                    count += 1
                    d += d_dict[key1][key2]
        d = d / count

        midu = nx.density(G)
        sum_deg = 0
        degree_list = []
        degree_distribution = [0 for i in range(N)]
        distance_distribution = [0 for i in range(N)]
        for node in G.nodes():
            z = G.out_degree(node)
            degree_list.append(z)
            sum_deg += z
            degree_distribution[z] += 1
        a_k = sum_deg / N
        for item in range(N):
            degree_distribution[item] = degree_distribution[item] / N

        # 计算初始距离一致性
        distance_list = []
        dis_dir = nx.all_pairs_bellman_ford_path_length(G)
        for node1, node1_dic in dis_dir:
            for dis2 in node1_dic.values():
                distance_list.append(dis2)
                distance_distribution[dis2] += 1
        for item in range(N):
            distance_distribution[item] = distance_distribution[item] / (N**2)

        deg_gnin = gini_coef(degree_list)
        dis_gnin = gini_coef(distance_list)
        c = nx.average_clustering(G)
        print(graph_name, N, M, a_k, d, midu, c)

        x = [i for i in range(N)]
        plt.subplot(subplot_loc[i][0], subplot_loc[i][1], subplot_loc[i][2])
        # print(graph_type[i])
        # print('------------------------------------')
        # print('deg:  ', degree_distribution)
        # print('dis:  ', distance_distribution)
        plt.plot(x, degree_distribution, color='#DC543F', linestyle='-', marker='o', markersize=3)
        plt.plot(x, distance_distribution, color='#3585BC', linestyle='-', marker='s', markersize=3)

        plt.xlim(-5, 60)
        plt.tick_params(labelsize=15)
        plt.xlabel('x', font, labelpad=8)
        plt.ylabel('P(x)', font2, labelpad=8)
        plt.title(lab_list[i], font2, x=-0.2)
    plt.subplots_adjust(wspace=0.35, hspace=0.25, left=0.07, right=0.98, bottom=0.1, top=0.95)
    plt.show()


def rewrite_alphachange(type):
    graph_type = ['new FW1']
    probe_list = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45]
    alpha_list = [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]
    for prob in probe_list:
        store = []
        for alpha in alpha_list:
            path = 'D:/论文/link prediction/final_data/new FW1/alpha' + str(alpha) + '/' + type + '/probe_size' + str(prob)
            f = open(path + '/average_auc.txt', 'r')
            lines = f.readlines()
            f.close()
            store.append(lines[0])
        path1 = 'D:/论文/link prediction/final_data/results/heatmap/probe_size' + str(prob)
        folder = os.path.exists(path1)
        if not folder:
            os.makedirs(path1)
        f = open(path1 + 'average_auc.txt', 'w')
        for s in store:
            f.write(s + '\n')
        f.close()


def structural_distribution_consistency(feature_previous, feature_now, alpha):
    feature_dif = []
    add_index = 0
    sum1 = sum(feature_previous)
    sum2 = sum(feature_now)
    for i in range(len(feature_now)):
        feature_dif.append(np.exp(-alpha * i) * (feature_previous[i] * sum2 - feature_now[i] * sum1))
    feature_structural = abs(sum(feature_dif) / (sum1 * sum2))
    return feature_structural


def compute_STM_dis(N, all_set, distance_feature_previous_list, add_edge, alpha):
    distance_feature_now_list = [0 for i in range(N)]

    train_graph = nx.DiGraph()
    for edge2 in all_set:
        train_graph.add_edge(edge2[0], edge2[1])

    train_graph.add_edge(add_edge[0], add_edge[1])
    train_node_list = []
    for node in train_graph.nodes():
        train_node_list.append(node)
    for node1_length in range(len(train_node_list)):
        for node2_length in range(len(train_node_list)):
            if nx.has_path(train_graph, train_node_list[node1_length], train_node_list[node2_length]):
                dis = nx.dijkstra_path_length(train_graph, train_node_list[node1_length], train_node_list[node2_length])
                distance_feature_now_list[dis] += 1
    distance_s = structural_distribution_consistency(distance_feature_previous_list, distance_feature_now_list, alpha)
    return distance_s


import math
def compute_STM(feature_previous, feature_now, alpha, type):
    feature_dif = []
    N = len(feature_previous)
    for i in range(len(feature_now)):
        if type == 'deg':
            z_value = feature_previous[i] / N - feature_now[i] / N
        elif type == 'dis':
            z_value = feature_previous[i] / (N*N) - feature_now[i] / (N*N)
        if z_value != 0:
            if -700 < -alpha*i < 700:
                exp_value = math.exp(-alpha * i)
            elif -alpha*i <= -700:
                exp_value = 0
            else:
                exp_value = np.inf
            feature_dif.append(exp_value * z_value)
        else:
            feature_dif.append(0)
    feature_structural = abs(sum(feature_dif))
    return feature_structural


def compute_auc_and_omega():
    graph_list = ['new FW1', 'new FW2', 'new FW3', 'sampled SmaGri', 'sampled Kohonen', 'sampled Delicious']
    probe_list = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45]
    for graph_name in graph_list:
        store_dis = []
        store_deg = []
        for probe in probe_list:
            print(graph_name, probe)
            G = nx.DiGraph()
            f = open('F:/论文/link prediction/distance_summary/distance_summary/6个数据集/' + graph_name + '.txt', 'r')
            lines = f.readlines()
            f.close()
            edge_set = []
            node_set = []
            for line in lines:
                item = line.strip('\n').split(' ')
                if item[0] != item[1]:
                    G.add_edge(item[0], item[1])
                    edge_set.append((item[0], item[1]))
            for node in G.nodes():
                node_set.append(node)
            M = nx.number_of_edges(G)
            N = nx.number_of_nodes(G)

            distance_feature_now_list = [0 for i in range(N)]
            dis_dir = nx.all_pairs_bellman_ford_path_length(G)
            for node1, node1_dic in dis_dir:
                for dis2 in node1_dic.values():
                    distance_feature_now_list[dis2] += 1

            degree_feature_now_list = [0 for i in range(N)]
            for node1 in G.nodes():
                degree_feature_now_list[G.out_degree(node1)] += 1

            test_number = int(M * probe)
            store1_dis = []
            store1_deg = []
            for z in range(1):
                # print(z)
                store2 = []
                test_set = random.sample(edge_set, test_number)
                train_set = []
                train_graph = nx.DiGraph()
                for edge1 in edge_set:
                    if edge1 not in test_set:
                        train_set.append(edge1)
                        train_graph.add_edge(edge1[0], edge1[1])
                print(len(edge_set), len(train_set))
                path = 'F:/论文/link prediction/distance_summary/' + str(probe) + '/'
                folder = os.path.exists(path)
                if not folder:
                    os.makedirs(path)
                f = open(path + graph_name + '_train.txt', 'w')
                for edge in train_set:
                    f.write(edge[0] + ' ' + edge[1] + '\n')
                f.close()

                distance_feature_previous_list = [0 for i in range(N)]
                dis_dir = nx.all_pairs_bellman_ford_path_length(train_graph)
                for node1, node1_dic in dis_dir:
                    for dis2 in node1_dic.values():
                        distance_feature_previous_list[dis2] += 1

                degree_feature_previous_list = [0 for i in range(N)]
                for node1 in train_graph.nodes():
                    degree_feature_previous_list[train_graph.out_degree(node1)] += 1
                if z == 0:
                    print(graph_name)
                    print('------------------------------------')
                    print('average_shortest_path_length:  ', )
                score_test_dis = compute_STM(distance_feature_previous_list, distance_feature_now_list, 1, 'dis')
                score_test_deg = compute_STM(degree_feature_previous_list, degree_feature_now_list, 1, 'deg')

                store1_dis.append(score_test_dis)
                store1_deg.append(score_test_deg)
            store_dis.append(sum(store1_dis) / 20)
            store_deg.append(sum(store1_deg) / 20)
        # path1 = 'F:/论文/link prediction/temp_data/results/relationship/deg_ver1/'
        # folder = os.path.exists(path1)
        # if not folder:
        #     os.makedirs(path1)
        # path2 = 'F:/论文/link prediction/temp_data/results/relationship/dis/'
        # folder = os.path.exists(path2)
        # if not folder:
        #     os.makedirs(path2)
        #
        # f = open(path1 + graph_name + 'relation.txt', 'w')
        # for i in store_deg:
        #     f.write(str(i) + '\n')
        # f.close()
        # f = open(path2 + graph_name + 'relation.txt', 'w')
        # for i in store_dis:
        #     f.write(str(i) + '\n')
        # f.close()


#新版重新排列
def rewrite_auc_and_omega_data(type):
    graph_list = ['new FW1', 'new FW2', 'new FW3', 'sampled SmaGri', 'sampled Kohonen', 'sampled Delicious']
    probe_list = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45]
    for graph_name in graph_list:
        if type == 'deg':
            # path = 'F:/论文/lp-new/link-prediction/' + graph + '/alpha2/' + link_type + '/probe_size' + str(ps)
            if graph_name == 'sampled SmaGri' or graph_name == 'sampled Kohonen' or graph_name == 'sampled Delicious':
                path = 'F:/论文/lp-new/link-prediction/' + graph_name + '/alpha8/' + type + '/'
            else:
                path = 'F:/论文/lp-new/link-prediction/' + graph_name + '/alpha2/' + type + '/'
        elif type == 'dis':
            # path = 'F:/论文/lp-new/link-prediction/' + graph + '/alpha-2/' + link_type + '/probe_size' + str(ps)
            if graph_name == 'sampled SmaGri' or graph_name == 'sampled Kohonen' or graph_name == 'sampled Delicious':
                path = 'F:/论文/lp-new/link-prediction/' + graph_name + '/alpha-8/' + type + '/'
            else:
                path = 'F:/论文/lp-new/link-prediction/' + graph_name + '/alpha-2/' + type + '/'

        #path = 'F:/论文/lp-new/link-prediction/' + graph_name + '/alpha8/' + type + '/'
        omega = []
        for prb in probe_list:
            if type == 'deg_ver1':
                type1 = 'deg'
            else:
                type1 = type
            f = open(path + 'probe_size' + str(prb) + '/omega_' + type1 + '.txt', 'r')
            lines = f.readlines()
            f.close()
            omega.append(lines[0])

        path1 = 'F:/论文/lp-new/link-prediction/results/relationship/' + str(type) + '/' + graph_name
        folder = os.path.exists(path1)
        if not folder:
            os.makedirs(path1)
        f = open(path1 + 'relation.txt', 'w')
        for i in omega:
            f.write(str(i) + '\n')
        f.close()

# 线性拟合
def fline(x, A, B):
  return A*x + B


def plot_relationship():
    font = {
        'family': 'simhei',
        'weight': 'bold',
        'size': 10,
    }
    font2 = {
        'family': 'simhei',
        'weight': 'bold',
        'size': 20,
    }
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    lab_list = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)']
    subplot_loc = [(2, 3, 1), (2, 3, 2), (2, 3, 3), (2, 3, 4), (2, 3, 5), (2, 3, 6)]
    graph_type = ['new FW1', 'new FW2', 'new FW3', 'sampled SmaGri', 'sampled Kohonen', 'sampled Delicious']
    for j in range(len(graph_type)):
        x = []
        y = []
        if j < 3:
            f = open(
                'F:/论文/lp-new/link-prediction/results/relationship/dis/' + graph_type[j] + 'relation.txt',
                'r')
        else:
            f = open(
                'F:/论文/lp-new/link-prediction/results/relationship/deg/' + graph_type[j] + 'relation.txt', 'r')
        lines = f.readlines()
        f.close()
        for line in lines:
            item = line.strip('\n')
            x.append(float(item))
        if j < 3:
            f = open('F:/论文/lp-new/link-prediction/results/probe_size/' + graph_type[j] + '/dis_auc.txt', 'r')
        else:
            f = open('F:/论文/lp-new/link-prediction/results/probe_size/' + graph_type[j] + '/deg_auc.txt', 'r')
        lines = f.readlines()
        f.close()
        for line in lines:
            item = line.strip('\n')
            y.append(float(item))
        plt.subplot(subplot_loc[j][0], subplot_loc[j][1], subplot_loc[j][2])
        plt.scatter(x, y, color='#000000', s=100)
        A1, B1 = optimize.curve_fit(fline, x, y)[0]

        x1 = np.array(x)
        y1 = A1 * x1 + B1

        x2 = np.array(x)
        y2 = A1 * x2 + B1

        plt.plot(x1, y1, "#609BD4", linewidth=4.0)
        # plt.ticklabel_format(axis='x', style='sci', scilimits=(0, 0))
        plt.tick_params(labelsize=15)
        # sst = np.sum(np.power(y - np.average(y), 2))
        # ssr = np.sum(np.power(y2 - np.average(y), 2))
        # r_2 = ssr / sst
        if j < 3:
            # if graph_type[j] == 'new FW2':
            #     plt.ylim(0.81, 0.84)
            plt.xlabel(r'$\delta_{dis}$', font2, labelpad=8)
            plt.ylabel(r'AUC of SCM$_{dis}$', font2, labelpad=8)
        else:
            plt.xlabel(r'$\delta_{deg}$', font2, labelpad=8)
            plt.ylabel(r'AUC of SCM$_{deg}$', font2, labelpad=8)
        sse = np.sum(np.power(y - y2, 2))
        plt.title(lab_list[j], font2, x=-0.25)
        if j == 1:
            plt.xlim(1.8, 4)
        if j == 2:
            plt.xlim(2, 6)
        if j == 3 or j==4:
            plt.ticklabel_format(axis="x", style="sci", scilimits=(0,0))
        if j == 5:
            plt.xlim(0.0002, 0.00055)
            plt.ticklabel_format(axis="x", style="sci", scilimits=(0,0))
        print(graph_type[j], A1, B1, sse)
    plt.subplots_adjust(wspace=0.35, hspace=0.25, left=0.07, right=0.98, bottom=0.1, top=0.95)
    plt.show()


def plot_ba_deg(graph):
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    subplot_loc = [(2, 4, 1), (2, 4, 2), (2, 4, 3), (2, 4, 4), (2, 4, 5), (2, 4, 6), (2, 4, 7), (2, 4, 8)]
    font2 = {
        'family': 'simhei',
        'weight': 'bold',
        'size': 20,
    }
    lab_list = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)', '(g)', '(h)']
    degree_type = [1, 2, 3, 4, 1, 2, 3, 4]
    cm1 = [0.76, 0.61, 0.57, 0.55, 0.76, 0.61, 0.57, 0.55,]
    cm2 = [2.37, 2.6, 2.7, 2.8, 2.37, 2.6, 2.7, 2.8]
    # color_list2 = ['#609BD4', '#F9A34E', '#038303', '#790578']
    ###color_list1 = ['#609BD4', '#F08082', '#7F007F', '#FFA400']
    #color_list2 = ['#75C29C', '#F6A248', '#CBBCE1', '#F3A1C2']
    #F27E35 '#427FA7'
    ##color_list1 = ['#DB6A6D', '#4E5D9B', '#E4DB3C', '#41BBBC']
    #color_list2 = ['#799EC7', '#B55750', '#9DB56A', '#7F659D']
    color_list1 = ['#D87DE5', '#F9A34E', '#038303', '#7F659D']
    color_list2 = ['#676767', '#42B497', '#D74B0C', '#3176B8']

    alpha_list1 = [-8, -6, -4, -2]
    alpha_list2 = [2, 4, 6, 8]
    for j in range(len(subplot_loc)):
        if j > 3:
            L = 4
        else:
            L = 4
        for i in range(L):
            y = []
            if j < 4:
                path = 'F:/论文/lp-new/' + graph + '/average_degree' + str(2 * degree_type[j]) \
                       + '/' + str(alpha_list1[i])
            else:
                path = 'F:/论文/lp-new/' + graph + '/average_degree' + str(2 * degree_type[j]) \
                       + '/' + str(alpha_list2[i])
            f = open(path + '/deg_avg.txt', 'r')
            lines = f.readlines()
            f.close()
            for line in lines:
                item = line.strip('\n')
                y.append(float(item))
            plt.subplot(subplot_loc[j][0], subplot_loc[j][1], subplot_loc[j][2])

            x = [k for k in range(len(y)-1)]
            if j < 4:
                plt.plot(x, y[:-1], color=color_list1[i], linewidth=2)
                plt.yscale('log')
                plt.xscale('log')
            else:
                plt.plot(x, y[:-1], color=color_list2[i], linewidth=2)
                plt.xscale('log')
                plt.yscale('log')
            plt.xlim(1, 1050)

            if j < 4:
                 y_fit = [(degree_type[j])**(2 - alpha_list1[i] - cm2[j]) * (k) ** ((0.5)*(- alpha_list1[i] - cm2[j])) for k in range(1, len(y))]
                 plt.plot(x, y_fit, color=color_list1[i], linestyle='--', linewidth=2)
            else:
                 y_fit = [(degree_type[j])**(2 - alpha_list2[i] - cm2[j]) * (k) ** (-1) for k in range(1, len(y))]
                 plt.plot(x, y_fit, color=color_list2[i], linestyle='--', linewidth=2)

            # if j < 4:
            #      y_fit = [(degree_type[j])**(2 - alpha_list1[i] - 3) * (k) ** ((0.5)*(- alpha_list1[i] - 4)) for k in range(1, len(y))]
            #      plt.plot(x, y_fit, color=color_list1[i], linestyle='--', linewidth=2)
            # else:
            #      y_fit = [(degree_type[j])**(2 - alpha_list2[i] - 3) * (k) ** (-1) for k in range(1, len(y))]
            #      plt.plot(x, y_fit, color=color_list2[i], linestyle='--', linewidth=2)
            # if j < 4:
            #      y_fit = np.log10()
            #      y_fit = [degree_type[j]**(2) * (k+1) ** (0.5*(- alpha_list1[i] - 2.6) - alpha_list1[i] / 2 - 0.5) for k in range(len(y) - 1)]
            #      plt.plot(x, y_fit, color=color_list1[i], linestyle='--', linewidth=2)
            # else:
            #     y_fit = [np.exp(-((8+degree_type[j])*(degree_type[j] - 2) / (8 * degree_type[j])) * alpha_list2[i]) * (k+1) ** (-1) for k in range(len(y) - 1)]
            #     plt.plot(x, y_fit, color=color_list2[i], linestyle='--', linewidth=2)

        plt.xlabel('t', font2, labelpad=8)
        plt.ylabel(r'$\Delta_{\alpha}^{D}$', font2, labelpad=8)
        plt.title(lab_list[j], font2, x=-0.2)
        plt.tick_params(labelsize=12)

    plt.subplots_adjust(wspace=0.4, hspace=0.3, left=0.07, right=0.99, bottom=0.1, top=0.95)
    plt.show()


def plot_ba_dis(graph):
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    subplot_loc = [(2, 4, 1), (2, 4, 2), (2, 4, 3), (2, 4, 4), (2, 4, 5), (2, 4, 6), (2, 4, 7), (2, 4, 8)]
    font2 = {
        'family': 'simhei',
        'weight': 'bold',
        'size': 20,
    }
    lab_list = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)', '(g)', '(h)']
    degree_type = [2, 4, 6, 8, 2, 4, 6, 8]
    # color_list2 = ['#609BD4', '#F9A34E', '#038303', '#790578']
    ###color_list1 = ['#609BD4', '#F08082', '#7F007F', '#FFA400']
    ## color_list2 = ['#75C29C', '#F6A248', '#CBBCE1', '#F3A1C2']
    # F27E35 '#427FA7'
    ##color_list1 = ['#DB6A6D', '#4E5D9B', '#E4DB3C', '#41BBBC']
    # color_list2 = ['#799EC7', '#B55750', '#9DB56A', '#7F659D']
    color_list1 = ['#D87DE5', '#F9A34E', '#038303', '#7F659D']
    color_list2 = ['#676767', '#42B497', '#D74B0C', '#3176B8']
    alpha_list1 = [-8, -6, -4, -2]
    alpha_list2 = [2, 4, 6, 8]
    for j in range(len(subplot_loc)):
        if j > 3:
            L = 4
        else:
            L = 4
        for i in range(L):
            y = []
            if j < 4:
                path = 'F:/论文/lp-new/' + graph + '/average_degree' + str(degree_type[j]) \
                       + '/' + str(alpha_list1[i])
            else:
                path = 'F:/论文/lp-new/' + graph + '/average_degree' + str(degree_type[j]) \
                       + '/' + str(alpha_list2[i])
            f = open(path + '/dis_avg.txt', 'r')
            lines = f.readlines()
            f.close()
            for line in lines:
                item = line.strip('\n')
                y.append(float(item))
            plt.subplot(subplot_loc[j][0], subplot_loc[j][1], subplot_loc[j][2])

            x = [k for k in range(len(y) - 1)]
            if j < 4:
                plt.plot(x, y[:-1], color=color_list1[i])
                plt.yscale('log')
                plt.xscale('log')
            else:
                plt.plot(x, y[:-1], color=color_list2[i])
                plt.xscale('log')
                plt.yscale('log')
            #plt.xlim(-50, 1050)
        # if j == 0:
        #     plt.ylim(0.0001, 20)
        plt.xlabel('t', font2, labelpad=8)
        plt.ylabel(r'$\Delta_{\alpha}^{L}$', font2, labelpad=8)
        plt.title(lab_list[j], font2, x=-0.2)
        plt.tick_params(labelsize=12)
    plt.subplots_adjust(wspace=0.4, hspace=0.3, left=0.07, right=0.99, bottom=0.1, top=0.95)
    plt.show()


def plot_ws_deg(graph):
    subplot_loc = [(2, 4, 1), (2, 4, 2), (2, 4, 3), (2, 4, 4), (2, 4, 5), (2, 4, 6), (2, 4, 7), (2, 4, 8)]
    font2 = {
        'family': 'simhei',
        'weight': 'bold',
        'size': 20,
    }
    lab_list = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)', '(g)', '(h)']
    rep_type = [0.01, 0.02, 0.05, 0.1, 0.25, 0.5, 0.75, 1]
    # color_list = ['#676767', '#42B497', '#D74B0C', '#3176B8', '#A4B3D5', '#FDA481', '#85CEB7', '#C2C2C2']
    # color_list1 = ['#525252', '#F24040', '#1A70DE', '#38AD6B']
    # color_list2 = ['#E370FF', '#82D1FF', '#FF8000', '#A1E533']
    # color_list2 = ['#666666', '#F176A8', '#6BC6AC', '#9C83B7', '#F9A34E']

    # color_list2 = ['#609BD4', '#F9A34E', '#038303', '#790578']
    #color_list2 = ['#799EC7', '#B55750', '#9DB56A', '#7F659D']
    #color_list = ['#676767', '#42B497', '#D74B0C', '#3176B8', '#609BD4', '#F9A34E', '#038303', '#7F659D']

    # color_list = ['#D87DE5', '#F9A34E', '#038303', '#7F659D', '#42B497', '#D74B0C', '#3176B8', '#676767']
    color_list = ['#D87DE5', '#F9A34E', '#038303', '#7F659D', '#676767', '#3176B8', '#D74B0C', '#42B497']
    alpha_list = [-8, -6, -4, -2, 8, 6, 4, 2]
    for j in range(len(subplot_loc)):
        plt.subplot(subplot_loc[j][0], subplot_loc[j][1], subplot_loc[j][2])
        for i in range(8):
            y = []
            path = 'F:/论文/lp-new/' + graph + '/re_probability' + str(rep_type[j]) \
                    + '/' + str(alpha_list[i])
            f = open(path + '/deg_avg.txt', 'r')
            lines = f.readlines()
            f.close()
            for line in lines:
                item = line.strip('\n')
                if float(item) != 0:
                    y.append(float(item))

            x = [k for k in range(len(y)-1)]
            plt.plot(x, y[:-1], color=color_list[i])
            plt.yscale('log')
            plt.xscale('log')
            plt.xlim(1, 2000)


        plt.xlabel('t', font2, labelpad=8)
        plt.ylabel(r'$\delta_{deg}$', font2, labelpad=8)
        plt.title(lab_list[j], font2, x=-0.2)
        plt.tick_params(labelsize=12)
    plt.subplots_adjust(wspace=0.4, hspace=0.3, left=0.07, right=0.99, bottom=0.1, top=0.95)
    plt.show()


def plot_ws_dis(graph):
    subplot_loc = [(2, 4, 1), (2, 4, 2), (2, 4, 3), (2, 4, 4), (2, 4, 5), (2, 4, 6), (2, 4, 7), (2, 4, 8)]
    font2 = {
        'family': 'simhei',
        'weight': 'bold',
        'size': 20,
    }
    lab_list = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)', '(g)', '(h)']
    rep_type = [0.01, 0.02, 0.05, 0.1, 0.25, 0.5, 0.75, 1]
    color_list = ['#676767', '#42B497', '#D74B0C', '#3176B8', '#A4B3D5', '#FDA481', '#85CEB7', '#C2C2C2']
    alpha_list = [-1, -0.75, -0.5, -0.25, 0.25, 0.5, 0.75, 1]
    # color_list2 = ['#A4B3D5', '#FDA481', '#85CEB7', '#C2C2C2']
    # color_list1 = ['#676767', '#42B497', '#D74B0C', '#3176B8']
    ##color_list1 = ['#525252', '#F24040', '#1A70DE', '#38AD6B']
    ## color_list2 = ['#E370FF', '#82D1FF', '#FF8000', '#A1E533']

    # color_list2 = ['#609BD4', '#F9A34E', '#038303', '#790578']
    # color_list2 = ['#799EC7', '#B55750', '#9DB56A', '#7F659D']
    color_list1 = ['#D87DE5', '#F9A34E', '#038303', '#7F659D']
    color_list2 = ['#676767', '#42B497', '#D74B0C', '#3176B8']

    alpha_list1 = [-8, -6, -4, -2]
    alpha_list2 = [2, 4, 6, 8]
    for j in range(len(subplot_loc)):
        # if j > 3:
        #     L = 5
        # else:
        #     L = 4
        # for i in range(8):
        for i in range(4):
            y = []
            # if j < 4:
            #     path = 'F:/论文/link prediction/final_data/' + graph + '/re_probability' + str(rep_type[j]) \
            #            + '/' + str(alpha_list1[i])
            # else:
            #     path = 'F:/论文/link prediction/final_data/' + graph + '/re_probability' + str(rep_type[j]) \
            #            + '/' + str(alpha_list2[i])
            path = 'F:/论文/lp-new/' + graph + '/re_probability' + str(rep_type[j]) \
               + '/' + str(alpha_list2[i])
            f = open(path + '/dis_avg.txt', 'r')
            lines = f.readlines()
            f.close()
            for line in lines:
                item = line.strip('\n')
                if float(item) != 0:
                    y.append(float(item))
            plt.subplot(subplot_loc[j][0], subplot_loc[j][1], subplot_loc[j][2])

            x = [k for k in range(len(y) - 1)]
            plt.plot(x, y[:-1], color=color_list2[i])
            plt.yscale('log')
            plt.xscale('log')
            plt.xlim(-50, 1050)
        # if j == 0:
        #     plt.ylim(0.0001, 20)
        plt.xlabel('t', font2, labelpad=8)
        plt.ylabel(r'$\delta_{dis}$', font2, labelpad=8)
        plt.title(lab_list[j], font2, x=-0.2)
        plt.tick_params(labelsize=12)

    plt.subplots_adjust(wspace=0.4, hspace=0.3, left=0.07, right=0.99, bottom=0.1, top=0.95)
    plt.show()


def plot_ws_insert_dis(graph):
    font2 = {
        'family': 'simhei',
        'weight': 'bold',
        'size': 20,
    }
    lab_list = []
    rep_type = [0.01, 0.02, 0.05, 0.1, 0.25, 0.5, 0.75, 1]
    color_list2 = ['#609BD4', '#F9A34E', '#038303', '#790578']
    # color_list2 = ['#799EC7', '#B55750', '#9DB56A', '#7F659D']
    color_list = ['#676767', '#42B497', '#D74B0C', '#3176B8']
    alpha_list2 = [0.25, 0.5, 0.75, 1]
    for j in range(8):
        for i in range(4):
            y = []
            path = 'F:/论文/link prediction/final_data/' + graph + '/re_probability' + str(rep_type[j]) \
               + '/' + str(alpha_list2[i])
            f = open(path + '/dis_avg.txt', 'r')
            lines = f.readlines()
            f.close()
            for line in lines:
                item = line.strip('\n')
                if float(item) != 0:
                    y.append(float(item))
            x = [k for k in range(len(y) - 1)]
            plt.plot(x, y[:-1], color=color_list2[i], linewidth=10)
            plt.yscale('log')
            # plt.xlim(-50, 1050)
        # if j == 0:
        #     plt.ylim(0.0001, 20)
        # plt.xlabel('t', font2, labelpad=8)
        # plt.ylabel(r'$\delta_{dis}$', font2, labelpad=8)
        plt.tick_params(labelsize=12)
        plt.show()

# 提取samgri网络
def sample_samgri():
    f = open('E:/安装包/SmaGri/SmaGri.net', 'r')
    lines = f.readlines()
    f.close()
    z = {}
    for item in lines:
        count = 0
        loc_node = 0
        line = item.strip('\n').split(' ')
        for item2 in line:
            if item2 != '' and count == 1:
                z[loc_node].append(int(item2))
            elif item2 != '' and count == 0:
                loc_node = item2
                z[loc_node] = []
                count = 1
    D = nx.DiGraph()
    for key in z.keys():
        for key2 in z[key]:
            D.add_edge(str(key), str(key2))

    node_list = list(D.nodes())
    count = 2
    choose_node = []
    for node in node_list:
        if node not in choose_node:
            choose_node.append(node)
        dfs = nx.dfs_successors(D, node)
        for value in dfs.values():
            for value1 in value:
                if count > 300:
                    break
                else:
                    if value1 not in choose_node:
                        choose_node.append(value1)
                    count += 1
        if count > 300:
            break
            nx.dfs_predecessors()
    E = nx.DiGraph()
    for edge in D.edges():
        if edge[0] in choose_node and edge[1] in choose_node:
            if edge[0] != edge[1]:
                E.add_edge(edge[0], edge[1])
    f = open('D:/论文/link prediction/distance_summary/distance_summary/6个数据集/sampled SmaGri.txt', 'w')
    for edge in E.edges():
        f.write(edge[0] + ' ' + edge[1] + '\n')
    f.close()


def cluster_network():
    cluster_type = ['C1', 'C2']
    N = 300
    C1_list = [i for i in range(150)]
    C2_list = [i for i in range(150, 300)]
    edge_list = []
    count = 0
    while count < 400:
        node_pair1 = random.sample(C1_list, 2)
        node_pair2 = random.sample(C2_list, 2)
        p = random.random()
        if p <= 0.5:
            edge_list.append((node_pair1[0], node_pair1[1]))
            edge_list.append((node_pair2[0], node_pair2[1]))
        else:
            edge_list.append((node_pair1[1], node_pair1[0]))
            edge_list.append((node_pair2[1], node_pair2[0]))
        count += 1
    count = 0
    while count < 50:
        node1 = random.choice(C1_list)
        node2 = random.choice(C2_list)
        p = random.random()
        if p <= 0.5:
            edge_list.append((node1, node2))
        else:
            edge_list.append((node2, node1))
        count += 1
    f = open('F:/论文/link prediction/distance_summary/distance_summary/6个数据集/cluster_network.txt', 'w')
    for edge in edge_list:
        f.write(str(edge[0]) + ' ' + str(edge[1]) + '\n')
    f.close()


def clean_tempral_base_probesize(graph, link_type):
    store = []
    probe_size = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45]
    for ps in probe_size:
        if link_type == 'deg':
            path = 'F:/论文/lp-new/link-prediction/' + graph + '/alpha8/' + link_type + '/time_size' + str(ps) + '/auc.txt'
        elif link_type == 'dis':
            path = 'F:/论文/lp-new/link-prediction/' + graph + '/alpha8/' + link_type + '/time_size' + str(ps) + '/auc.txt'
        else:
            path = 'F:/论文/lp-new/link-prediction/' + graph + '/' + link_type + '/probe_size' + str(ps) + '/average_auc.txt'
        f = open(path, 'r')
        lines = f.readlines()
        f.close()
        store.append(lines[0])
    path1 = 'F:/论文/lp-new/link-prediction/results/probe_size/' + graph
    folder = os.path.exists(path1)
    if not folder:
        os.makedirs(path1)
    f = open(path1 + '/' + link_type + '_auc.txt', 'w')
    for i in store:
        f.write(str(i) + '\n')
    f.close()


def plot_tempral_probsize():
    graph_type = ['contacts-prox-high-school-2013', 'email-dnc', 'SFHH-conf-sensor']
    link_type1 = ['deg', 'dis', 'cn', 'pa', 'lr', 'rwr']
    x = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45]
    lab_list = ['(a)', '(b)', '(c)']

    color_list = ['#525252', '#F24040', '#1A70DE', '#38AD6B', '#B078DE', '#CC9900']
    point_type = ['s', 'o', '^', 'v', 'D', '<']
    subplot_loc = [(1, 3, 1), (1, 3, 2), (1, 3, 3)]
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    font = {
        'family': 'simhei',
        'weight': 'bold',
        'size': 10,
    }
    font2 = {
        'family': 'simhei',
        'weight': 'bold',
        'size': 20,
    }
    a1 = plt.figure(figsize=(16, 5))
    for i in range(len(graph_type)):
        plt.subplot(subplot_loc[i][0], subplot_loc[i][1], subplot_loc[i][2])
        for j in range(len(link_type1)):
            y = []
            # yerror = []

            path = 'F:/论文/lp-new/link-prediction/results/probe_size/' + graph_type[i] + '/' + link_type1[j]
            f = open(path + '_auc.txt', 'r')
            lines = f.readlines()
            f.close()
            for line in lines:
                item = line.strip('\n')
                y.append(float(item))

            # f = open(path + '_auc_errorbar.txt', 'r')
            # lines = f.readlines()
            # f.close()
            # for line in lines:
            #     item = line.strip('\n')
            #     yerror.append(float(item))

            plt.plot(x, y, color=color_list[j], marker=point_type[j], linestyle='-')
            plt.xlim(0.05, 0.5)
            plt.ylim(0.25, 1)
            # if graph_type[i] == 'new FW2':
            #     plt.ylim(0.45, 1)
            # elif graph_type[i] == 'new FW3':
            #     plt.ylim(0.45, 1)
            # elif graph_type[i] == 'sampled SmaGri' or graph_type[i] == 'new FW1':
            #     plt.ylim(0.35, 0.95)
            # elif graph_type[i] == 'sampled Delicious':
            #     plt.ylim(0.25, 1)
            # else:
            #     plt.ylim(0.3, 0.9)
            plt.xticks([0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45])
            plt.tick_params(labelsize=11)

            plt.xlabel('the size of probe set', font2, labelpad=8)
            plt.ylabel('AUC', font2, labelpad=10)
            # plt.errorbar(x, y, color=color_list[j], marker=point_type[j], linestyle='-', yerr=yerror, capsize=3)
            plt.title(lab_list[i], font2, x=-0.17)

    plt.subplots_adjust(wspace=0.3, hspace=0.3, left=0.07, right=0.98, bottom=0.15, top=0.9)
    plt.show()


def plot_tempral_alphachange(type):
    graph_type = ['contacts-prox-high-school-2013', 'email-dnc', 'SFHH-conf-sensor']
    probe_list = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45]
    alpha_list = [8, 6, 4, 2, 0, -2, -4, -6, -8]
    lab_list = ['(d)', '(e)', '(f)']
    subplot_loc = [(1, 3, 1), (1, 3, 2), (1, 3, 3)]
    # color2 = LinearSegmentedColormap.from_list("", ['#2D2DE0', '#FFFFFF', '#DE2727'])
    color2 = LinearSegmentedColormap.from_list("", ['#4791C5', '#FFFFFF', '#DE6C5B'])
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    font = {
        'family': 'simhei',
        'weight': 'bold',
        'size': 10,
    }
    font2 = {
        'family': 'simhei',
        'weight': 'bold',
        'size': 20,
    }

    a1 = plt.figure(figsize=(16, 5))
    for i in range(len(graph_type)):
        plt.subplot(subplot_loc[i][0], subplot_loc[i][1], subplot_loc[i][2])
        data = []
        for j in range(len(alpha_list)):
            store = []
            for k in range(len(probe_list)):
                path = 'F:/论文/lp-new/link-prediction/' + graph_type[i] + '/alpha' + str(alpha_list[j]) + '/' + type + \
                       '/time_size' + str(probe_list[k])
                f = open(path + '/auc.txt', 'r')
                lines = f.readlines()
                f.close()
                store.append(float(lines[0]))
            data.append(store)
        data = pd.DataFrame(data)
        if type == "dis":
            sns.heatmap(data, robust=True, vmin=0.2, vmax=0.8, cmap=color2, annot_kws={"front_size": 11})
        else:
            sns.heatmap(data, robust=True, vmin=0.2, vmax=0.8, cmap=color2, annot_kws={"front_size": 15})
        plt.xticks([i+0.5 for i in range(9)], [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45])
        plt.yticks([i+0.5 for i in range(9)], [8, 6, 4, 2, 0, -2, -4, -6, -8], rotation='horizontal')
        plt.xlabel('The size of probe set', font2, labelpad=8)
        plt.ylabel(r'$\alpha$', font2, labelpad=10)
        plt.tick_params(labelsize=11)
        plt.title(lab_list[i], font2, x=-0.17)

    plt.subplots_adjust(wspace=0.2, hspace=0.3, left=0.05, right=0.98, bottom=0.15, top=0.9)
    plt.show()


def rewrite_tempral_auc_and_omega_data(type):
    graph_list = ['contacts-prox-high-school-2013', 'email-dnc', 'SFHH-conf-sensor']
    probe_list = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45]
    for graph_name in graph_list:
        path = 'F:/论文/lp-new/link-prediction/' + graph_name + '/alpha8/' + type + '/'

        omega = []
        for prb in probe_list:
            if type == 'deg_ver1':
                type1 = 'deg'
            else:
                type1 = type
            f = open(path + 'time_size' + str(prb) + '/omega_' + type1 + '.txt', 'r')
            lines = f.readlines()
            f.close()
            omega.append(lines[0])

        path1 = 'F:/论文/lp-new/link-prediction/results/relationship/' + str(type) + '/' + graph_name
        folder = os.path.exists(path1)
        if not folder:
            os.makedirs(path1)
        f = open(path1 + 'relation.txt', 'w')
        for i in omega:
            f.write(str(i) + '\n')
        f.close()


def plot_tempral_relationship():
    font = {
        'family': 'simhei',
        'weight': 'bold',
        'size': 10,
    }
    font2 = {
        'family': 'simhei',
        'weight': 'bold',
        'size': 20,
    }
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    lab_list = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)']
    subplot_loc = [(1, 3, 1), (1, 3, 2), (1, 3, 3)]
    a1 = plt.figure(figsize=(16, 5))
    graph_type = ['contacts-prox-high-school-2013', 'email-dnc', 'SFHH-conf-sensor']
    for j in range(len(graph_type)):
        x = []
        y = []
        if j == 0 or j == 2:
            f = open(
                'F:/论文/lp-new/link-prediction/results/relationship/dis/' + graph_type[j] + 'relation.txt',
                'r')
        else:
            f = open(
                'F:/论文/lp-new/link-prediction/results/relationship/deg/' + graph_type[j] + 'relation.txt', 'r')
        lines = f.readlines()
        f.close()
        for line in lines:
            item = line.strip('\n')
            x.append(float(item))
        if j == 0 or j == 2:
            f = open('F:/论文/lp-new/link-prediction/results/probe_size/' + graph_type[j] + '/dis_auc.txt', 'r')
        else:
            f = open('F:/论文/lp-new/link-prediction/results/probe_size/' + graph_type[j] + '/deg_auc.txt', 'r')
        lines = f.readlines()
        f.close()
        for line in lines:
            item = line.strip('\n')
            y.append(float(item))
        plt.subplot(subplot_loc[j][0], subplot_loc[j][1], subplot_loc[j][2])
        plt.scatter(x, y, color='#000000', s=100)
        A1, B1 = optimize.curve_fit(fline, x, y)[0]

        x1 = np.array(x)
        y1 = A1 * x1 + B1

        x2 = np.array(x)
        y2 = A1 * x2 + B1

        plt.plot(x1, y1, "#609BD4", linewidth=4.0)
        # plt.ticklabel_format(axis='x', style='sci', scilimits=(0, 0))
        plt.tick_params(labelsize=15)
        # sst = np.sum(np.power(y - np.average(y), 2))
        # ssr = np.sum(np.power(y2 - np.average(y), 2))
        # r_2 = ssr / sst
        if j == 0 or j == 2:
            # if graph_type[j] == 'new FW2':
            #     plt.ylim(0.81, 0.84)
            plt.xlabel(r'$\delta_{dis}$', font2, labelpad=8)
            plt.ylabel(r'AUC of SCM$_{dis}$', font2, labelpad=8)
        else:
            plt.xlabel(r'$\delta_{deg}$', font2, labelpad=8)
            plt.ylabel(r'AUC of SCM$_{deg}$', font2, labelpad=8)
        sse = np.sum(np.power(y - y2, 2))
        plt.title(lab_list[j], font2, x=-0.25)
        if j == 1:
            plt.ylim(0.75, 0.88)
            plt.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
        # if j == 2:
        #     plt.xlim(2, 6)
        # if j == 3 or j==4:
        #     plt.ticklabel_format(axis="x", style="sci", scilimits=(0,0))
        # if j == 5:
        #     plt.xlim(0.0002, 0.00055)
        #     plt.ticklabel_format(axis="x", style="sci", scilimits=(0,0))
        print(graph_type[j], A1, B1, sse)
    plt.subplots_adjust(wspace=0.35, hspace=0.25, left=0.09, right=0.98, bottom=0.15, top=0.9)
    plt.show()


def plot_add_tempral(alpha):
    graph_type = ['1', '2', '3', '4', '5', '6']
    link_type1 = ['deg', 'dis', 'cn', 'pa', 'lr', 'rwr']
    x = []
    lab_list = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)']

    color_list = ['#525252', '#F24040', '#1A70DE', '#38AD6B', '#B078DE', '#CC9900']
    point_type = ['s', 'o', '^', 'v', 'D', '<']
    subplot_loc = [(2, 3, 1), (2, 3, 2), (2, 3, 3), (2, 3, 4), (2, 3, 5), (2, 3, 6)]
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    font = {
        'family': 'simhei',
        'weight': 'bold',
        'size': 10,
    }
    font2 = {
        'family': 'simhei',
        'weight': 'bold',
        'size': 20,
    }
    a1 = plt.figure(figsize=(16, 5))
    for i in range(len(graph_type)):
        plt.subplot(subplot_loc[i][0], subplot_loc[i][1], subplot_loc[i][2])
        for j in range(len(link_type1)):
            y = []
            # yerror = []

            if j < 2:
                path = 'F:/论文/lp-new/link-prediction-time-vary/' + graph_type[i] + '/alpha' + str(alpha) + '/' + \
                       link_type1[j] + '_auc.txt'
            else:
                path = 'F:/论文/lp-new/link-prediction-time-vary/' + graph_type[i] + '/' + link_type1[j] + '/auc.txt'

            f = open(path, 'r')
            lines = f.readlines()
            f.close()

            ans = 0
            for line in lines:
                item = line.strip('\n')
                y.append(float(item))
                ans += 1
            x = [_ for _ in range(ans)]

            # f = open(path + '_auc_errorbar.txt', 'r')
            # lines = f.readlines()
            # f.close()
            # for line in lines:
            #     item = line.strip('\n')
            #     yerror.append(float(item))

            plt.plot(x, y, color=color_list[j], marker=point_type[j], linestyle='-')
            # plt.xlim(0.05, 0.5)
            # plt.ylim(0.25, 1)

            # if graph_type[i] == 'new FW2':
            #     plt.ylim(0.45, 1)
            # elif graph_type[i] == 'new FW3':
            #     plt.ylim(0.45, 1)
            # elif graph_type[i] == 'sampled SmaGri' or graph_type[i] == 'new FW1':
            #     plt.ylim(0.35, 0.95)
            # elif graph_type[i] == 'sampled Delicious':
            #     plt.ylim(0.25, 1)
            # else:
            #     plt.ylim(0.3, 0.9)

            # plt.xticks([0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45])
            # plt.tick_params(labelsize=11)

            plt.xlabel('the size of probe set', font2, labelpad=8)
            plt.ylabel('AUC', font2, labelpad=10)
            # plt.errorbar(x, y, color=color_list[j], marker=point_type[j], linestyle='-', yerr=yerror, capsize=3)
            plt.title(lab_list[i], font2, x=-0.17)

    plt.subplots_adjust(wspace=0.3, hspace=0.3, left=0.07, right=0.98, bottom=0.15, top=0.9)
    plt.show()


#新改的ws演化数据处理
def new_temporal_data_clean():
    probe_size = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45]
    re_p = [0.01, 0.02, 0.05, 0.1, 0.25, 0.5, 0.75, 1]
    alpha_list = [-8, -6, -4, -2, 0, 2, 4, 6, 8]

    for rp in re_p:
        for alp in alpha_list:
            store_avg_deg = []
            store_avg_dis = []
            dis_n = []
            for i in range(20):
                store_dis = []
                store_deg = []
                path = 'F:/论文/lp_new/WS_ver2/re_probability' + str(rp) + '/' + str(alp) + '/number' + str(i)

                f = open(path + '/deg_index.txt', 'r')
                lines = f.readlines()
                f.close()
                for line in lines:
                    store_deg.append(float(line))

                f = open(path + '/dis_index.txt', 'r')
                lines = f.readlines()
                f.close()
                for line in lines:
                    store_dis.append(float(line))

                if len(store_avg_deg) < len(store_deg):
                    if store_avg_deg == []:
                        store_avg_deg = [0 for i in range(len(store_deg))]
                        store_avg_dis = [0 for i in range(len(store_deg))]
                        dis_n = [0 for i in range(len(store_deg))]
                    else:
                        for le in range(len(store_deg) - len(store_avg_deg) ):
                            store_avg_deg.append(0)
                            store_avg_dis.append(0)
                            dis_n.append(0)

                for i1 in range(len(store_deg)):
                    store_avg_deg[i1] += store_deg[i1]
                    store_avg_dis[i1] += store_dis[i1]
                    if store_deg[i1] != 0:
                        dis_n[i1] += 1

            for i in range(len(store_avg_deg)):
                store_avg_deg[i] = store_avg_deg[i] / 20
                store_avg_dis[i] = store_avg_dis[i] / 20
                # if dis_n[i] != 0:
                #     store_avg_deg[i] = store_avg_deg[i] / dis_n[i]
                #     store_avg_dis[i] = store_avg_dis[i] / dis_n[i]

            path1 = 'F:/论文/lp_new/WS_ver2/re_probability' + str(rp) + '/' + str(alp)
            folder = os.path.exists(path1)
            if not folder:
                os.makedirs(path1)

            f = open(path1 + '/deg_avg.txt', 'w')
            for i in store_avg_deg:
                if i != 0:
                    f.write(str(i) + '\n')
            f.close()

            f = open(path1 + '/dis_avg.txt', 'w')
            for i in store_avg_dis:
                if i != 0:
                    f.write(str(i) + '\n')
            f.close()


#新改的WS，画deg
def new_plot_ws_deg():
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    subplot_loc = [(2, 4, 1), (2, 4, 2), (2, 4, 3), (2, 4, 4), (2, 4, 5), (2, 4, 6), (2, 4, 7), (2, 4, 8)]
    font2 = {
        'family': 'simhei',
        'weight': 'bold',
        'size': 20,
    }
    lab_list = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)', '(g)', '(h)']
    rep_type = [0.01, 0.02, 0.05, 0.1, 0.25, 0.5, 0.75, 1]
    # color_list = ['#676767', '#42B497', '#D74B0C', '#3176B8', '#A4B3D5', '#FDA481', '#85CEB7', '#C2C2C2']
    # color_list1 = ['#525252', '#F24040', '#1A70DE', '#38AD6B']
    # color_list2 = ['#E370FF', '#82D1FF', '#FF8000', '#A1E533']
    # color_list2 = ['#666666', '#F176A8', '#6BC6AC', '#9C83B7', '#F9A34E']

    # color_list2 = ['#609BD4', '#F9A34E', '#038303', '#790578']
    #color_list2 = ['#799EC7', '#B55750', '#9DB56A', '#7F659D']
    #color_list = ['#676767', '#42B497', '#D74B0C', '#3176B8', '#609BD4', '#F9A34E', '#038303', '#7F659D']

    # color_list = ['#D87DE5', '#F9A34E', '#038303', '#7F659D', '#42B497', '#D74B0C', '#3176B8', '#676767']
    color_list = ['#D87DE5', '#F9A34E', '#038303', '#7F659D', '#676767', '#3176B8', '#D74B0C', '#42B497']
    alpha_list = [-8, -6, -4, -2, 8, 6, 4, 2]
    for j in range(len(subplot_loc)):
        plt.subplot(subplot_loc[j][0], subplot_loc[j][1], subplot_loc[j][2])
        for i in range(8):
            y = []
            path = 'F:/论文/lp_new/WS_ver2/re_probability' + str(rep_type[j]) + '/' + str(alpha_list[i])
            f = open(path + '/deg_avg.txt', 'r')
            lines = f.readlines()
            f.close()
            for line in lines:
                item = line.strip('\n')
                if float(item) != 0:
                    y.append(float(item))

            x = [k for k in range(len(y)-1)]
            plt.plot(x, y[:-1], color=color_list[i])
            plt.yscale('log')
            plt.xscale('log')
            # plt.xlim(1, 2000)


        plt.xlabel('t', font2, labelpad=8)
        plt.ylabel(r'$\Delta_{\alpha}^{D}$', font2, labelpad=8)
        plt.title(lab_list[j], font2, x=-0.2)
        plt.tick_params(labelsize=12)
    plt.subplots_adjust(wspace=0.4, hspace=0.3, left=0.07, right=0.99, bottom=0.1, top=0.95)
    plt.show()


#新改的WS，画dis
def new_plot_ws_dis_1graph():
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    subplot_loc = [(2, 4, 1), (2, 4, 2), (2, 4, 3), (2, 4, 4), (2, 4, 5), (2, 4, 6), (2, 4, 7), (2, 4, 8)]
    font2 = {
        'family': 'simhei',
        'weight': 'bold',
        'size': 20,
    }
    lab_list = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)', '(g)', '(h)']
    rep_type = [0.01, 0.02, 0.05, 0.1, 0.25, 0.5, 0.75, 1]
    # color_list = ['#676767', '#42B497', '#D74B0C', '#3176B8', '#A4B3D5', '#FDA481', '#85CEB7', '#C2C2C2']
    # color_list1 = ['#525252', '#F24040', '#1A70DE', '#38AD6B']
    # color_list2 = ['#E370FF', '#82D1FF', '#FF8000', '#A1E533']
    # color_list2 = ['#666666', '#F176A8', '#6BC6AC', '#9C83B7', '#F9A34E']

    # color_list2 = ['#609BD4', '#F9A34E', '#038303', '#790578']
    #color_list2 = ['#799EC7', '#B55750', '#9DB56A', '#7F659D']
    #color_list = ['#676767', '#42B497', '#D74B0C', '#3176B8', '#609BD4', '#F9A34E', '#038303', '#7F659D']

    # color_list = ['#D87DE5', '#F9A34E', '#038303', '#7F659D', '#42B497', '#D74B0C', '#3176B8', '#676767']
    color_list = ['#D87DE5', '#F9A34E', '#038303', '#7F659D', '#676767', '#3176B8', '#D74B0C', '#42B497']
    alpha_list = [-8, -6, -4, -2, 8, 6, 4, 2]
    for j in range(len(subplot_loc)):
        plt.subplot(subplot_loc[j][0], subplot_loc[j][1], subplot_loc[j][2])
        for i in range(8):
            y = []
            path = 'F:/论文/lp_new/WS_ver2/re_probability' + str(rep_type[j]) + '/' + str(alpha_list[i])
            f = open(path + '/dis_avg.txt', 'r')
            lines = f.readlines()
            f.close()
            for line in lines:
                item = line.strip('\n')
                if float(item) != 0:
                    y.append(float(item))

            x = [k for k in range(len(y)-1)]
            plt.plot(x, y[:-1], color=color_list[i])
            plt.yscale('log')
            # plt.xscale('log')
            # plt.xlim(1, 2000)


        plt.xlabel('t', font2, labelpad=8)
        plt.ylabel(r'$\Delta_{\alpha}^{L}$', font2, labelpad=8)
        plt.title(lab_list[j], font2, x=-0.2)
        plt.tick_params(labelsize=12)
    plt.subplots_adjust(wspace=0.4, hspace=0.3, left=0.07, right=0.99, bottom=0.1, top=0.95)
    plt.show()


def new_plot_ws_dis_2graph():
    subplot_loc = [(2, 4, 1), (2, 4, 2), (2, 4, 3), (2, 4, 4), (2, 4, 5), (2, 4, 6), (2, 4, 7), (2, 4, 8)]
    font2 = {
        'family': 'simhei',
        'weight': 'bold',
        'size': 20,
    }
    lab_list = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)', '(g)', '(h)']
    rep_type = [0.01, 0.02, 0.05, 0.1, 0.25, 0.5, 0.75, 1]
    # color_list = ['#676767', '#42B497', '#D74B0C', '#3176B8', '#A4B3D5', '#FDA481', '#85CEB7', '#C2C2C2']
    # alpha_list = [-1, -0.75, -0.5, -0.25, 0.25, 0.5, 0.75, 1]

    color_list1 = ['#D87DE5', '#F9A34E', '#038303', '#7F659D']
    color_list2 = ['#676767', '#42B497', '#D74B0C', '#3176B8']

    alpha_list1 = [-8, -6, -4, -2]
    alpha_list2 = [2, 4, 6, 8]
    for j in range(len(subplot_loc)):
        for i in range(4):
            y = []
            path = 'F:/论文/lp_new/WS_ver2/re_probability' + str(rep_type[j]) + '/' + str(alpha_list2[i])
            f = open(path + '/dis_avg.txt', 'r')
            lines = f.readlines()
            f.close()
            for line in lines:
                item = line.strip('\n')
                if float(item) != 0:
                    y.append(float(item))
            plt.subplot(subplot_loc[j][0], subplot_loc[j][1], subplot_loc[j][2])

            x = [k for k in range(len(y) - 1)]
            plt.plot(x, y[:-1], color=color_list2[i])
            plt.yscale('log')
            # plt.xscale('log')
            # plt.xlim(-50, 1050)
        # if j == 0:
        #     plt.ylim(0.0001, 20)
        plt.xlabel('t', font2, labelpad=8)
        plt.ylabel(r'$\delta_{dis}$', font2, labelpad=8)
        plt.title(lab_list[j], font2, x=-0.2)
        plt.tick_params(labelsize=12)

    plt.subplots_adjust(wspace=0.4, hspace=0.3, left=0.07, right=0.99, bottom=0.1, top=0.95)
    plt.show()


    for j in range(len(subplot_loc)):
        for i in range(4):
            y = []
            path = 'F:/论文/lp_new/WS_ver2/re_probability' + str(rep_type[j]) + '/' + str(alpha_list1[i])
            f = open(path + '/dis_avg.txt', 'r')
            lines = f.readlines()
            f.close()
            for line in lines:
                item = line.strip('\n')
                if float(item) != 0:
                    y.append(float(item))
            plt.subplot(subplot_loc[j][0], subplot_loc[j][1], subplot_loc[j][2])

            x = [k for k in range(len(y) - 1)]
            plt.plot(x, y[:-1], color=color_list1[i])
            plt.yscale('log')
            # plt.xscale('log')
            # plt.xlim(-50, 1050)
        # if j == 0:
        #     plt.ylim(0.0001, 20)
        plt.xlabel('t', font2, labelpad=8)
        plt.ylabel(r'$\delta_{dis}$', font2, labelpad=8)
        plt.title(lab_list[j], font2, x=-0.2)
        plt.tick_params(labelsize=12)

    plt.subplots_adjust(wspace=0.4, hspace=0.3, left=0.07, right=0.99, bottom=0.1, top=0.95)
    plt.show()

#新加的时序网络的演变
def new_plot_tempoarl_evolution(type):
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    graph_list = ['contacts-prox-high-school-2013', 'SFHH-conf-sensor']
    subplot_loc = [(1, 2, 1), (1, 2, 2)]
    font2 = {
        'family': 'simhei',
        'weight': 'bold',
        'size': 20,
    }
    lab_list = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)', '(g)', '(h)']
    color_list = ['#D87DE5', '#F9A34E', '#038303', '#7F659D', '#676767', '#3176B8', '#D74B0C', '#42B497']

    # color_list1 = ['#D87DE5', '#F9A34E', '#038303', '#7F659D']
    # color_list2 = ['#676767', '#42B497', '#D74B0C', '#3176B8']

    alpha_list = [-8, -6, -4, -2, 2, 8, 6, 4]
    a1 = plt.figure(figsize=(10, 5))
    for j in range(2):
        plt.subplot(subplot_loc[j][0], subplot_loc[j][1], subplot_loc[j][2])
        for i in range(8):
            y = []
            path = 'F:/论文/lp-new/link-prediction-time-vary/evolution/' + str(graph_list[j]) + '/' + str(alpha_list[i])
            f = open(path + '/' + type + '_index.txt', 'r')
            lines = f.readlines()
            f.close()
            for line in lines:
                item = line.strip('\n')
                if float(item) != 0:
                    y.append(float(item))

            x = [k for k in range(len(y)-1)]
            plt.plot(x, y[:-1], color=color_list[i])

            plt.yscale('log')
            # plt.xscale('log')
            # plt.xlim(1, 2000)
        if type == 'deg':
            plt.xlabel('t', font2, labelpad=8)
            plt.ylabel(r'$\Delta_{\alpha}^{D}$', font2, labelpad=8)
            plt.title(lab_list[j], font2, x=-0.2)
            plt.tick_params(labelsize=12)
        else:
            plt.xlabel('t', font2, labelpad=8)
            plt.ylabel(r'$\Delta_{\alpha}^{L}$', font2, labelpad=8)
            plt.title(lab_list[j], font2, x=-0.2)
            plt.tick_params(labelsize=12)
    # plt.subplots_adjust(wspace=0.4, hspace=0.3, left=0.07, right=0.99, bottom=0.1, top=0.95)
    # plt.show()

    # for j in range(2, len(subplot_loc)):
    #     plt.subplot(subplot_loc[j][0], subplot_loc[j][1], subplot_loc[j][2])
    #     for i in range(8):
    #         y = []
    #         path = 'F:/论文/lp-new/link-prediction-time-vary/evolution/' + str(graph_list[j-2]) + '/' + str(alpha_list[i])
    #         f = open(path + '/dis_index.txt', 'r')
    #         lines = f.readlines()
    #         f.close()
    #         for line in lines:
    #             item = line.strip('\n')
    #             if float(item) != 0:
    #                 y.append(float(item))
    #
    #         x = [k for k in range(len(y)-1)]
    #         plt.plot(x, y[:-1], color=color_list[i])
    #         plt.yscale('log')
    #         plt.xscale('log')
    #         # plt.xlim(1, 2000)
    #
    #     plt.xlabel('t', font2, labelpad=8)
    #     plt.ylabel(r'$\delta_{dis}$', font2, labelpad=8)
    #     plt.title(lab_list[j], font2, x=-0.2)
    #     plt.tick_params(labelsize=12)
    plt.subplots_adjust(wspace=0.4, hspace=0.4, left=0.11, right=0.99, bottom=0.15, top=0.9)
    plt.show()


#新加的时序网络的演变，随着probsize改变的数据处理
def new_clean_tempral_base_probesize():
    graph_type = ['1', '2', '6', 'contacts-prox-high-school-2013', 'SFHH-conf-sensor']
    link_list = ['deg', 'dis', 'cn', 'pa', 'lr', 'rwr']
    for graph in graph_type:
        for link_type in link_list:
            store = []
            if link_type == 'deg':
                if graph == '1':
                    path = 'F:/论文/lp-new/link-prediction-time-vary/' + graph + '/alpha-8/' + link_type + '_auc.txt'
                else:
                    path = 'F:/论文/lp-new/link-prediction-time-vary/' + graph + '/alpha8/' + link_type + '_auc.txt'
            elif link_type == 'dis':
                path = 'F:/论文/lp-new/link-prediction-time-vary/' + graph + '/alpha8/' + link_type + '_auc.txt'
            else:
                path = 'F:/论文/lp-new/link-prediction-time-vary/' + graph + '/' + link_type + '/auc.txt'
            f = open(path, 'r')
            lines = f.readlines()
            f.close()
            for line in lines:
                store.append(line)

            path1 = 'F:/论文/lp-new/link-prediction-time-vary/results/probe_size/' + graph
            folder = os.path.exists(path1)
            if not folder:
                os.makedirs(path1)

            f = open(path1 + '/' + link_type + '_auc.txt', 'w')
            for i in store:
                f.write(str(i))
            f.close()


#新加的时序网络的演变，随着probsize改变而变化
def new_plot_probesize():
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    graph_type = ['1', '2', 'contacts-prox-high-school-2013', 'SFHH-conf-sensor']
    link_type1 = ['deg', 'dis', 'pa', 'lr', 'rwr', 'cn']

    lab_list = ['(a)', '(b)', '(c)', '(d)', '(e)']

    color_list = ['#525252', '#F24040', '#38AD6B', '#B078DE', '#CC9900', '#1A70DE']
    point_type = ['s', 'o', '^', 'v', 'D', '<']
    subplot_loc = [(2, 2, 1), (2, 2, 2), (2, 2, 3), (2, 2, 4)]

    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    font = {
        'family': 'simhei',
        'weight': 'bold',
        'size': 10,
    }
    font2 = {
        'family': 'simhei',
        'weight': 'bold',
        'size': 20,
    }
    a1 = plt.figure(figsize=(10, 8))
    for i in range(len(graph_type)):
        plt.subplot(subplot_loc[i][0], subplot_loc[i][1], subplot_loc[i][2])
        for j in range(len(link_type1)):
            y = []
            ans = 0
            path = 'F:/论文/lp-new/link-prediction-time-vary/results/probe_size/' + graph_type[i] + '/' + link_type1[j]
            f = open(path + '_auc.txt', 'r')
            lines = f.readlines()
            f.close()
            for line in lines:
                item = line.strip('\n')
                y.append(float(item))
                ans += 1
            # x = [_ for _ in range(ans)]
            # plt.plot(x, y, color=color_list[j], marker=point_type[j], linestyle='-')
            # plt.xticks(x)

            x = [_ for _ in range(ans)]
            if i > 1:
                j_g = int(len(y) / 100)
                y1 = []
                yeer = []
                for y_value in range(100):
                    y1.append(sum(y[y_value:y_value+j_g]) / j_g)
                    yeer.append(np.std(y[y_value:y_value+j_g]))

                x1 = [_ for _ in range(100)]

                y_par = []
                x_par = []
                yerr_par = []
                # for k in range(30):
                #     y_par.append(y1[int(k*(len(y1) / 30))])
                #     x_par.append(x1[int(k * (len(y1) / 30))])
                #     yerr_par.append(yeer[int(k * (len(y1) / 30))])
                # plt.plot(x_par, y_par, color=color_list[j], marker=point_type[j], linestyle='-')
                # plt.errorbar(x_par, y_par, yerr_par, capsize=1.5, elinewidth=1.5, ecolor=color_list[j], ls='none')

                for k in range(100):
                    y_par.append(y[int(k*(len(y) / 100))])
                    x_par.append(x[int(k * (len(y) / 100))])
                plt.plot(x_par, y_par, color=color_list[j], marker=point_type[j], linestyle='-', markersize=3)

                # plt.plot(x1, y1, color=color_list[j], marker=point_type[j], linestyle='-')
                # plt.errorbar(x1, y1, yeer, capsize=1, elinewidth=1, ecolor=color_list[j], ls='none')
                ##plt.xticks(x1, [_ + 10 for _ in range(100)])
            else:
                plt.plot(x, y, color=color_list[j], marker=point_type[j], linestyle='-')
                plt.xticks(x)


            # plt.xlim(0.05, 0.5)
            # if i > 1:
            #     plt.ylim(0.5, 1)

            # plt.xticks([0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45])
            plt.tick_params(labelsize=11)

            plt.xlabel('t', font2, labelpad=8)
            plt.ylabel('AUC', font2, labelpad=10)
            # plt.errorbar(x, y, color=color_list[j], marker=point_type[j], linestyle='-', yerr=yerror, capsize=3)
            plt.title(lab_list[i], font2, x=-0.17)

    plt.subplots_adjust(wspace=0.3, hspace=0.5, left=0.09, right=0.98, bottom=0.1, top=0.9)
    plt.show()


def new_plot_probesize_otherver_of1and2():
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    graph_type = ['1', '2']
    link_type1 = ['deg', 'dis', 'cn', 'pa', 'lr', 'rwr']

    lab_list = ['(a)', '(b)']

    color_list = ['#525252', '#F24040', '#1A70DE', '#38AD6B', '#B078DE', '#CC9900']
    point_type = ['s', 'o', '^', 'v', 'D', '<']
    subplot_loc = [(1, 2, 1), (1, 2, 2)]

    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    font = {
        'family': 'simhei',
        'weight': 'bold',
        'size': 10,
    }
    font2 = {
        'family': 'simhei',
        'weight': 'bold',
        'size': 20,
    }
    a1 = plt.figure(figsize=(10, 5))
    for i in range(len(graph_type)):
        plt.subplot(subplot_loc[i][0], subplot_loc[i][1], subplot_loc[i][2])
        for j in range(len(link_type1)):
            y = []
            ans = 0
            path = 'F:/论文/lp-new/link-prediction-time-vary/results/probe_size/' + graph_type[i] + '/' + link_type1[j]
            f = open(path + '_auc.txt', 'r')
            lines = f.readlines()
            f.close()
            for line in lines:
                item = line.strip('\n')
                y.append(float(item))
                ans += 1
            x = [_ for _ in range(ans)]
            plt.plot(x, y, color=color_list[j], marker=point_type[j], linestyle='-')
            plt.xticks(x)

            plt.tick_params(labelsize=11)

            plt.xlabel('t', font2, labelpad=8)
            plt.ylabel('AUC', font2, labelpad=10)
            # plt.errorbar(x, y, color=color_list[j], marker=point_type[j], linestyle='-', yerr=yerror, capsize=3)
            plt.title(lab_list[i], font2, x=-0.17)

    plt.subplots_adjust(wspace=0.3, hspace=0.5, left=0.09, right=0.98, bottom=0.1, top=0.9)
    plt.show()


def new_plot_probesize_otherver_ofcpshandsfs():
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    graph_type = ['contacts-prox-high-school-2013', 'SFHH-conf-sensor']
    link_type1 = ['deg', 'dis', 'cn', 'pa', 'lr', 'rwr']

    lab_list = ['(a)', '(b)', '(c)', '(d)', '(e)']

    color_list = ['#525252', '#F24040', '#1A70DE', '#38AD6B', '#B078DE', '#CC9900']
    point_type = ['s', 'o', '^', 'v', 'D', '<']
    subplot_loc = [(2, 3, 1), (2, 3, 2), (2, 3, 3), (2, 3, 4), (2, 3, 5), (2, 3, 6)]

    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    font = {
        'family': 'simhei',
        'weight': 'bold',
        'size': 10,
    }
    font2 = {
        'family': 'simhei',
        'weight': 'bold',
        'size': 20,
    }
    a1 = plt.figure(figsize=(10, 8))
    for i in range(len(graph_type)):
        for j in range(len(link_type1)):
            plt.subplot(subplot_loc[j][0], subplot_loc[j][1], subplot_loc[j][2])
            y = []
            ans = 0
            path = 'F:/论文/lp-new/link-prediction-time-vary/results/probe_size/' + graph_type[i] + '/' + link_type1[j]
            f = open(path + '_auc.txt', 'r')
            lines = f.readlines()
            f.close()
            for line in lines:
                item = line.strip('\n')
                y.append(float(item))
                ans += 1
            x = [_ for _ in range(ans)]
            plt.plot(x, y, color=color_list[j], marker=point_type[j], linestyle='-')
            plt.xticks(x)

            # x = [_ for _ in range(ans)]
            # if i > 1:
            #     j_g = int(len(y) / 10)
            #     y1 = []
            #     left = 0
            #     for y_value in range(10):
            #         y1.append(sum(y[left:left+j_g]) / j_g)
            #     x1 = [_ for _ in range(10)]
            #     plt.plot(x1, y1, color=color_list[j], marker=point_type[j], linestyle='-')
            #     plt.xticks(x1)
            # else:
            #     plt.plot(x, y, color=color_list[j], marker=point_type[j], linestyle='-')
            #     plt.xticks(x)


            # plt.xlim(0.05, 0.5)
            # if i > 1:
            #     plt.ylim(0.5, 1)

            # plt.xticks([0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45])
            plt.tick_params(labelsize=11)

            plt.xlabel('t', font2, labelpad=8)
            plt.ylabel('AUC', font2, labelpad=10)
            # plt.errorbar(x, y, color=color_list[j], marker=point_type[j], linestyle='-', yerr=yerror, capsize=3)
            plt.title(lab_list[i], font2, x=-0.17)

        plt.subplots_adjust(wspace=0.3, hspace=0.5, left=0.09, right=0.98, bottom=0.1, top=0.9)
        plt.show()


def new_plot_tempral_alphachange(type):
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'

    graph_type = ['1', '2', 'contacts-prox-high-school-2013', 'SFHH-conf-sensor']
    alpha_list = [8, 6, 4, 2, 0, -2, -4, -6, -8]
    lab_list = ['(a)', '(b)', '(c)', '(d)']
    subplot_loc = [(2, 2, 1), (2, 2, 2), (2, 2, 3), (2, 2, 4)]
    # color2 = LinearSegmentedColormap.from_list("", ['#2D2DE0', '#FFFFFF', '#DE2727'])
    color2 = LinearSegmentedColormap.from_list("", ['#4791C5', '#FFFFFF', '#DE6C5B'])
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    font = {
        'family': 'simhei',
        'weight': 'bold',
        'size': 25,
    }
    font2 = {
        'family': 'simhei',
        'weight': 'bold',
        'size': 25,
    }

    a1 = plt.figure(figsize=(10, 8))
    for i in range(len(graph_type)):
        plt.subplot(subplot_loc[i][0], subplot_loc[i][1], subplot_loc[i][2])
        data = []
        for j in range(len(alpha_list)):
            store = []
            path = 'F:/论文/lp-new/link-prediction-time-vary/' + graph_type[i] + '/alpha' + str(alpha_list[j]) + '/' + \
                   type
            f = open(path + '_auc.txt', 'r')
            lines = f.readlines()
            f.close()
            for line in lines:
                store.append(float(line))

            data.append(store)

            # if i > 1:
            #     j_g = int(len(store) / 1000)
            #     y1 = []
            #     for y_value in range(1000):
            #         y1.append(sum(store[y_value:y_value + j_g]) / j_g)
            #     data.append(y1)
            # else:
            #     data.append(store)

        data = pd.DataFrame(data)
        if type == "dis":
            sns.heatmap(data, robust=True, vmin=0, vmax=1, cmap=color2, annot_kws={"front_size": 11})
        else:
            sns.heatmap(data, robust=True, vmin=0, vmax=1, cmap=color2, annot_kws={"front_size": 15})
        if i == 2:
            plt.xticks(np.arange(0, 7001, 1000), np.arange(0, 7001, 1000), rotation=0)
        if i == 3:
            plt.xticks(np.arange(0, 3001, 500), np.arange(0, 3001, 500), rotation=0)
        plt.yticks([i+0.5 for i in range(9)], [8, 6, 4, 2, 0, -2, -4, -6, -8], rotation='horizontal')
        plt.xlabel('t', font2, labelpad=10)
        plt.ylabel(r'$\alpha$', font2, labelpad=10)
        plt.tick_params(labelsize=15)
        plt.title(lab_list[i], font, x=-0.17)

    plt.subplots_adjust(wspace=0.2, hspace=0.4, left=0.1, right=0.98, bottom=0.1, top=0.95)
    plt.show()


def new_plot_tempral_alphachange1graph():
    graph_type = ['1', '2', 'contacts-prox-high-school-2013', 'SFHH-conf-sensor']
    alpha_list = [8, 6, 4, 2, 0, -2, -4, -6, -8]
    lab_list = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)', '(g)', '(h)']
    subplot_loc = [(2, 4, 1), (2, 4, 2), (2, 4, 3), (2, 4, 4), (2, 4, 5), (2, 4, 6), (2, 4, 7), (2, 4, 8)]
    # color2 = LinearSegmentedColormap.from_list("", ['#2D2DE0', '#FFFFFF', '#DE2727'])
    color2 = LinearSegmentedColormap.from_list("", ['#4791C5', '#FFFFFF', '#DE6C5B'])
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    font = {
        'family': 'simhei',
        'weight': 'bold',
        'size': 10,
    }
    font2 = {
        'family': 'simhei',
        'weight': 'bold',
        'size': 20,
    }

    a1 = plt.figure(figsize=(16, 8))
    for i in range(len(graph_type)):
        plt.subplot(subplot_loc[i][0], subplot_loc[i][1], subplot_loc[i][2])
        data = []
        for j in range(len(alpha_list)):
            store = []
            path = 'F:/论文/lp-new/link-prediction-time-vary/' + graph_type[i] + '/alpha' + str(alpha_list[j])
            f = open(path + '/deg_auc.txt', 'r')
            lines = f.readlines()
            f.close()
            for line in lines:
                store.append(float(line))

            if i > 1:
                j_g = int(len(store) / 10)
                y1 = []
                left = 0
                for y_value in range(10):
                    y1.append(sum(store[left:left + j_g]) / j_g)
                data.append(y1)
            else:
                data.append(store)

        data = pd.DataFrame(data)
        sns.heatmap(data, robust=True, vmin=0, vmax=1, cmap=color2, annot_kws={"front_size": 15})

        # plt.xticks([i+0.5 for i in range(len(data))], [i for i in range(len(data))])
        # plt.yticks([i+0.5 for i in range(9)], [8, 6, 4, 2, 0, -2, -4, -6, -8], rotation='horizontal')
        plt.xlabel('t', font2, labelpad=8)
        plt.ylabel(r'$\alpha$', font2, labelpad=10)
        plt.tick_params(labelsize=11)
        plt.title(lab_list[i], font2, x=-0.17)

        plt.subplot(subplot_loc[i+4][0], subplot_loc[i+4][1], subplot_loc[i+4][2])
        data = []
        for j in range(len(alpha_list)):
            store = []
            path = 'F:/论文/lp-new/link-prediction-time-vary/' + graph_type[i] + '/alpha' + str(alpha_list[j])
            f = open(path + '/dis_auc.txt', 'r')
            lines = f.readlines()
            f.close()
            for line in lines:
                store.append(float(line))

            if i > 1:
                j_g = int(len(store) / 10)
                y1 = []
                left = 0
                for y_value in range(10):
                    y1.append(sum(store[left:left + j_g]) / j_g)
                data.append(y1)
            else:
                data.append(store)

        data = pd.DataFrame(data)
        sns.heatmap(data, robust=True, vmin=0, vmax=1, cmap=color2, annot_kws={"front_size": 11})

        # plt.xticks([i+0.5 for i in range(len(data))], [i for i in range(len(data))])
        plt.yticks([i + 0.5 for i in range(9)], [8, 6, 4, 2, 0, -2, -4, -6, -8], rotation='horizontal')
        plt.xlabel('t', font2, labelpad=8)
        plt.ylabel(r'$\alpha$', font2, labelpad=10)
        plt.tick_params(labelsize=11)
        plt.title(lab_list[i], font2, x=-0.17)

    plt.subplots_adjust(wspace=0.4, hspace=0.3, left=0.07, right=0.99, bottom=0.1, top=0.95)
    plt.show()


def compute_avg():
    graph_type = ['SFHH-conf-sensor']
    alpha_list = [8, 6, 4, 2, 0, -2, -4, -6, -8]
    link_type = ['cn']
    for graph in graph_type:
        for alp in alpha_list:
            path = 'F:/论文/lp-new/link-prediction-time-vary/' + graph + '/alpha' + str(alp)

            f = open(path + '/deg_auc.txt', 'r')
            lines = f.readlines()
            f.close()
            ans = 0
            count = 0
            for line in lines:
                ans += float(line.strip('\n'))
                count += 1

            f = open(path + '/average_deg_auc.txt', 'w')
            f.write(str(ans / count))
            f.close()

            f = open(path + '/dis_auc.txt', 'r')
            lines = f.readlines()
            f.close()
            ans = 0
            count = 0
            for line in lines:
                ans += float(line.strip('\n'))
                count += 1

            f = open(path + '/average_dis_auc.txt', 'w')
            f.write(str(ans / count))
            f.close()

        for link in link_type:
            path = 'F:/论文/lp-new/link-prediction-time-vary/' + graph + '/' + link

            f = open(path + '/auc.txt', 'r')
            lines = f.readlines()
            f.close()
            ans = 0
            count = 0
            for line in lines:
                ans += float(line.strip('\n'))
                count += 1

            f = open(path + '/average_auc.txt', 'w')
            f.write(str(ans / count))
            f.close()


def plot_ba_dis_new(graph):
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    subplot_loc = [(1, 4, 1), (1, 4, 2), (1, 4, 3), (1, 4, 4)]
    font2 = {
        'family': 'simhei',
        'weight': 'bold',
        'size': 20,
    }
    lab_list = ['(a)', '(b)', '(c)', '(d)', '(a)', '(b)', '(c)', '(d)']
    degree_type = [2, 4, 6, 8, 2, 4, 6, 8]
    # color_list2 = ['#609BD4', '#F9A34E', '#038303', '#790578']
    ###color_list1 = ['#609BD4', '#F08082', '#7F007F', '#FFA400']
    ## color_list2 = ['#75C29C', '#F6A248', '#CBBCE1', '#F3A1C2']
    # F27E35 '#427FA7'
    ##color_list1 = ['#DB6A6D', '#4E5D9B', '#E4DB3C', '#41BBBC']
    # color_list2 = ['#799EC7', '#B55750', '#9DB56A', '#7F659D']
    color_list1 = ['#D87DE5', '#F9A34E', '#038303', '#7F659D']
    color_list2 = ['#676767', '#42B497', '#D74B0C', '#3176B8']
    alpha_list1 = [-8, -6, -4, -2]
    alpha_list2 = [2, 4, 6, 8]
    alpha_list = [-8, -6, -4, -2, 2, 4, 6, 8]
    color_list = ['#D87DE5', '#F9A34E', '#038303', '#7F659D', '#676767', '#42B497', '#D74B0C', '#3176B8']
    figure, ax = plt.subplots(figsize=(20, 5))
    for j in range(4):
        # figure, ax = plt.subplots(figsize=(10, 8))
        for i in range(8):
            y = []
            if j < 4:
                path = 'F:/论文/lp-new/' + graph + '/average_degree' + str(degree_type[j]) + '/' + str(alpha_list[i])
            else:
                path = 'F:/论文/lp-new/' + graph + '/average_degree' + str(degree_type[j]) + '/' + str(alpha_list[i])

            f = open(path + '/dis_avg.txt', 'r')
            lines = f.readlines()
            f.close()
            for line in lines:
                item = line.strip('\n')
                y.append(float(item))
            plt.subplot(subplot_loc[j][0], subplot_loc[j][1], subplot_loc[j][2])

            x = [k for k in range(len(y) - 1)]
            if i < 4:
                plt.plot(x, y[:-1], color=color_list[i])
            else:
                plt.plot(x, y[:-1], color=color_list[i])
            plt.yscale('log')
            plt.xscale('log')
            #plt.xlim(-50, 1050)
        # if j == 0:
        #     plt.ylim(0.0001, 20)
        plt.xlabel('t', font2, labelpad=8)
        plt.ylabel(r'$\Delta_{\alpha}^{L}$', font2, labelpad=8)
        plt.title(lab_list[j], font2, x=-0.1)
        plt.tick_params(labelsize=12)

    plt.subplots_adjust(wspace=0.4, hspace=0.3, left=0.06, right=0.99, bottom=0.12, top=0.9)
    plt.show()


def compute_avg_dynamics():
    graph_type = ['contacts-prox-high-school-2013', 'SFHH-conf-sensor']
    alpha_list = [8, 6, 4, 2, 0, -2, -4, -6, -8]
    link_type = ['cn']
    for graph in graph_type:
        for alp in alpha_list:
            path = 'F:/论文/lp-new/dynamic_stablization_link_prediction/' + graph + '/alpha' + str(alp)

            f = open(path + '/deg_auc.txt', 'r')
            lines = f.readlines()
            f.close()
            ans = 0
            count = 0
            for line in lines:
                ans += float(line.strip('\n'))
                count += 1

            f = open(path + '/average_deg_auc.txt', 'w')
            f.write(str(ans / count))
            f.close()

            f = open(path + '/dis_auc.txt', 'r')
            lines = f.readlines()
            f.close()
            ans = 0
            count = 0
            for line in lines:
                ans += float(line.strip('\n'))
                count += 1

            f = open(path + '/average_dis_auc.txt', 'w')
            f.write(str(ans / count))
            f.close()

        # for link in link_type:
        #     path = 'F:/论文/lp-new/dynamic_stablization_link_prediction/' + graph + '/' + link
        #
        #     f = open(path + '/auc.txt', 'r')
        #     lines = f.readlines()
        #     f.close()
        #     ans = 0
        #     count = 0
        #     for line in lines:
        #         ans += float(line.strip('\n'))
        #         count += 1
        #
        #     f = open(path + '/average_auc.txt', 'w')
        #     f.write(str(ans / count))
        #     f.close()


def new_plot_tempral_alphachange_dynamic(type):
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'

    graph_type = ['contacts-prox-high-school-2013', 'SFHH-conf-sensor', 'contacts-prox-high-school-2013', 'SFHH-conf-sensor']
    alpha_list = [8, 6, 4, 2, 0, -2, -4, -6, -8]
    lab_list = ['(a)', '(b)', '(c)', '(d)']
    subplot_loc = [(2, 2, 1), (2, 2, 2), (2, 2, 3), (2, 2, 4)]
    # color2 = LinearSegmentedColormap.from_list("", ['#2D2DE0', '#FFFFFF', '#DE2727'])
    color2 = LinearSegmentedColormap.from_list("", ['#4791C5', '#FFFFFF', '#DE6C5B'])
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    font = {
        'family': 'simhei',
        'weight': 'bold',
        'size': 25,
    }
    font2 = {
        'family': 'simhei',
        'weight': 'bold',
        'size': 25,
    }

    a1 = plt.figure(figsize=(10, 8))
    for i in range(len(graph_type)):
        plt.subplot(subplot_loc[i][0], subplot_loc[i][1], subplot_loc[i][2])
        data = []
        for j in range(len(alpha_list)):
            store = []
            if i < 2:
                path = 'F:/论文/lp-new/dynamic_stablization_link_prediction/' + graph_type[i] + '/alpha' + str(
                    alpha_list[j]) + '/' + type
            else:
                path = 'F:/论文/lp-new/link-prediction-time-vary/' + graph_type[i] + '/alpha' + str(alpha_list[j]) + '/' + \
                    type
            f = open(path + '_auc.txt', 'r')
            lines = f.readlines()
            f.close()
            for line in lines:
                store.append(float(line))

            data.append(store)

            # if i > 1:
            #     j_g = int(len(store) / 1000)
            #     y1 = []
            #     for y_value in range(1000):
            #         y1.append(sum(store[y_value:y_value + j_g]) / j_g)
            #     data.append(y1)
            # else:
            #     data.append(store)

        data = pd.DataFrame(data)
        if type == "dis":
            sns.heatmap(data, robust=True, vmin=0, vmax=1, cmap=color2, annot_kws={"front_size": 11})
        else:
            sns.heatmap(data, robust=True, vmin=0, vmax=1, cmap=color2, annot_kws={"front_size": 15})
        if i == 0 or i == 2:
            plt.xticks(np.arange(0, 7001, 1000), np.arange(0, 7001, 1000), rotation=0)
        if i == 1 or i == 3:
            plt.xticks(np.arange(0, 3001, 500), np.arange(0, 3001, 500), rotation=0)
        plt.yticks([i+0.5 for i in range(9)], [8, 6, 4, 2, 0, -2, -4, -6, -8], rotation='horizontal')
        plt.xlabel('t', font2, labelpad=10)
        plt.ylabel(r'$\alpha$', font2, labelpad=10)
        plt.tick_params(labelsize=15)
        plt.title(lab_list[i], font, x=-0.17)

    plt.subplots_adjust(wspace=0.2, hspace=0.4, left=0.1, right=0.98, bottom=0.1, top=0.95)
    plt.show()


def new_clean_tempral_base_probesize_dynamic():
    graph_type = ['contacts-prox-high-school-2013', 'SFHH-conf-sensor']
    link_list = ['deg', 'dis', 'cn', 'pa', 'lr', 'rwr']
    for graph in graph_type:
        for link_type in link_list:
            store = []
            if link_type == 'deg':
                path = 'F:/论文/lp-new/dynamic_stablization_link_prediction/' + graph + '/alpha8/' + link_type + '_auc.txt'
            elif link_type == 'dis':
                path = 'F:/论文/lp-new/dynamic_stablization_link_prediction/' + graph + '/alpha8/' + link_type + '_auc.txt'
            else:
                path = 'F:/论文/lp-new/link-prediction-time-vary/' + graph + '/' + link_type + '/auc.txt'
            f = open(path, 'r')
            lines = f.readlines()
            f.close()
            for line in lines:
                store.append(line)

            path1 = 'F:/论文/lp-new/dynamic_stablization_link_prediction/results/probe_size/' + graph
            folder = os.path.exists(path1)
            if not folder:
                os.makedirs(path1)

            f = open(path1 + '/' + link_type + '_auc.txt', 'w')
            for i in store:
                f.write(str(i))
            f.close()


def new_plot_probesize_dynamic():
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    graph_type = ['contacts-prox-high-school-2013', 'SFHH-conf-sensor', 'contacts-prox-high-school-2013', 'SFHH-conf-sensor']
    link_type1 = ['deg', 'dis', 'pa', 'lr', 'rwr', 'cn']

    lab_list = ['(a)', '(b)', '(c)', '(d)', '(e)']

    color_list = ['#525252', '#F24040', '#38AD6B', '#B078DE', '#CC9900', '#1A70DE']
    point_type = ['s', 'o', '^', 'v', 'D', '<']
    subplot_loc = [(2, 2, 1), (2, 2, 2), (2, 2, 3), (2, 2, 4)]

    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    font = {
        'family': 'simhei',
        'weight': 'bold',
        'size': 10,
    }
    font2 = {
        'family': 'simhei',
        'weight': 'bold',
        'size': 20,
    }
    a1 = plt.figure(figsize=(10, 8))
    for i in range(len(graph_type)):
        plt.subplot(subplot_loc[i][0], subplot_loc[i][1], subplot_loc[i][2])
        for j in range(len(link_type1)):
            y = []
            ans = 0
            if i < 2:
                path = 'F:/论文/lp-new/dynamic_stablization_link_prediction/results/probe_size/' + graph_type[i] + '/' + link_type1[j]
            else:
                path = 'F:/论文/lp-new/link-prediction-time-vary/results/probe_size/' + graph_type[i] + '/' + \
                       link_type1[j]
            f = open(path + '_auc.txt', 'r')
            lines = f.readlines()
            f.close()
            for line in lines:
                item = line.strip('\n')
                y.append(float(item))
                ans += 1
            # x = [_ for _ in range(ans)]
            # plt.plot(x, y, color=color_list[j], marker=point_type[j], linestyle='-')
            # plt.xticks(x)

            x = [_ for _ in range(ans)]
            if i > -1:
                j_g = int(len(y) / 100)
                y1 = []
                yeer = []
                for y_value in range(100):
                    y1.append(sum(y[y_value:y_value+j_g]) / j_g)
                    yeer.append(np.std(y[y_value:y_value+j_g]))

                x1 = [_ for _ in range(100)]

                y_par = []
                x_par = []
                yerr_par = []
                # for k in range(30):
                #     y_par.append(y1[int(k*(len(y1) / 30))])
                #     x_par.append(x1[int(k * (len(y1) / 30))])
                #     yerr_par.append(yeer[int(k * (len(y1) / 30))])
                # plt.plot(x_par, y_par, color=color_list[j], marker=point_type[j], linestyle='-')
                # plt.errorbar(x_par, y_par, yerr_par, capsize=1.5, elinewidth=1.5, ecolor=color_list[j], ls='none')

                for k in range(100):
                    y_par.append(y[int(k*(len(y) / 100))])
                    x_par.append(x[int(k * (len(y) / 100))])
                plt.plot(x_par, y_par, color=color_list[j], marker=point_type[j], linestyle='-', markersize=3)

                # plt.plot(x1, y1, color=color_list[j], marker=point_type[j], linestyle='-')
                # plt.errorbar(x1, y1, yeer, capsize=1, elinewidth=1, ecolor=color_list[j], ls='none')
                ##plt.xticks(x1, [_ + 10 for _ in range(100)])
            else:
                plt.plot(x, y, color=color_list[j], marker=point_type[j], linestyle='-')
                plt.xticks(x)


            # plt.xlim(0.05, 0.5)
            # if i > 1:
            #     plt.ylim(0.5, 1)
            plt.yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
            # plt.xticks([0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45])
            plt.tick_params(labelsize=11)

            plt.xlabel('t', font2, labelpad=8)
            plt.ylabel('AUC', font2, labelpad=10)
            # plt.errorbar(x, y, color=color_list[j], marker=point_type[j], linestyle='-', yerr=yerror, capsize=3)
            plt.title(lab_list[i], font2, x=-0.17)

    plt.subplots_adjust(wspace=0.3, hspace=0.5, left=0.09, right=0.98, bottom=0.1, top=0.9)
    plt.show()


if __name__ == '__main__':
    # plot_alphachange('deg')
    # plot_probsize()
    # new_plot_probesize_dynamic()
    # new_clean_tempral_base_probesize_dynamic()
    # new_plot_tempral_alphachange_dynamic('dis')
    # compute_avg_dynamics()

    # plot_ba_dis_new('BA_new')
    # plot_ba_deg_new('BA_new')
    # new_plot_probesize_otherver_ofcpshandsfs()
    # new_plot_probesize_otherver_of1and2()
    # plot_graph()
    # compute_avg()
    # new_plot_tempral_alphachange('dis')
    # new_plot_tempral_alphachange('deg')
    # new_plot_probesize()
    # new_clean_tempral_base_probesize()
    # new_plot_tempoarl_evolution('dis')
    # new_plot_tempoarl_evolution('dis')
    # new_plot_ws_dis_1graph()
    # new_plot_ws_deg()
    # new_temporal_data_clean()
    # plot_ba_dis('BA_new')
    # plot_ba_deg('BA_new')

    # plot_add_tempral(8)
    # plot_tempral_relationship()
    # rewrite_tempral_auc_and_omega_data('deg')
    # plot_tempral_alphachange('deg')
    # plot_tempral_probsize()
    # graph_type = ['contacts-prox-high-school-2013', 'email-dnc', 'SFHH-conf-sensor']
    # link_type = ['dis', 'deg']
    # for i in graph_type:
    #     for j in link_type:
    #         clean_tempral_base_probesize(i, j)


    # plot_ws_insert_dis('WS')
    # cluster_network()
    # plot_ws_deg('WS')
    # plot_ws_dis('WS')
    # graph_type = ['new FW1', 'new FW2', 'new FW3', 'sampled SmaGri', 'sampled Kohonen', 'sampled Delicious']
    # link_type = ['ra', 'rw', 'fl', 'jaccard', 'lpi']
    # link_type = ['rwr']
    # for i in graph_type:
    #     for j in link_type:
    #         clean_base_probesize(i, j)
    #         clean_base_probesize_errorbar(i, j)
    plot_graph()

    #rewrite_auc_and_omega_data('dis')
    # sample_samgri()
    # plot_deg_dis_0('BA')
    # plot_ba_dis('BA')
    # plot_ba_deg('BA')
    # plot_relationship()
    # compute_auc_and_omega()
    # rewrite_alphachange('deg')
    # graph_list = ['sampled SciMet']
    # for i in graph_list:
    #     plot_graph()
    #plot_alphachange('deg')

    # plot_probsize()
    # clean_base_probesize('new FW2', 'dis')
    # clean_base_probesize_errorbar('new FW2', 'dis')
    # graph_type = ['new FW1', 'new FW2', 'new FW3', 'sampled SmaGri', 'sampled Kohonen', 'sampled Delicious']
    # link_type = ['dis', 'deg']
    # for i in graph_type:
    #     for j in link_type:
    #         clean_base_probesize(i, j)
    #         clean_base_probesize_errorbar(i, j)
    # plot_probsize()