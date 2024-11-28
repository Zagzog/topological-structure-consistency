# -*- coding:utf-8 -*-
"""
created by zazo
2024.1.8
"""
import networkx as nx
import random
import numpy as np
from matplotlib import pyplot as plt
import os
import pandas as pd
import pyreadr


def structural_distribution_consistency(feature_previous, feature_now, alpha):
    feature_dif = []
    add_index = 0
    for i in range(len(feature_now)):
        # if feature_previous[i] - feature_now[i] != 0:
        #     feature_dif.append(exp(-alpha * i) * (feature_previous[i] - feature_now[i]))
        try:
            feature_previous[i]
        except IndexError as e:  # 未捕获到异常，程序直接报错
            add_index = 1
        if i == 0:
            continue
        if add_index == 0:
            if feature_previous[i] - feature_now[i] != 0:
                feature_dif.append((i**(-alpha)) * (feature_previous[i] - feature_now[i]))
        else:
            if feature_previous[i] - feature_now[i] != 0:
                feature_dif.append((i**(-alpha)) * (0 - feature_now[i]))
    feature_structural = abs(sum(feature_dif))
    return feature_structural


def generate_org_ws_network(N, K):
    G = nx.Graph()
    for i in range(N):
        for j in range(1, int(K/2) + 1):
            if i - j >= 1 and i + j < N:
                G.add_edge(i, i - j)
                G.add_edge(i, i + j)
            elif i - j < 1:
                G.add_edge(i, i + N - j)
                G.add_edge(i, i + j)
            elif i + j >= N:
                G.add_edge(i, i + j - N)
                G.add_edge(i, i - j)
    return G


def ws_network(N, p_ws, alpha, K):
    degree_average = []
    distance_average = []
    degree_n = []
    dis_n = []
    for tier_n in range(50):
        G = generate_org_ws_network(N, 6)

        edge_list = list(G.edges())
        node_list = list(G.nodes())

        degree_omega = []
        distance_omega = []

        degree_feature_previous_list = [0 for i in range(N)]
        distance_feature_previous_list = [0 for i in range(N+1)]
        degree_feature_now_list = [0 for i in range(N)]

        # 计算最开始时度一致性
        degree_dic = {}
        for node in node_list:
            degree_dic[node] = G.degree(node)
            degree_feature_previous_list[G.degree(node)] += 1
            degree_feature_now_list[G.degree(node)] += 1
        for item in range(len(degree_feature_previous_list)):
            degree_feature_previous_list[item] = degree_feature_previous_list[item] / N
            degree_feature_now_list[item] = degree_feature_now_list[item] / N

        # 断边重连
        for i in range(N):
            for j in range(1, int(K/2)+1):
                r = random.random()
                if r <= p_ws:
                    while True:
                        add_node = random.choice(node_list)
                        if add_node != i and (i, add_node) not in G.edges():
                            break
                    if i + j < N:
                        G.remove_edge(i, i + j)
                        degree_feature_now_list[degree_dic[i + j]] -= 1 / N
                        degree_dic[i + j] -= 1
                    else:
                        G.remove_edge(i, i+j-N)
                        degree_feature_now_list[degree_dic[i+j-N]] -= 1 / N
                        degree_dic[i+j-N] -= 1

                    G.add_edge(i, add_node)
                    degree_feature_now_list[degree_dic[add_node] + 1] += 1 / N

                    degree_s = structural_distribution_consistency(degree_feature_previous_list,
                                                                   degree_feature_now_list,
                                                                   alpha)
                    degree_omega.append(degree_s)

                    # 计算新网络的距离一致性

                    # 更新状态
                    for item in range(N):
                        degree_feature_previous_list[item] = degree_feature_now_list[item]
                    degree_feature_now_list = [0 for i in range(N)]
                else:
                    degree_omega.append(0)

        if len(degree_average) < len(degree_omega):
            degree_average = [0 for i in range(len(degree_omega))]
            degree_n = [0 for i in range(len(degree_omega))]

        for i1 in range(len(degree_omega)):
            degree_average[i1] += degree_omega[i1]
            if degree_omega[i1] != 0:
                degree_n[i1] += 1

        path = 'F:/论文/lp-new/WS_new/re_probability' + str(p_ws) + '/' + str(alpha) + '/number' + str(tier_n)
        folder = os.path.exists(path)
        if not folder:
            os.makedirs(path)
        f = open(path + '/deg_index.txt', 'w')
        for value in degree_omega:
            f.write(str(value) + '\n')
        f.close()

    # 计算50次平均结果
    for i in range(len(degree_average)):
        if degree_n[i] != 0:
            degree_average[i] = degree_average[i] / degree_n[i]

    f = open('F:/论文/lp-new/WS_new/re_probability' + str(p_ws) + '/' + str(alpha) + '/deg_avg.txt',
             'w')
    for value in degree_average:
        if value != 0:
            f.write(str(value) + '\n')
    f.close()



def structural_distribution_consistency_exp(feature_previous, feature_now, alpha):
    feature_dif = []
    add_index = 0
    for i in range(len(feature_now)):
        # if feature_previous[i] - feature_now[i] != 0:
        #     feature_dif.append(exp(-alpha * i) * (feature_previous[i] - feature_now[i]))
        try:
            feature_previous[i]
        except IndexError as e:  # 未捕获到异常，程序直接报错
            add_index = 1
        if i == 0:
            continue
        if add_index == 0:
            if feature_previous[i] - feature_now[i] != 0:
                feature_dif.append(np.exp(-alpha * i) * (feature_previous[i] - feature_now[i]))
        else:
            if feature_previous[i] - feature_now[i] != 0:
                feature_dif.append(np.exp(-alpha * i) * (0 - feature_now[i]))
    feature_structural = abs(sum(feature_dif))
    return feature_structural


def generate_org_ws_network_exp(N, K):
    G = nx.Graph()
    for i in range(N):
        for j in range(1, int(K/2) + 1):
            if i - j >= 1 and i + j < N:
                G.add_edge(i, i - j)
                G.add_edge(i, i + j)
            elif i - j < 1:
                G.add_edge(i, i + N - j)
                G.add_edge(i, i + j)
            elif i + j >= N:
                G.add_edge(i, i + j - N)
                G.add_edge(i, i - j)
    return G


def ws_network_exp(N, p_ws, alpha, K):
    degree_average = []
    distance_average = []
    degree_n = []
    dis_n = []
    for tier_n in range(50):
        G = generate_org_ws_network(N, 6)

        edge_list = list(G.edges())
        node_list = list(G.nodes())
        degree_omega = []
        distance_omega = []
        degree_feature_previous_list = [0 for i in range(N)]
        distance_feature_previous_list = [0 for i in range(N+1)]
        degree_feature_now_list = [0 for i in range(N)]

        # 计算最开始时度一致性
        degree_dic = {}
        for node in node_list:
            degree_dic[node] = G.degree(node)
            degree_feature_previous_list[G.degree(node)] += 1
            degree_feature_now_list[G.degree(node)] += 1
        for item in range(len(degree_feature_previous_list)):
            degree_feature_previous_list[item] = degree_feature_previous_list[item] / N
            degree_feature_now_list[item] = degree_feature_now_list[item] / N

        # 计算最开始时距离一致性
        dis_dic = dict(nx.all_pairs_shortest_path_length(G))
        for node1 in node_list:
            if node1 not in dis_dic.keys():
                dis_dic[node1] = {}
                for node2 in node_list:
                    dis_dic[node1][node2] = N
            else:
                for node2 in node_list:
                    if node2 not in dis_dic[node1].keys():
                        dis_dic[node1][node2] = N

        for node1 in node_list:
            for node2 in node_list:
                distance_feature_previous_list[dis_dic[node1][node2]] += 1
        for item in range(len(distance_feature_previous_list)):
            distance_feature_previous_list[item] = distance_feature_previous_list[item] / (N ** 2)

        # 断边重连
        for i in range(N):
            for j in range(1, int(K/2)+1):
                r = random.random()
                if r <= p_ws:
                    while True:
                        add_node = random.choice(node_list)
                        if add_node != i and (i, add_node) not in G.edges():
                            break
                    if i + j < N:
                        G.remove_edge(i, i + j)
                        degree_feature_now_list[degree_dic[i + j]] -= 1 / N
                        degree_dic[i + j] -= 1
                    else:
                        G.remove_edge(i, i+j-N)
                        degree_feature_now_list[degree_dic[i+j-N]] -= 1 / N
                        degree_dic[i+j-N] -= 1

                    G.add_edge(i, add_node)
                    degree_feature_now_list[degree_dic[add_node] + 1] += 1 / N

                    degree_s = structural_distribution_consistency_exp(degree_feature_previous_list,
                                                                   degree_feature_now_list,
                                                                   alpha)
                    degree_omega.append(degree_s)

                    # 计算新网络的距离一致性
                    distance_feature_now_list = [0 for i in range(N + 1)]
                    dis_dic_new = dict(nx.all_pairs_shortest_path_length(G))
                    for node1 in node_list:
                        if node1 not in dis_dic_new.keys():
                            dis_dic_new[node1] = {}
                            for node2 in node_list:
                                dis_dic_new[node1][node2] = N
                        else:
                            for node2 in node_list:
                                if node2 not in dis_dic_new[node1].keys():
                                    dis_dic_new[node1][node2] = N

                    for node1 in node_list:
                        for node2 in node_list:
                            distance_feature_now_list[dis_dic_new[node1][node2]] += 1

                    for item in range(len(distance_feature_now_list)):
                        distance_feature_now_list[item] = distance_feature_now_list[item] / (N ** 2)

                    distance_s = structural_distribution_consistency_exp(distance_feature_previous_list,
                                                                     distance_feature_now_list, alpha)
                    distance_omega.append(distance_s)

                    # 更新状态
                    for item in range(N):
                        distance_feature_previous_list[item] = distance_feature_now_list[item]
                        degree_feature_previous_list[item] = degree_feature_now_list[item]
                else:
                    distance_omega.append(0)
                    degree_omega.append(0)

        if len(degree_average) < len(degree_omega):
            degree_average = [0 for i in range(len(degree_omega))]
            distance_average = [0 for i in range(len(distance_omega))]
            degree_n = [0 for i in range(len(degree_omega))]
            dis_n = [0 for i in range(len(degree_omega))]

        for i1 in range(len(degree_omega)):
            degree_average[i1] += degree_omega[i1]
            distance_average[i1] += distance_omega[i1]
            if degree_omega[i1] != 0:
                degree_n[i1] += 1
                dis_n[i1] += 1

        path = 'F:/论文/lp_new/WS_ver2/re_probability' + str(p_ws) + '/' + str(alpha) + '/number' + str(tier_n)
        folder = os.path.exists(path)
        if not folder:
            os.makedirs(path)
        f = open(path + '/dis_index.txt', 'w')
        for value in distance_omega:
            f.write(str(value) + '\n')
        f.close()
        f = open(path + '/deg_index.txt', 'w')
        for value in degree_omega:
            f.write(str(value) + '\n')
        f.close()

    # 计算50次平均结果
    for i in range(len(degree_average)):
        if degree_n[i] != 0:
            degree_average[i] = degree_average[i] / degree_n[i]
            distance_average[i] = distance_average[i] / dis_n[i]

    f = open('F:/论文/lp_new/WS_ver2/re_probability' + str(p_ws) + '/' + str(alpha) + '/dis_avg.txt',
             'w')
    for value in distance_average:
        if value != 0:
            f.write(str(value) + '\n')
    f.close()
    f = open('F:/论文/lp_new/WS_ver2/re_probability' + str(p_ws) + '/' + str(alpha) + '/deg_avg.txt',
             'w')
    for value in degree_average:
        if value != 0:
            f.write(str(value) + '\n')
    f.close()


if __name__ == '__main__':
    alpha_list = [-8, -6, -4, -2, 0, 2, 4, 6, 8]
    p_list = [0.01, 0.02, 0.05, 0.1, 0.25, 0.5, 0.75, 1]
    for alp in alpha_list:
        for p_ws in p_list:
            print("alpha:", alp)
            ws_network(1000, p_ws, alp, 6)