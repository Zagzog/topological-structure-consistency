# -*- coding:utf-8 -*-
"""
created by zazo
2023.5.30
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


def build_temporal_network(graph_loc, dir_type, tim_size):
    time_edge_dict = dict()
    f = open(graph_loc, 'r', encoding='UTF-8')
    lines = f.readlines()
    f.close()
    node_dict = {}
    edge_dict = {}
    for line in lines:
        item = line.strip('\n').split(',')
        if item[2] not in time_edge_dict:
            time_edge_dict[item[2]] = [(item[0], item[1])]
        else:
            time_edge_dict[item[2]].append((item[0], item[1]))

        edge_dict[(item[0], item[1])] = 0
        if item[0] not in node_dict:
            node_dict[item[0]] = item[0]
        if item[1] not in node_dict:
            node_dict[item[1]] = item[1]

    node_list = list(node_dict.keys())

    choose_num = int(len(time_edge_dict) * (1 - tim_size))

    time_stamp_list = list(time_edge_dict.keys())

    if dir_type == 'direct':
        G = nx.DiGraph()
    else:
        G = nx.Graph()

    rest_time_edge = dict()
    rest_time_edge_list = []
    for i in range(len(time_stamp_list)):
        time_stamp = time_stamp_list[i]
        if i <= choose_num:
            for edge in time_edge_dict[time_stamp]:
                G.add_edge(edge[0], edge[1])
        else:
            rest_time_edge[time_stamp] = time_edge_dict[time_stamp]
            for edge_sample in time_edge_dict[time_stamp]:
                rest_time_edge_list.append(edge_sample)

    return G, rest_time_edge_list, node_list, edge_dict


def structural_distribution_consistency(feature_previous, feature_now, alpha, type, num_node):
    feature_dif = []
    add_index = 0
    temp_now = [0 for _ in range(len(feature_now))]
    temp_pre = [0 for _ in range(len(feature_previous))]
    size_now = len(feature_now)
    size_pre = len(feature_previous)
    true_now = 0
    true_pre = 0
    for i in range(size_now - 1):
        true_now += feature_now[i]
    for i in range(size_pre - 1):
        true_pre += feature_previous[i]

    for i in range(size_now):
        if type == "deg":
            temp_now[i] = feature_now[i] / num_node
        elif type == "dis":
            temp_now[i] = feature_now[i] / true_now

    for i in range(size_pre):
        if type == "deg":
            temp_pre[i] = feature_previous[i] / num_node
        elif type == "dis":
            temp_pre[i] = feature_previous[i] / true_pre

    for i in range(size_now):
        try:
            temp_pre[i]
        except IndexError as e:  # 未捕获到异常，程序直接报错
            add_index = 1

        if add_index == 0:
            if i == num_node:
                continue
            elif temp_pre[i] - temp_now[i] != 0 and i != 0:
                feature_dif.append((i**(-alpha)) * (temp_pre[i] - temp_now[i]))
        else:
            if i == num_node:
                continue
            elif temp_pre[i] - temp_now[i] != 0 and i != 0:
                feature_dif.append((i**(-alpha)) * (0 - temp_now[i]))
    #print(feature_dif)
    feature_structural = abs(sum(feature_dif))
    return feature_structural


def score_compute_dis(N, distance_previous_list, add_edge, alpha, dis_dict, node_list):
    temp_dis_dict = dict()
    for i in dis_dict.keys():
        temp_dis_dict[i] = dis_dict[i]

    distance_now_list = [0 for _ in range(N + 1)]
    for node1 in node_list:
        for node2 in node_list:
            new_dis1 = temp_dis_dict[node1][add_edge[0]] + temp_dis_dict[add_edge[1]][node2] + 1
            new_dis2 = temp_dis_dict[node1][add_edge[1]] + temp_dis_dict[add_edge[0]][node2] + 1
            new_dis = min(new_dis1, new_dis2)
            if temp_dis_dict[node1][node2] > new_dis:
                temp_dis_dict[node1][node2] = new_dis

    for node1_length in range(len(node_list)):
        for node2_length in range(len(node_list)):
            dis = temp_dis_dict[node_list[node1_length]][node_list[node2_length]]
            distance_now_list[dis] += 1

    distance_s = structural_distribution_consistency(distance_previous_list, distance_now_list, alpha, "dis", N)
    if distance_s == 0:
        score = np.inf
    else:
        score = 1 / distance_s

    return score


def score_compute_degree(N, degree_previous_list, add_edge, alpha, deg_dict, type):
    degree_now_list = [0 for i in range(N)]
    for i in range(len(degree_previous_list)):
        degree_now_list[i] = degree_previous_list[i]

    if type == 'direct':
        degree_now_list[deg_dict[add_edge[0]]] -= 1
        degree_now_list[deg_dict[add_edge[0]] + 1] += 1
    else:
        degree_now_list[deg_dict[add_edge[0]]] -= 1
        degree_now_list[deg_dict[add_edge[1]]] -= 1
        degree_now_list[deg_dict[add_edge[0]] + 1] += 1
        degree_now_list[deg_dict[add_edge[1]] + 1] += 1

    degree_s = structural_distribution_consistency(degree_previous_list, degree_now_list, alpha, "deg", N)
    if degree_s == 0:
        score = np.inf
    else:
        score = 1 / degree_s

    return score


def temporal_link_prediction(alpha, tim_size):
    graph_list = ['contacts-prox-high-school-2013', 'email-dnc', 'SFHH-conf-sensor']
    graph_type = ['undirect', 'direct', 'undirect']
    for i in range(len(graph_list)):
        print(graph_list[i])
        path = "F:/论文/lp-new/时序网络-补充/" + graph_list[i] + '.edges'
        org_graph, rest_time_edge, node_list, edge_dict = build_temporal_network(path, graph_type[i], tim_size)
        N = nx.number_of_nodes(org_graph)

        supplement_edge = []
        if graph_type[i] == 'direct':
            for node1 in node_list:
                for node2 in node_list:
                    if node1 != node2:
                        if (node1, node2) not in edge_dict:
                            supplement_edge.append((node1, node2))
        else:
            for node1_loc in range(N):
                for node2_loc in range(node1_loc, N):
                    node1 = node_list[node1_loc]
                    node2 = node_list[node2_loc]
                    if node1 != node2:
                        if (node1, node2) not in edge_dict:
                            supplement_edge.append((node1, node2))

        dis_dic = dict(nx.all_pairs_shortest_path_length(org_graph))
        for node1 in node_list:
            if node1 not in dis_dic.keys():
                dis_dic[node1] = {}
                for node2 in node_list:
                    dis_dic[node1][node2] = N
            else:
                for node2 in node_list:
                    if node2 not in dis_dic[node1].keys():
                        dis_dic[node1][node2] = N

        distance_previous_list = [0 for _ in range(N + 1)]
        for node1 in node_list:
            for node2 in node_list:
                distance_previous_list[dis_dic[node1][node2]] += 1

        degree_dic = {}
        if graph_type[i] == 'direct':
            for node in org_graph.nodes():
                degree_dic[node] = org_graph.out_degree(node)
        else:
            for node in org_graph.nodes():
                degree_dic[node] = org_graph.degree(node)

        for node in node_list:
            if node not in degree_dic.keys():
                degree_dic[node] = 0

        degree_previous_list = [0 for i in range(N)]
        for key in degree_dic.keys():
            degree_previous_list[degree_dic[key]] += 1

        n_1_dis = 0
        n_2_dis = 0
        n_1_deg = 0
        n_2_deg = 0
        store2_dis = []
        store2_deg = []
        for all_n in range(len(rest_time_edge)):
            test_edge = rest_time_edge[all_n]
            supplement_e = random.choice(supplement_edge)

            score_test_dis = score_compute_dis(N, distance_previous_list, test_edge, alpha, dis_dic, node_list)
            score_supplement_dis = score_compute_dis(N, distance_previous_list, supplement_e, alpha, dis_dic, node_list)
            if score_test_dis != np.inf:
                store2_dis.append(1 / score_test_dis)
            else:
                store2_dis.append(0)

            if score_test_dis > score_supplement_dis:
                n_1_dis += 1
            elif score_test_dis == score_supplement_dis:
                n_2_dis += 1

            score_test_deg = score_compute_degree(N, degree_previous_list, test_edge, alpha, degree_dic, node_list)
            score_supplement_deg = score_compute_degree(N, degree_previous_list, supplement_e, alpha, degree_dic, node_list)
            if score_test_deg != np.inf:
                store2_deg.append(1 / score_test_deg)
            else:
                store2_deg.append(0)
            if score_test_deg > score_supplement_deg:
                n_1_deg += 1
            elif score_test_deg == score_supplement_deg:
                n_2_deg += 1

        auc_dis = (n_1_dis + 0.5 * n_2_dis) / len(rest_time_edge)
        auc_deg = (n_1_deg + 0.5 * n_2_deg) / len(rest_time_edge)

        path = 'F:/论文/lp-new/link-prediction/' + graph_list[i] + '/alpha' + str(alpha) + '/dis/time_size' + str(
            tim_size)
        folder = os.path.exists(path)
        if not folder:
            os.makedirs(path)
        f = open(path + '/auc.txt', 'w')
        f.write(str(auc_dis))
        f.close()

        f = open(path + '/omega_dis.txt', 'w')
        f.write(str(sum(store2_dis) / len(rest_time_edge)))
        f.close()

        path = 'F:/论文/lp-new/link-prediction/' + graph_list[i] + '/alpha' + str(alpha) + '/deg/time_size' + str(
            tim_size)
        folder = os.path.exists(path)
        if not folder:
            os.makedirs(path)
        f = open(path + '/auc.txt', 'w')
        f.write(str(auc_deg))
        f.close()

        f = open(path + '/omega_deg.txt', 'w')
        f.write(str(sum(store2_deg) / len(rest_time_edge)))
        f.close()


def build_temporal_network_time_vary(graph_loc, dir_type):
    time_edge_dict = dict()
    f = open(graph_loc, 'r', encoding='UTF-8')
    lines = f.readlines()
    f.close()
    node_dict = {}

    for line in lines:
        item = line.strip('\n').split(',')
        if item[2] not in time_edge_dict:
            time_edge_dict[item[2]] = [(item[0], item[1])]
        else:
            time_edge_dict[item[2]].append((item[0], item[1]))

        if item[0] not in node_dict:
            node_dict[item[0]] = item[0]
        if item[1] not in node_dict:
            node_dict[item[1]] = item[1]

    node_list = list(node_dict.keys())
    graph_dict = {}
    for key in time_edge_dict.keys():
        if dir_type == 'direct':
            G = nx.DiGraph()
        else:
            G = nx.Graph()

        for edge in time_edge_dict[key]:
            G.add_edge(edge[0], edge[1])
        graph_dict[key] = G

    time_stamp_list = list(time_edge_dict.keys())
    rest_time_edge_dict = {}
    for i in range(1, len(time_stamp_list)):
        rest_time_edge_dict[time_stamp_list[i-1]] = []
        for edge1 in time_edge_dict[time_stamp_list[i]]:
            if edge1 not in time_edge_dict[time_stamp_list[i-1]]:
                rest_time_edge_dict[time_stamp_list[i-1]].append(edge1)

    return graph_dict, rest_time_edge_dict, node_list, time_edge_dict


def temporal_link_prediction_time_vary(alpha):
    graph_list = ['contacts-prox-high-school-2013', 'email-dnc', 'SFHH-conf-sensor']
    graph_type = ['undirect', 'direct', 'undirect']
    for i in range(len(graph_list)):
        path = "F:/论文/lp-new/时序网络-补充/" + graph_list[i] + '.edges'
        org_graph_dict, rest_time_edge_dict, node_list, edge_time_dict = \
            build_temporal_network_time_vary(path, graph_type[i])
        print(graph_list[i])
        auc_deg_list =[]
        auc_dis_list = []
        omega_dis_list = []
        omega_deg_list = []
        for j in rest_time_edge_dict.keys():

            N = len(node_list)

            supplement_edge = []
            if graph_type[i] == 'direct':
                for node1 in node_list:
                    for node2 in node_list:
                        if node1 != node2:
                            if (node1, node2) not in edge_time_dict[j]:
                                supplement_edge.append((node1, node2))
            else:
                for node1_loc in range(N):
                    for node2_loc in range(node1_loc, N):
                        node1 = node_list[node1_loc]
                        node2 = node_list[node2_loc]
                        if node1 != node2:
                            if (node1, node2) not in edge_time_dict[j]:
                                supplement_edge.append((node1, node2))

            dis_dic = dict(nx.all_pairs_shortest_path_length(org_graph_dict[j]))
            for node1 in node_list:
                if node1 not in dis_dic.keys():
                    dis_dic[node1] = {}
                    for node2 in node_list:
                        dis_dic[node1][node2] = N
                else:
                    for node2 in node_list:
                        if node2 not in dis_dic[node1].keys():
                            dis_dic[node1][node2] = N

            distance_previous_list = [0 for _ in range(N + 1)]
            for node1 in node_list:
                for node2 in node_list:
                    distance_previous_list[dis_dic[node1][node2]] += 1

            degree_dic = {}
            if graph_type[i] == 'direct':
                for node in org_graph_dict[j].nodes():
                    degree_dic[node] = org_graph_dict[j].out_degree(node)
            else:
                for node in org_graph_dict[j].nodes():
                    degree_dic[node] = org_graph_dict[j].degree(node)

            for node in node_list:
                if node not in degree_dic.keys():
                    degree_dic[node] = 0

            degree_previous_list = [0 for i in range(N)]
            for key in degree_dic.keys():
                degree_previous_list[degree_dic[key]] += 1

            n_1_dis = 0
            n_2_dis = 0
            n_1_deg = 0
            n_2_deg = 0
            store2_dis = []
            store2_deg = []
            for all_n in range(len(rest_time_edge_dict[j])):
                test_edge = rest_time_edge_dict[j][all_n]
                supplement_e = random.choice(supplement_edge)

                score_test_dis = score_compute_dis(N, distance_previous_list, test_edge, alpha, dis_dic, node_list)
                score_supplement_dis = score_compute_dis(N, distance_previous_list, supplement_e, alpha, dis_dic, node_list)
                if score_test_dis != np.inf:
                    store2_dis.append(1 / score_test_dis)
                else:
                    store2_dis.append(0)

                if score_test_dis > score_supplement_dis:
                    n_1_dis += 1
                elif score_test_dis == score_supplement_dis:
                    n_2_dis += 1

                score_test_deg = score_compute_degree(N, degree_previous_list, test_edge, alpha, degree_dic, node_list)
                score_supplement_deg = score_compute_degree(N, degree_previous_list, supplement_e, alpha, degree_dic, node_list)
                if score_test_deg != np.inf:
                    store2_deg.append(1 / score_test_deg)
                else:
                    store2_deg.append(0)
                if score_test_deg > score_supplement_deg:
                    n_1_deg += 1
                elif score_test_deg == score_supplement_deg:
                    n_2_deg += 1

            if len(rest_time_edge_dict[j]) != 0:
                auc_dis = (n_1_dis + 0.5 * n_2_dis) / len(rest_time_edge_dict[j])
                auc_deg = (n_1_deg + 0.5 * n_2_deg) / len(rest_time_edge_dict[j])
                auc_deg_list.append(auc_deg)
                auc_dis_list.append(auc_dis)
                omega_dis_list.append(sum(store2_dis) / len(rest_time_edge_dict[j]))
                omega_deg_list.append(sum(store2_deg) / len(rest_time_edge_dict[j]))

        path = 'F:/论文/lp-new/link-prediction-time-vary/' + graph_list[i] + '/alpha' + str(alpha) + '/'
        folder = os.path.exists(path)
        if not folder:
            os.makedirs(path)

        f = open(path + '/deg_auc.txt', 'w')
        for j in range(len(auc_deg_list)):
            f.write(str(auc_deg_list[j]) + '\n')
        f.close()

        f = open(path + '/dis_auc.txt', 'w')
        for j in range(len(auc_dis_list)):
            f.write(str(auc_dis_list[j]) + '\n')
        f.close()

        f = open(path + '/omega_deg.txt', 'w')
        for j in range(len(omega_deg_list)):
            f.write(str(omega_deg_list[j]) + '\n')
        f.close()

        f = open(path + '/omega_dis.txt', 'w')
        for j in range(len(omega_dis_list)):
            f.write(str(omega_dis_list[j]) + '\n')
        f.close()


if __name__ == "__main__":
    # alpha_list = [-8]
    # for alpha in alpha_list:
    #     print(alpha)
    #     temporal_link_prediction_time_vary(alpha)

