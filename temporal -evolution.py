# -*- coding:utf-8 -*-
"""
created by zazo
2023.5.30
"""
import os
import numpy as np
import pandas as pd
import networkx as nx
import random
import pyreadr


def structural_distribution_consistency(feature_previous, feature_now, alpha, type, num_node):
    feature_dif = []
    add_index = 0
    temp_now = [0 for _ in range(len(feature_now))]
    temp_pre = [0 for _ in range(len(feature_previous))]
    size_now = len(feature_now)
    size_pre = len(feature_previous)
    true_now = 0
    true_pre = 0
    for i in range(size_now):
        true_now += feature_now[i]
    for i in range(size_pre):
        true_pre += feature_previous[i]

    if alpha == 0:
        if type == "deg":
            feature_structural = abs(sum(feature_previous) / num_node - sum(feature_now) / num_node)
        elif type == "dis":
            feature_structural = abs(sum(feature_previous) / true_pre - sum(feature_now) / true_now)
    else:
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
            # if feature_previous[i] - feature_now[i] != 0:
            #     feature_dif.append(exp(-alpha * i) * (feature_previous[i] - feature_now[i]))
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


def build_temporal_network_time_vary(graph_loc, dir_type):
    time_edge_dict = dict()
    f = open(graph_loc, 'r', encoding='UTF-8')
    lines = f.readlines()
    f.close()
    node_dict = {}

    for line in lines:
        item = line.strip('\n').split(',')
        if item[0] == item[1]:
            continue
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

        for node in node_list:
            if node not in G.nodes():
                G.add_node(node)
        graph_dict[key] = G

    time_stamp_list = list(time_edge_dict.keys())
    rest_time_edge_dict = {}
    for i in range(1, len(time_stamp_list)):
        rest_time_edge_dict[time_stamp_list[i-1]] = []
        for edge1 in time_edge_dict[time_stamp_list[i]]:
            if edge1 not in time_edge_dict[time_stamp_list[i-1]]:
                rest_time_edge_dict[time_stamp_list[i-1]].append(edge1)

    return graph_dict, rest_time_edge_dict, node_list, time_edge_dict


def temp_network_add_1(graph_loc):
    graph_dict = {} #存储每时刻的时序网络
    time_edge_dict = {} #存储每时刻的边信息
    rest_time_edge_dict = {} #存储下一时刻会出现的边，作为计算auc的测试边集
    re_label = {}  # 对节点标签重标号
    ans = 0
    graph_name = ['spr', 'sum', 'fall', 'win']

    for gn in range(len(graph_name)):
        file_path = graph_loc + "/1998_" + graph_name[gn] + "_node_list.csv"
        data_frame = pd.read_csv(file_path)
        for i in data_frame.index.values:
            if data_frame.values[i, 0] not in re_label:
                re_label[data_frame.values[i, 0]] = ans
                ans += 1

    node_list = [i for i in range(ans)] #存储所有节点信息

    for gn in range(len(graph_name)):
        file_path = graph_loc + "/1998_" + graph_name[gn] + "_edge_list.csv"
        data_frame = pd.read_csv(file_path)

        time_edge_dict[gn] = []
        for i in data_frame.index.values:
            edge = (re_label[data_frame.values[i, 0]], re_label[data_frame.values[i, 1]])
            if edge[0] == edge[1]:
                continue
            if edge not in time_edge_dict[gn]:
                time_edge_dict[gn].append(edge)

    for key in time_edge_dict.keys():
        G = nx.DiGraph()
        for edge in time_edge_dict[key]:
            G.add_edge(edge[0], edge[1])

        for node in node_list:
            if node not in G.nodes():
                G.add_node(node)
        graph_dict[key] = G

    # print(nx.number_of_edges(graph_dict[0]), nx.number_of_edges(graph_dict[1]))

    for i in range(1, 4):
        rest_time_edge_dict[i-1] = []
        for edge1 in time_edge_dict[i]:
            if edge1 not in time_edge_dict[i-1]:
                rest_time_edge_dict[i-1].append(edge1)

    return graph_dict, rest_time_edge_dict, node_list, time_edge_dict


def temp_network_add_2(graph_loc):
    graph_dict = {}  # 存储每时刻的时序网络
    time_edge_dict = {}  # 存储每时刻的边信息
    rest_time_edge_dict = {}  # 存储下一时刻会出现的边，作为计算auc的测试边集
    graph_name = ['1', '2', '3', '4']

    old_df = [['0']*26 for i in range(26)]
    for gn in range(len(graph_name)):
        file_path = graph_loc + "/klas12b-net-" + graph_name[gn] + ".dat"
        data_frame = pd.read_csv(file_path, header=None)

        time_edge_dict[gn] = []
        for i in data_frame.index.values:
            column_value = data_frame.values[i, 0].split()
            for j in range(len(column_value)):
                if i == j:
                    continue
                if column_value[j] == '1':
                    time_edge_dict[gn].append((i, j))
                    old_df[i][j] = column_value[j]
                elif column_value[j] == '9' and old_df[i][j] == '1':
                    time_edge_dict[gn].append((i, j))
                    old_df[i][j] = '1'

    node_list = [i for i in range(26)]

    for key in time_edge_dict.keys():
        G = nx.DiGraph()
        for edge in time_edge_dict[key]:
            G.add_edge(edge[0], edge[1])

        for node in node_list:
            if node not in G.nodes():
                G.add_node(node)

        graph_dict[key] = G

    for i in range(1, 4):
        rest_time_edge_dict[i-1] = []
        for edge1 in time_edge_dict[i]:
            if edge1 not in time_edge_dict[i-1]:
                rest_time_edge_dict[i-1].append(edge1)

    return graph_dict, rest_time_edge_dict, node_list, time_edge_dict


def temp_network_add_3(graph_loc):
    graph_dict = {}  # 存储每时刻的时序网络
    time_edge_dict = {}  # 存储每时刻的边信息
    rest_time_edge_dict = {}  # 存储下一时刻会出现的边，作为计算auc的测试边集
    graph_name = [str(i) for i in range(93, 103)]

    re_label = {}  # 对节点标签重标号
    ans = 0

    for gn in range(len(graph_name)):
        file_path = graph_loc + "/a_sign_of_the_times.xlsx"
        data_frame = pd.read_excel(file_path, sheet_name="S" + graph_name[gn], header=None)

        for i in data_frame.index.values:
            if data_frame.values[i, 0] not in re_label:
                re_label[data_frame.values[i, 0]] = ans
                ans += 1

        time_edge_dict[gn] = []
        for i in data_frame.index.values:
            for j in data_frame.columns.values:
                if j == 0:
                    continue
                if j > i:
                    if data_frame.values[i, j] == 1 or data_frame.values[i, j] == -1:
                        time_edge_dict[gn].append((i, j))

    node_list = [i for i in range(ans)]

    for key in time_edge_dict.keys():
        G = nx.Graph()
        for edge in time_edge_dict[key]:
            G.add_edge(edge[0], edge[1])

        for node in node_list:
            if node not in G.nodes():
                G.add_node(node)

        graph_dict[key] = G

    for i in range(1, len(time_edge_dict)):
        rest_time_edge_dict[i-1] = []
        for edge1 in time_edge_dict[i]:
            if edge1 not in time_edge_dict[i-1]:
                rest_time_edge_dict[i-1].append(edge1)

    return graph_dict, rest_time_edge_dict, node_list, time_edge_dict


def temp_network_add_4(graph_loc):
    graph_dict = {}  # 存储每时刻的时序网络
    time_edge_dict = {}  # 存储每时刻的边信息
    rest_time_edge_dict = {}  # 存储下一时刻会出现的边，作为计算auc的测试边集
    graph_name = ['2006', '2015']

    re_label = {}  # 对节点标签重标号
    ans = 0

    for gn in range(len(graph_name)):
        file_path = graph_loc + "/" + graph_name[gn] + "_FIFA_1M.csv"
        data_frame = pd.read_csv(file_path, header=None)

        for i in data_frame.index.values:
            if data_frame.values[i, 0] not in re_label:
                re_label[data_frame.values[i, 0]] = ans
                ans += 1

        time_edge_dict[gn] = []
        for i in data_frame.index.values:
            if i == 0:
                continue
            for j in data_frame.columns.values:
                if j == 0:
                    continue
                if j > i and data_frame.values[i, j] != "0":
                    time_edge_dict[gn].append((i, j))

    node_list = [i for i in range(ans)]
    for key in time_edge_dict.keys():
        G = nx.Graph()
        for edge in time_edge_dict[key]:
            G.add_edge(edge[0], edge[1])

        for node in node_list:
            if node not in G.nodes():
                G.add_node(node)
        graph_dict[key] = G

    for i in range(1, len(time_edge_dict)):
        rest_time_edge_dict[i-1] = []
        for edge1 in time_edge_dict[i]:
            if edge1 not in time_edge_dict[i-1]:
                rest_time_edge_dict[i-1].append(edge1)

    return graph_dict, rest_time_edge_dict, node_list, time_edge_dict


def temp_network_add_5(graph_loc):
    graph_dict = {} #存储每时刻的时序网络
    time_edge_dict = {} #存储每时刻的边信息
    rest_time_edge_dict = {} #存储下一时刻会出现的边，作为计算auc的测试边集

    graph_name = ['1', '2']

    node_list = [] #存储所有节点信息

    for gn in range(len(graph_name)):
        file_path = graph_loc + "/personal relationships" + graph_name[gn] + ".txt"
        data_frame = pd.read_csv(file_path, sep='\t', header=None)

        time_edge_dict[gn] = []

        for i in data_frame.index.values:
            temp_store = data_frame.values[i, 0].split()
            edge = (temp_store[0], temp_store[1])

            if temp_store[0] not in node_list:
                node_list.append(temp_store[0])
            if temp_store[1] not in node_list:
                node_list.append(temp_store[1])

            if edge[0] == edge[1]:
                continue
            if edge not in time_edge_dict[gn]:
                time_edge_dict[gn].append(edge)

    for key in time_edge_dict.keys():
        G = nx.DiGraph()
        for edge in time_edge_dict[key]:
            G.add_edge(edge[0], edge[1])

        for node in node_list:
            if node not in G.nodes():
                G.add_node(node)

        graph_dict[key] = G

    for i in range(1, len(time_edge_dict)):
        rest_time_edge_dict[i-1] = []
        for edge1 in time_edge_dict[i]:
            if edge1 not in time_edge_dict[i-1]:
                rest_time_edge_dict[i-1].append(edge1)

    return graph_dict, rest_time_edge_dict, node_list, time_edge_dict


def temp_network_add_6(graph_loc):
    graph_dict = {} #存储每时刻的时序网络
    time_edge_dict = {} #存储每时刻的边信息
    rest_time_edge_dict = {} #存储下一时刻会出现的边，作为计算auc的测试边集

    file_path = graph_loc + "/Glasgow-friendship.RData"
    file = pyreadr.read_r(file_path)
    graph_name = ['friendship.1', 'friendship.2', 'friendship.3']

    for gn in range(len(graph_name)):
        data_frame = file[graph_name[gn]]
        time_edge_dict[gn] = []

        row_l = len(data_frame.index.values)
        for i in range(row_l):
            for j in range(row_l):
                if i == j:
                    continue
                if data_frame.values[i, j] == 1 or data_frame.values[i, j] == 2:
                    time_edge_dict[gn].append((i, j))

    node_list = [i for i in range(160)]

    for key in time_edge_dict.keys():
        G = nx.DiGraph()
        for edge in time_edge_dict[key]:
            G.add_edge(edge[0], edge[1])

        for node in node_list:
            if node not in G.nodes():
                G.add_node(node)
        graph_dict[key] = G

    for i in range(1, len(time_edge_dict)):
        rest_time_edge_dict[i - 1] = []
        for edge1 in time_edge_dict[i]:
            if edge1 not in time_edge_dict[i - 1]:
                rest_time_edge_dict[i - 1].append(edge1)

    return graph_dict, rest_time_edge_dict, node_list, time_edge_dict


def temp_network_build(graph_loc, number_label, dir_type):
    if number_label == '1':
        return temp_network_add_1(graph_loc)
    elif number_label == '2':
        return temp_network_add_2(graph_loc)
    elif number_label == '3':
        return temp_network_add_3(graph_loc)
    elif number_label == '4':
        return temp_network_add_4(graph_loc)
    elif number_label == '5':
        return temp_network_add_5(graph_loc)
    elif number_label == '6':
        return temp_network_add_6(graph_loc)
    else:
        return build_temporal_network_time_vary(graph_loc, dir_type)


def temporal_link_prediction_time_vary(alpha):
    graph_list = ['1', '2', '3', '4', '5', '6', 'contacts-prox-high-school-2013', 'SFHH-conf-sensor']
    graph_type = ['direct', 'direct', 'undirect', 'undirect', 'direct', 'direct', 'undirect', 'undirect']
    graph_path = ['C:/Users/86178/Desktop/时序网络/1', 'C:/Users/86178/Desktop/时序网络/2/klas12b',
                  'C:/Users/86178/Desktop/时序网络/3', 'C:/Users/86178/Desktop/时序网络/4/FIFA CSV/CSV',
                  'C:/Users/86178/Desktop/时序网络/5', 'C:/Users/86178/Desktop/时序网络/6/Glasgow_data',
                  "F:/论文/lp-new/时序网络-补充/contacts-prox-high-school-2013.edges",
                  "F:/论文/lp-new/时序网络-补充/SFHH-conf-sensor.edges"
                  ]

    for i in range(6, 7):
        org_graph_dict, rest_time_edge_dict, node_list, edge_time_dict = \
            temp_network_build(graph_path[i], graph_list[i], graph_type[i])

        print('graph', graph_list[i])

        auc_deg_list = []
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

                score_test_deg = score_compute_degree(N, degree_previous_list, test_edge, alpha, degree_dic, graph_type[i])
                score_supplement_deg = score_compute_degree(N, degree_previous_list, supplement_e, alpha, degree_dic, graph_type[i])
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

        path = 'F:/论文/lp-new/link-prediction-time-vary-test/' + graph_list[i] + '/alpha' + str(alpha) + '/'
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



def build_temporal_network_time_vary_without_singlenode(graph_loc, dir_type):
    time_edge_dict = dict()
    f = open(graph_loc, 'r', encoding='UTF-8')
    lines = f.readlines()
    f.close()
    node_dict = {}

    for line in lines:
        item = line.strip('\n').split(',')
        if item[0] == item[1]:
            continue
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


def temp_network_build_without_singlenode(graph_loc, number_label, dir_type):
    if number_label == '1':
        return temp_network_add_1(graph_loc)
    elif number_label == '2':
        return temp_network_add_2(graph_loc)
    elif number_label == '3':
        return temp_network_add_3(graph_loc)
    elif number_label == '4':
        return temp_network_add_4(graph_loc)
    elif number_label == '5':
        return temp_network_add_5(graph_loc)
    elif number_label == '6':
        return temp_network_add_6(graph_loc)
    else:
        return build_temporal_network_time_vary_without_singlenode(graph_loc, dir_type)


def temporal_network_data():
    graph_list = ['contacts-prox-high-school-2013', 'SFHH-conf-sensor']
    graph_type = ['undirect', 'undirect']
    graph_path = [
                  "F:/论文/lp-new/时序网络-补充/contacts-prox-high-school-2013.edges",
                  "F:/论文/lp-new/时序网络-补充/SFHH-conf-sensor.edges"
                  ]

    for i in range(2):
        org_graph_dict, rest_time_edge_dict, node_list, edge_time_dict = \
            temp_network_build_without_singlenode(graph_path[i], graph_list[i], graph_type[i])
        midu_list = []
        d_list = []
        c_list = []
        xiaolv_list = []
        for key1 in org_graph_dict.keys():
            G = org_graph_dict[key1]
            midu = nx.density(G)
            midu_list.append(midu)

            d_dict = dict(nx.all_pairs_dijkstra_path_length(G))
            d = 0
            count = 0
            for key1 in d_dict:
                for key2 in d_dict[key1]:
                    if key1 != key2:
                        count += 1
                        d += d_dict[key1][key2]
            d = d / count

            # d = nx.average_shortest_path_length(G)
            d_list.append(d)

            c = nx.average_clustering(G)
            c_list.append(c)

            xiaolv = nx.global_efficiency(G)
            xiaolv_list.append(xiaolv)
        avg_midu = sum(midu_list) / len(midu_list)
        avg_d = sum(d_list) / len(d_list)
        avg_c = sum(c_list) / len(c_list)
        avg_xiaolv = sum(xiaolv_list) / len(xiaolv_list)
        print(avg_midu, avg_d, avg_c, avg_xiaolv)