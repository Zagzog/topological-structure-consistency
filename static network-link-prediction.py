# -*- coding:utf-8 -*-
"""
created by zazo
2023.5.11
"""
import networkx as nx
import random
import numpy as np
import os


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
    # print(sum(feature_dif))
    # print(feature_structural)
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


def division_set_dis(alpha, probe_size, graph_name):
    G = nx.DiGraph()
    f = open('F:/论文/lp-new/network/' + graph_name + '.txt', 'r')
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
    test_number = int(M * probe_size)
    auc_list = []
    store1 = []
    for z in range(20):
        #print("iter: ", z)
        test_set = random.sample(edge_set, test_number)
        train_set = []
        for edge1 in edge_set:
            if edge1 not in test_set:
                train_set.append(edge1)

        supplement_edge = []
        for node1 in node_set:
            for node2 in node_set:
                if node1 != node2:
                    if (node1, node2) not in edge_set:
                        supplement_edge.append((node1, node2))

        train_graph = nx.DiGraph()
        for edge2 in train_set:
            train_graph.add_edge(edge2[0], edge2[1])
        train_node_list = []
        for node in train_graph.nodes():
            train_node_list.append(node)

        dis_dic = dict(nx.all_pairs_shortest_path_length(train_graph))
        for node1 in node_set:
            if node1 not in dis_dic.keys():
                dis_dic[node1] = {}
                for node2 in node_set:
                    dis_dic[node1][node2] = N
            else:
                for node2 in node_set:
                    if node2 not in dis_dic[node1].keys():
                        dis_dic[node1][node2] = N

        distance_previous_list = [0 for _ in range(N + 1)]
        for node1 in node_set:
            for node2 in node_set:
                distance_previous_list[dis_dic[node1][node2]] += 1

        n_1 = 0
        n_2 = 0
        store2 = []
        for all_n in range(1000):
            test_edge = random.choice(test_set)
            supplement_e = random.choice(supplement_edge)
            score_test = score_compute_dis(N, distance_previous_list, test_edge, alpha, dis_dic, node_set)
            score_supplement = score_compute_dis(N, distance_previous_list, supplement_e, alpha, dis_dic, node_set)
            if score_test != np.inf:
                store2.append(1 / score_test)
            else:
                store2.append(0)

            if score_test > score_supplement:
                n_1 += 1
            elif score_test == score_supplement:
                n_2 += 1
        auc = (n_1 + 0.5 * n_2) / 1000
        auc_list.append(auc)
        store1.append(sum(store2) / 1000)
    path = 'F:/论文/lp-new/link-prediction/' + graph_name + '/alpha' + str(alpha) + '/dis/probe_size' + str(probe_size)
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)
    f = open(path + '/auc.txt', 'w')
    for value in auc_list:
        f.write(str(value) + '\n')
    f.close()
    f = open(path + '/average_auc.txt', 'w')
    f.write(str(sum(auc_list) / 20))
    f.close()
    f = open(path + '/omega_dis.txt', 'w')
    f.write(str(sum(store1) / 20))
    f.close()


def score_compute_degree(N, degree_previous_list, add_edge, alpha, deg_dict, node_list):
    degree_now_list = [0 for i in range(N)]
    for i in range(len(degree_previous_list)):
        degree_now_list[i] = degree_previous_list[i]

    degree_now_list[deg_dict[add_edge[0]]] -= 1
    degree_now_list[deg_dict[add_edge[0]] + 1] += 1

    degree_s = structural_distribution_consistency(degree_previous_list, degree_now_list, alpha, "deg", N)
    if degree_s == 0:
        score = np.inf
    else:
        score = 1 / degree_s

    return score


def division_set_degree(alpha, probe_size, graph_name):
    G = nx.DiGraph()
    f = open('F:/论文/lp-new/network/' + graph_name + '.txt', 'r')
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
    test_number = int(M * probe_size)
    auc_list = []
    store1 = []
    for z in range(20):
        test_set = random.sample(edge_set, test_number)
        train_set = []
        for edge1 in edge_set:
            if edge1 not in test_set:
                train_set.append(edge1)

        supplement_edge = []
        for node1 in node_set:
            for node2 in node_set:
                if node1 != node2:
                    if (node1, node2) not in edge_set:
                        supplement_edge.append((node1, node2))

        degree_previous_list = [0 for i in range(N)]

        train_graph = nx.DiGraph()
        for edge2 in train_set:
            train_graph.add_edge(edge2[0], edge2[1])
        train_node_list = []

        degree_dic = {}
        for node in train_graph.nodes():
            train_node_list.append(node)
            degree_dic[node] = train_graph.out_degree(node)

        for node in node_set:
            if node not in degree_dic.keys():
                degree_dic[node] = 0

        for key in degree_dic.keys():
            degree_previous_list[degree_dic[key]] += 1

        n_1 = 0
        n_2 = 0
        store2 = []
        for all_n in range(1000):
            test_edge = random.choice(test_set)
            supplement_e = random.choice(supplement_edge)
            score_test = score_compute_degree(N, degree_previous_list, test_edge, alpha, degree_dic, node_set)
            score_supplement = score_compute_degree(N, degree_previous_list, supplement_e, alpha, degree_dic, node_set)
            if score_test != np.inf:
                store2.append(1 / score_test)
            else:
                store2.append(0)
            if score_test > score_supplement:
                n_1 += 1
            elif score_test == score_supplement:
                n_2 += 1
        auc = (n_1 + 0.5 * n_2) / 1000
        auc_list.append(auc)
        store1.append(sum(store2) / 1000)
    path = 'F:/论文/lp-new/link-prediction/' + graph_name + '/alpha' + str(alpha) + '/deg/probe_size' + str(probe_size)
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)
    f = open(path + '/auc.txt', 'w')
    for value in auc_list:
        f.write(str(value) + '\n')
    f.close()
    f = open(path + '/average_auc.txt', 'w')
    f.write(str(sum(auc_list) / 20))
    f.close()
    f = open(path + '/omega_deg.txt', 'w')
    f.write(str(sum(store1) / 20))
    f.close()


# cn
def score_compute_cn(all_set, add_edge):
    # cn_graph = nx.DiGraph()
    # in_degree_dic = {}
    # for edge2 in all_set:
    #     cn_graph.add_edge(edge2[0], edge2[1])
    #     if edge2[1] in in_degree_dic:
    #         in_degree_dic[edge2[1]].append(edge2[0])
    #     else:
    #         in_degree_dic[edge2[1]] = [edge2[0]]
    # cn_graph.add_edge(add_edge[0], add_edge[1])
    # if add_edge[1] in in_degree_dic:
    #     in_degree_dic[add_edge[1]].append(add_edge[0])
    # else:
    #     in_degree_dic[add_edge[1]] = [add_edge[0]]
    # if add_edge[0] not in in_degree_dic:
    #     in_degree_dic[add_edge[0]] = []
    #
    # node1_neighbor = []
    # node2_neighbor = []
    # for neighbor in in_degree_dic[add_edge[0]]:
    #     node1_neighbor.append(neighbor)
    # for neighbor in in_degree_dic[add_edge[1]]:
    #     node2_neighbor.append(neighbor)
    # score = 0
    # for neighbor in node1_neighbor:
    #     if neighbor in node2_neighbor:
    #         score += 1

    cn_graph = nx.DiGraph()
    for edge2 in all_set:
        cn_graph.add_edge(edge2[0], edge2[1])
    cn_graph.add_edge(add_edge[0], add_edge[1])

    node1_neighbor = []
    node2_neighbor = []
    for neighbor in cn_graph.neighbors(add_edge[0]):
        node1_neighbor.append(neighbor)
    for neighbor in cn_graph.neighbors(add_edge[1]):
        node2_neighbor.append(neighbor)
    score = 0
    for neighbor in node1_neighbor:
        if neighbor in node2_neighbor:
            score += 1

    # cn_graph = nx.DiGraph()
    # out_degree_dic = {}
    # for edge2 in all_set:
    #     cn_graph.add_edge(edge2[0], edge2[1])
    #     if edge2[0] in out_degree_dic:
    #         out_degree_dic[edge2[0]].append(edge2[1])
    #     else:
    #         out_degree_dic[edge2[0]] = [edge2[1]]
    # cn_graph.add_edge(add_edge[0], add_edge[1])
    # if add_edge[0] in out_degree_dic:
    #     out_degree_dic[add_edge[0]].append(add_edge[1])
    # else:
    #     out_degree_dic[add_edge[0]] = [add_edge[1]]
    # if add_edge[1] not in out_degree_dic:
    #     out_degree_dic[add_edge[1]] = []
    #
    # node1_neighbor = []
    # node2_neighbor = []
    # for neighbor in out_degree_dic[add_edge[0]]:
    #     node1_neighbor.append(neighbor)
    # for neighbor in out_degree_dic[add_edge[1]]:
    #     node2_neighbor.append(neighbor)
    # score = 0
    # for neighbor in node1_neighbor:
    #     if neighbor in node2_neighbor:
    #         score += 1

    return score


def division_set_cn(probe_size, graph_name):
    G = nx.DiGraph()
    f = open('D:/论文/link prediction/distance_summary/distance_summary/6个数据集/' + graph_name + '.txt', 'r')
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
    test_number = int(M * probe_size)
    auc_list = []
    for z in range(20):
        print(z)
        test_set = random.sample(edge_set, test_number)
        train_set = []
        for edge1 in edge_set:
            if edge1 not in test_set:
                train_set.append(edge1)

        supplement_edge = []
        for node1 in node_set:
            for node2 in node_set:
                if node1 != node2:
                    if (node1, node2) not in edge_set:
                        supplement_edge.append((node1, node2))

        n_1 = 0
        n_2 = 0
        for all_n in range(1000):
            test_edge = random.choice(test_set)
            supplement_e = random.choice(supplement_edge)
            score_test = score_compute_cn(train_set, test_edge)
            score_supplement = score_compute_cn(train_set, supplement_e)
            if score_test > score_supplement:
                n_1 += 1
            elif score_test == score_supplement:
                n_2 += 1
        auc = (n_1 + 0.5 * n_2) / 1000
        auc_list.append(auc)
    path = 'D:/论文/link prediction/fuxian/' + graph_name + '/cn/probe_size' + str(probe_size)
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)
    f = open(path + '/auc.txt', 'w')
    for value in auc_list:
        f.write(str(value) + '\n')
    f.close()
    f = open(path + '/average_auc.txt', 'w')
    f.write(str(sum(auc_list) / 20))
    f.close()


# pa
def score_compute_pa(all_set, add_edge):
    pa_graph = nx.DiGraph()
    for edge1 in all_set:
        pa_graph.add_edge(edge1[0], edge1[1])
    pa_graph.add_edge(add_edge[0], add_edge[1])
    # node1_neighbor = []
    # node2_neighbor = []
    # for neighbor in pa_graph.neighbors(add_edge[0]):
    #     node1_neighbor.append(neighbor)
    # for neighbor in pa_graph.neighbors(add_edge[1]):
    #     node2_neighbor.append(neighbor)
    # score = len(node1_neighbor) * len(node2_neighbor)
    out_degree1 = pa_graph.degree(add_edge[0])
    out_degree2 = pa_graph.degree(add_edge[1])
    score = out_degree1 * out_degree2
    return score


def division_set_pa(probe_size, graph_name):
    G = nx.DiGraph()
    f = open('D:/论文/link prediction/distance_summary/distance_summary/6个数据集/' + graph_name + '.txt', 'r')
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
    test_number = int(M * probe_size)
    auc_list = []
    for z in range(20):
        print(z)
        test_set = random.sample(edge_set, test_number)
        train_set = []
        for edge1 in edge_set:
            if edge1 not in test_set:
                train_set.append(edge1)

        supplement_edge = []
        for node1 in node_set:
            for node2 in node_set:
                if node1 != node2:
                    if (node1, node2) not in edge_set:
                        supplement_edge.append((node1, node2))

        n_1 = 0
        n_2 = 0
        for all_n in range(1000):
            test_edge = random.choice(test_set)
            supplement_e = random.choice(supplement_edge)
            score_test = score_compute_pa(train_set, test_edge)
            score_supplement = score_compute_pa(train_set, supplement_e)
            if score_test > score_supplement:
                n_1 += 1
            elif score_test == score_supplement:
                n_2 += 1
        auc = (n_1 + 0.5 * n_2) / 1000
        auc_list.append(auc)
    path = 'D:/论文/link prediction/fuxian/' + graph_name + '/pa/probe_size' + str(probe_size)
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)
    f = open(path + '/auc.txt', 'w')
    for value in auc_list:
        f.write(str(value) + '\n')
    f.close()
    f = open(path + '/average_auc.txt', 'w')
    f.write(str(sum(auc_list) / 20))
    f.close()


# salton
def score_compute_salton(all_set, add_edge):
    # 按无向图考虑
    salton_graph = nx.DiGraph()
    for edge2 in all_set:
        salton_graph.add_edge(edge2[0], edge2[1])
    salton_graph.add_edge(add_edge[0], add_edge[1])

    node1_neighbor = []
    node2_neighbor = []
    for neighbor in salton_graph.neighbors(add_edge[0]):
        node1_neighbor.append(neighbor)
    for neighbor in salton_graph.neighbors(add_edge[1]):
        node2_neighbor.append(neighbor)

    number_cn = 0
    for neighbor in node1_neighbor:
        if neighbor in node2_neighbor:
            number_cn += 1
    if number_cn == 0:
        score = 0
    else:
        score = number_cn / (len(node1_neighbor) * len(node2_neighbor)) ** (1 / 2)
    # salton_graph = nx.DiGraph()
    # out_degree_dic = {}
    # for edge2 in all_set:
    #     salton_graph.add_edge(edge2[0], edge2[1])
    #     if edge2[0] in out_degree_dic:
    #         out_degree_dic[edge2[0]].append(edge2[1])
    #     else:
    #         out_degree_dic[edge2[0]] = [edge2[1]]
    # salton_graph.add_edge(add_edge[0], add_edge[1])
    # if add_edge[0] in out_degree_dic:
    #     out_degree_dic[add_edge[0]].append(add_edge[1])
    # else:
    #     out_degree_dic[add_edge[0]] = [add_edge[1]]
    # if add_edge[1] not in out_degree_dic:
    #     out_degree_dic[add_edge[1]] = []
    #
    # node1_neighbor = []
    # node2_neighbor = []
    # for neighbor in out_degree_dic[add_edge[0]]:
    #     node1_neighbor.append(neighbor)
    # for neighbor in out_degree_dic[add_edge[0]]:
    #     node2_neighbor.append(neighbor)
    #
    # number_cn = 0
    # for neighbor in node1_neighbor:
    #     if neighbor in node2_neighbor:
    #         number_cn += 1
    # if number_cn == 0:
    #     score = 0
    # else:
    #     score = number_cn / (len(node1_neighbor) * len(node2_neighbor))**(1/2)

    return score


def division_set_salton(probe_size, graph_name):
    G = nx.DiGraph()
    f = open('D:/论文/link prediction/distance_summary/distance_summary/6个数据集/' + graph_name + '.txt', 'r')
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
    test_number = int(M * probe_size)
    auc_list = []
    for z in range(20):
        print(z)
        test_set = random.sample(edge_set, test_number)
        train_set = []
        for edge1 in edge_set:
            if edge1 not in test_set:
                train_set.append(edge1)

        supplement_edge = []
        for node1 in node_set:
            for node2 in node_set:
                if node1 != node2:
                    if (node1, node2) not in edge_set:
                        supplement_edge.append((node1, node2))

        n_1 = 0
        n_2 = 0
        for all_n in range(1000):
            test_edge = random.choice(test_set)
            supplement_e = random.choice(supplement_edge)
            score_test = score_compute_salton(train_set, test_edge)
            score_supplement = score_compute_salton(train_set, supplement_e)
            if score_test > score_supplement:
                n_1 += 1
            elif score_test == score_supplement:
                n_2 += 1
        auc = (n_1 + 0.5 * n_2) / 1000
        auc_list.append(auc)
    path = 'D:/论文/link prediction/fuxian/' + graph_name + '/salton/probe_size' + str(probe_size)
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)
    f = open(path + '/auc.txt', 'w')
    for value in auc_list:
        f.write(str(value) + '\n')
    f.close()
    f = open(path + '/average_auc.txt', 'w')
    f.write(str(sum(auc_list) / 20))
    f.close()


# Jaccard
def score_compute_jaccard(all_set, add_edge):
    jaccard_graph = nx.DiGraph()
    for edge2 in all_set:
        jaccard_graph.add_edge(edge2[0], edge2[1])
    jaccard_graph.add_edge(add_edge[0], add_edge[1])

    node1_neighbor = []
    node2_neighbor = []
    for neighbor in jaccard_graph.neighbors(add_edge[0]):
        node1_neighbor.append(neighbor)
    for neighbor in jaccard_graph.neighbors(add_edge[1]):
        node2_neighbor.append(neighbor)

    number_cn = 0
    for neighbor in node1_neighbor:
        if neighbor in node2_neighbor:
            number_cn += 1
    for neighbor1 in node1_neighbor:
        if neighbor1 not in node2_neighbor:
            node2_neighbor.append(neighbor1)
    number_an = len(node2_neighbor)
    if number_cn == 0:
        score = 0
    else:
        score = number_cn / number_an
    return score


def division_set_jaccard(probe_size, graph_name):
    G = nx.DiGraph()
    f = open('D:/论文/link prediction/distance_summary/distance_summary/6个数据集/' + graph_name + '.txt', 'r')
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
    test_number = int(M * probe_size)
    auc_list = []
    for z in range(20):
        print(z)
        test_set = random.sample(edge_set, test_number)
        train_set = []
        for edge1 in edge_set:
            if edge1 not in test_set:
                train_set.append(edge1)

        supplement_edge = []
        for node1 in node_set:
            for node2 in node_set:
                if node1 != node2:
                    if (node1, node2) not in edge_set:
                        supplement_edge.append((node1, node2))

        n_1 = 0
        n_2 = 0
        for all_n in range(1000):
            test_edge = random.choice(test_set)
            supplement_e = random.choice(supplement_edge)
            score_test = score_compute_jaccard(train_set, test_edge)
            score_supplement = score_compute_jaccard(train_set, supplement_e)
            if score_test > score_supplement:
                n_1 += 1
            elif score_test == score_supplement:
                n_2 += 1
        auc = (n_1 + 0.5 * n_2) / 1000
        auc_list.append(auc)
    path = 'D:/论文/link prediction/fuxian/' + graph_name + '/jaccard/probe_size' + str(probe_size)
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)
    f = open(path + '/auc.txt', 'w')
    for value in auc_list:
        f.write(str(value) + '\n')
    f.close()
    f = open(path + '/average_auc.txt', 'w')
    f.write(str(sum(auc_list) / 20))
    f.close()


# sorenson
def score_compute_sorenson(all_set, add_edge):
    sorenson_graph = nx.DiGraph()
    for edge2 in all_set:
        sorenson_graph.add_edge(edge2[0], edge2[1])
    sorenson_graph.add_edge(add_edge[0], add_edge[1])

    node1_neighbor = []
    node2_neighbor = []
    for neighbor in sorenson_graph.neighbors(add_edge[0]):
        node1_neighbor.append(neighbor)
    for neighbor in sorenson_graph.neighbors(add_edge[1]):
        node2_neighbor.append(neighbor)

    number_cn = 0
    for neighbor in node1_neighbor:
        if neighbor in node2_neighbor:
            number_cn += 1
    if number_cn == 0:
        score = 0
    else:
        score = 2 * number_cn / (len(node1_neighbor) + len(node2_neighbor))

    return score


def division_set_sorenson(probe_size, graph_name):
    G = nx.DiGraph()
    f = open('D:/论文/link prediction/distance_summary/distance_summary/6个数据集/' + graph_name + '.txt', 'r')
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
    test_number = int(M * probe_size)
    auc_list = []
    for z in range(20):
        print(z)
        test_set = random.sample(edge_set, test_number)
        train_set = []
        for edge1 in edge_set:
            if edge1 not in test_set:
                train_set.append(edge1)

        supplement_edge = []
        for node1 in node_set:
            for node2 in node_set:
                if node1 != node2:
                    if (node1, node2) not in edge_set:
                        supplement_edge.append((node1, node2))

        n_1 = 0
        n_2 = 0
        for all_n in range(1000):
            test_edge = random.choice(test_set)
            supplement_e = random.choice(supplement_edge)
            score_test = score_compute_sorenson(train_set, test_edge)
            score_supplement = score_compute_sorenson(train_set, supplement_e)
            if score_test > score_supplement:
                n_1 += 1
            elif score_test == score_supplement:
                n_2 += 1
        auc = (n_1 + 0.5 * n_2) / 1000
        auc_list.append(auc)
    path = 'D:/论文/link prediction/fuxian/' + graph_name + '/sorenson/probe_size' + str(probe_size)
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)
    f = open(path + '/auc.txt', 'w')
    for value in auc_list:
        f.write(str(value) + '\n')
    f.close()
    f = open(path + '/average_auc.txt', 'w')
    f.write(str(sum(auc_list) / 20))
    f.close()


# LR
def compute_lr(all_set):
    compute_graph = nx.DiGraph()
    for edge1 in all_set:
        compute_graph.add_edge(edge1[0], edge1[1])
    N = nx.number_of_nodes(compute_graph)
    compute_graph.add_node('ground_node')
    s_new = {}
    s_old = {}

    # 统计向每个节点输入“值”的邻居
    in_neighbor_dic = {}
    for node in compute_graph.nodes():
        in_neighbor_dic[node] = []
        s_new[node] = 0
        if node != 'ground_node':
            compute_graph.add_edge('ground_node', node)
            compute_graph.add_edge(node, 'ground_node')
            s_old[node] = 1
        else:
            s_old[node] = 0
    for edge in compute_graph.edges():
        in_neighbor_dic[edge[1]].append(edge[0])
    out_degree_dic = {}
    for node in compute_graph.nodes():
        out_degree_dic[node] = compute_graph.out_degree(node)
    error = 1
    while error > 0.00000001:
        error_list = []
        for node in s_old.keys():
            neighbor_value = []
            for in_neighbor in in_neighbor_dic[node]:
                neighbor_value.append(s_old[in_neighbor] / out_degree_dic[in_neighbor])
            s_new[node] = sum(neighbor_value)
        for node in s_old.keys():
            error_list.append(abs(s_new[node] - s_old[node]))
        for node in s_old.keys():
            s_old[node] = s_new[node]
        error = sum(error_list)
    final_s = {}
    for node in s_new.keys():
        if node != 'ground_node':
            final_s[node] = s_new[node] + (s_new['ground_node'] / N)

    return final_s


def score_compute_lr(edge_set, add_edge):
    store_set = []
    for edge1 in edge_set:
        store_set.append(edge1)
    score_old = compute_lr(store_set)
    store_set.append(add_edge)
    score_new = compute_lr(store_set)
    score_list = []
    for key in score_new.keys():
        if key in score_old:
            score_list.append(abs(score_new[key] - score_old[key]))
        else:
            score_list.append(abs(score_new[key]))
    score = 1 / sum(score_list)
    return score


def division_set_lr(probe_size, graph_name):
    G = nx.DiGraph()
    f = open('D:/论文/link prediction/distance_summary/distance_summary/6个数据集/' + graph_name + '.txt', 'r')
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
    test_number = int(M * probe_size)
    auc_list = []

    for z in range(20):
        print(z)
        test_set = random.sample(edge_set, test_number)
        train_set = []
        for edge1 in edge_set:
            if edge1 not in test_set:
                train_set.append(edge1)

        supplement_edge = []
        for node1 in node_set:
            for node2 in node_set:
                if node1 != node2:
                    if (node1, node2) not in edge_set:
                        supplement_edge.append((node1, node2))

        n_1 = 0
        n_2 = 0
        for all_n in range(1000):
            test_edge = random.choice(test_set)
            supplement_e = random.choice(supplement_edge)
            score_test = score_compute_lr(train_set, test_edge)
            score_supplement = score_compute_lr(train_set, supplement_e)
            if score_test > score_supplement:
                n_1 += 1
            elif score_test == score_supplement:
                n_2 += 1
        auc = (n_1 + 0.5 * n_2) / 1000
        auc_list.append(auc)
    path = 'D:/论文/link prediction/fuxian/' + graph_name + '/lr/probe_size' + str(probe_size)
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)
    f = open(path + '/auc.txt', 'w')
    for value in auc_list:
        f.write(str(value) + '\n')
    f.close()
    f = open(path + '/average_auc.txt', 'w')
    f.write(str(sum(auc_list) / 20))
    f.close()


# RWR
def compute_rwr(all_set, rwr):
    compute_graph = nx.DiGraph()
    for edge1 in all_set:
        compute_graph.add_edge(edge1[0], edge1[1])
    rewrite_number = {}
    count = 0
    for node in compute_graph.nodes():
        rewrite_number[node] = count
        count += 1
    N = nx.number_of_nodes(compute_graph)
    A = np.mat(np.zeros((N, N)))
    for edge1 in all_set:
        A[rewrite_number[edge1[0]], rewrite_number[edge1[1]]] = 1
    D = np.mat(np.zeros((N, N)))
    for i in range(N):
        if A.sum(axis=1)[i] != 0:
            D[i, i] = A.sum(axis=1)[i]
        else:
            D[i, i] = 0.0000001
    P = np.linalg.inv(D) * A
    E = np.mat(np.eye(N, N))

    p_0 = np.mat([0 for i in range(N)]).T
    p_0[rewrite_number['0'], 0] = 1
    pi_s = (1 - rwr) * (E - rwr * P.T).I * p_0
    final_s = {}
    for node in compute_graph.nodes():
        final_s[node] = pi_s[rewrite_number[node], 0]
    return final_s


def score_compute_rwr(edge_set, add_edge, rwr):
    store_set = []
    for edge1 in edge_set:
        store_set.append(edge1)
    score_old = compute_rwr(store_set, rwr)
    store_set.append(add_edge)
    score_new = compute_rwr(store_set, rwr)
    score_list = []
    for key in score_new.keys():
        if key in score_old:
            score_list.append(abs(score_new[key] - score_old[key]))
        else:
            score_list.append(abs(score_new[key]))
    score = 1 / sum(score_list)
    return score
    # rwr_graph = nx.DiGraph()
    # for edge1 in edge_set:
    #     rwr_graph.add_edge(edge1[0], edge1[1])
    # rwr_graph.add_edge(add_edge[0], add_edge[1])
    # rewrite_number = {}
    # count = 0
    # for node in rwr_graph.nodes():
    #     rewrite_number[node] = count
    #     count += 1
    # N = nx.number_of_nodes(rwr_graph)
    # A = np.mat(np.zeros((N, N)))
    # for edge1 in edge_set:
    #     A[rewrite_number[edge1[0]], rewrite_number[edge1[1]]] = 1
    # D = np.mat(np.zeros((N, N)))
    # for i in range(N):
    #     if A.sum(axis=1)[i] != 0:
    #         D[i, i] = A.sum(axis=1)[i]
    #     else:
    #         D[i, i] = 0.0000001
    # P = np.linalg.inv(D) * A
    # E = np.mat(np.eye(N, N))
    #
    # p_0 = np.mat([0 for i in range(N)]).T
    # p_0[rewrite_number[add_edge[0]], 0] = 1
    # pi_s = (1 - rwr) * (E - rwr * P.T).I * p_0
    # score = pi_s[rewrite_number[add_edge[1]], 0]
    # return score


def division_set_rwr(probe_size, rwr, graph_name):
    G = nx.DiGraph()
    f = open('D:/论文/link prediction/distance_summary/distance_summary/6个数据集/' + graph_name + '.txt', 'r')
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
    test_number = int(M * probe_size)
    auc_list = []
    for z in range(20):
        print(z)
        test_set = random.sample(edge_set, test_number)
        train_set = []
        train_graph = nx.DiGraph()
        for edge1 in edge_set:
            if edge1 not in test_set:
                train_set.append(edge1)
                train_graph.add_edge(edge1[0], edge1[1])

        supplement_edge = []
        for node1 in node_set:
            for node2 in node_set:
                if node1 != node2:
                    if (node1, node2) not in edge_set:
                        supplement_edge.append((node1, node2))

        n_1 = 0
        n_2 = 0
        for all_n in range(1000):
            test_edge = random.choice(test_set)
            supplement_e = random.choice(supplement_edge)
            score_test = score_compute_rwr(train_set, test_edge, rwr)
            score_supplement = score_compute_rwr(train_set, supplement_e, rwr)
            if score_test > score_supplement:
                n_1 += 1
            elif score_test == score_supplement:
                n_2 += 1
        auc = (n_1 + 0.5 * n_2) / 1000
        auc_list.append(auc)
    path = 'D:/论文/link prediction/fuxian/' + graph_name + '/rwr_ver2/probe_size' + str(probe_size)
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)
    f = open(path + '/auc.txt', 'w')
    for value in auc_list:
        f.write(str(value) + '\n')
    f.close()
    f = open(path + '/average_auc.txt', 'w')
    f.write(str(sum(auc_list) / 20))
    f.close()


if __name__ == '__main__':
    graph_list = ['sampled Delicious']
    # alpha_list = [-8, -6, -4, -2, 0, 2, 4, 6, 8]
    alpha_list = [0]
    size_probe_set = [0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45]
    for graph_type in graph_list:
        for alpha in alpha_list:
            print(graph_type, alpha)
            for sps in size_probe_set:
                # division_set_rwr(sps, 0.1, graph_type)
                # division_set_lr(sps, graph_type)
                # division_set_sorenson(sps, graph_type)
                # division_set_salton(sps, graph_type)
                # division_set_jaccard(sps, graph_type)
                # division_set_pa(sps, graph_type)
                # division_set_cn(sps, graph_type)
                division_set_degree(alpha, sps, graph_type)
                division_set_dis(alpha, sps, graph_type)
    # alpha_list = [-0.25, -0.5, -0.75, -1, 0.25, 0.5, 0.75, 1]  0.81525
    # for alp in alpha_list:
    #     print(alp)
    #     d = [0 for i in range(1000 - 2)]
    #     d_s = [0 for i in range(1000 - 2)]
    #     for j in range(50):
    #         print(j)
    #         G, do, dis = scale_free_network(1000, 1, 2, alp)
    #         for i in range(len(do)):
    #             # d[i] += do[i]
    #             d_s[i] += dis[i]
    #     # y = []
    #     # for dd in d:
    #     #     y.append(dd / 50)
    #     y_dis = []
    #     for ss in d_s:
    #         y_dis.append(ss / 50)
    #     path = 'F:/desktop/link prediction/data/k-2/alpha' + str(alp)
    #     folder = os.path.exists(path)
    #     if not folder:
    #         os.makedirs(path)
    #     # f = open(path + '/degree_omega.txt', 'w')
    #     # for yy in y:
    #     #     f.write(str(yy) + '\n')
    #     # f.close()
    #     f = open(path + '/distance_omega.txt', 'w')
    #     for yy in y:
    #         f.write(str(yy) + '\n')
    #     f.close()
    # #     y2 = []
    # #     for vv in y:
    # #         if vv != 0:
    # #             y2.append(vv)
    # #
    # #     x2 = [i for i in range(len(y2))]
    # #     plt.plot(x2, y2)
    # #     plt.xscale('log')
    # #     plt.yscale('log')
    # # plt.show()
