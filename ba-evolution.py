# -*- coding:utf-8 -*-
"""
created by zazo
2024.1.7
"""
import networkx as nx
import random
import numpy as np
from matplotlib import pyplot as plt
import os
from scipy.optimize import curve_fit
import math


def scale_free_network(N, m, alpha):
    degree_average = [0 for i in range(N - 10 + 1)]
    distance_average = [0 for i in range(N - 10 + 1)]

    for i in range(50):
        node_number = 10
        node_list = []
        F = False
        while not F:
            ba_graph = nx.random_regular_graph(m * 2, 10)
            F = nx.is_connected(ba_graph)

        for node in ba_graph.nodes():
            node_list.append(node)

        degree_omega = []
        distance_omega = []

        # 计算初始度一致性
        degree_previous_list = [0 for i in range(N)]
        degree_now_list = [0 for i in range(N)]

        for node in node_list:
            degree_previous_list[ba_graph.degree(node)] += 1

        # 计算初始距离一致性
        distance_previous_list = [0 for i in range(N)]
        distance_now_list = [0 for i in range(N)]

        dis_dic = dict(nx.all_pairs_shortest_path_length(ba_graph))
        for node1 in node_list:
            for node2 in node_list:
                distance_previous_list[dis_dic[node1][node2]] += 1

        # 给网络加点
        while node_number < N:
            M = nx.number_of_edges(ba_graph)
            sum_degree = 2 * M

            add_length = 0
            add_edge_list = []
            add_edge_node = []

            # 构建ba网络
            while add_length < m:
                for node in node_list:
                    if (node, node_number) not in add_edge_list and (node_number, node) not in add_edge_list:
                        r = random.random()
                        if r < ba_graph.degree(node) / sum_degree:
                            add_edge_list.append((node, node_number))
                            add_edge_node.append(node)
                            add_length += 1
                            if add_length >= m:
                                break
            node_list.append(node_number)

            for add_edge in add_edge_list:
                ba_graph.add_edge(add_edge[0], add_edge[1])

            # 计算度一致性
            for node in node_list:
                degree_now_list[ba_graph.degree(node)] += 1

            degree_s = structural_distribution_consistency(degree_previous_list, degree_now_list, alpha, "deg", len(node_list))
            degree_omega.append(degree_s)

            # 计算距离一致性
            dis_dic = dict(nx.all_pairs_shortest_path_length(ba_graph))
            for node1 in node_list:
                for node2 in dis_dic[node1].keys():
                    distance_now_list[dis_dic[node1][node2]] += 1

            distance_s = structural_distribution_consistency(distance_previous_list, distance_now_list, alpha, "dis", len(node_list))
            distance_omega.append(distance_s)

            for item in range(N):
                degree_previous_list[item] = degree_now_list[item]
                distance_previous_list[item] = distance_now_list[item]

            node_number += 1

            degree_now_list = [0 for i in range(N)]
            distance_now_list = [0 for i in range(N)]

        for i1 in range(len(degree_omega)):
            degree_average[i1] += degree_omega[i1]
            distance_average[i1] += distance_omega[i1]

        path = 'F:/论文/lp-new/BA_new_10000/average_degree' + str(2 * m) + '/' + str(alpha) + '/number' + str(
            i)
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
    for i in range(N - 10 + 1):
        degree_average[i] = degree_average[i] / 50
        distance_average[i] = distance_average[i] / 50

    f = open('F:/论文/lp-new/BA_new_10000/average_degree' + str(2 * m) + '/' + str(alpha) + '/dis_avg.txt',
             'w')
    for value in distance_average:
        f.write(str(value) + '\n')
    f.close()
    f = open('F:/论文/lp-new/BA_new_10000/average_degree' + str(2 * m) + '/' + str(alpha) + '/deg_avg.txt',
             'w')
    for value in degree_average:
        f.write(str(value) + '\n')
    f.close()


def structural_distribution_consistency(feature_previous, feature_now, alpha, type, num_node):
    feature_dif = []
    add_index = 0
    temp_now = [0 for _ in range(len(feature_now))]
    temp_pre = [0 for _ in range(len(feature_previous))]
    size_now = len(feature_now)
    size_pre = len(feature_previous)

    for i in range(size_now):
        if type == "deg":
            temp_now[i] = feature_now[i] / num_node
        elif type == "dis":
            temp_now[i] = feature_now[i] / (num_node ** 2)

    for i in range(size_pre):
        if type == "deg":
            temp_pre[i] = feature_previous[i] / (num_node - 1)
        elif type == "dis":
            temp_pre[i] = feature_previous[i] / ((num_node -1) ** 2)

    for i in range(size_now):
        # if feature_previous[i] - feature_now[i] != 0:
        #     feature_dif.append(exp(-alpha * i) * (feature_previous[i] - feature_now[i]))
        try:
            temp_pre[i]
        except IndexError as e:  # 未捕获到异常，程序直接报错
            add_index = 1

        if add_index == 0:
            if temp_pre[i] - temp_now[i] != 0 and i != 0:
                feature_dif.append((i**(-alpha)) * (temp_pre[i] - temp_now[i]))
        else:
            if temp_pre[i] - temp_now[i] != 0 and i != 0:
                feature_dif.append((i**(-alpha)) * (0 - temp_now[i]))

    feature_structural = abs(sum(feature_dif))
    return feature_structural


def fit_powerlaw(x, a, b):
    return a*x**b


def scale_free_network_degree(N, m):
    degree_max_list = [0 for i in range(N - 10 + 1)]
    degree_dis = [0 for i in range(N)]

    for i in range(50):
        node_number = 10
        node_list = []
        F = False
        while not F:
            ba_graph = nx.random_regular_graph(m * 2, 20)
            F = nx.is_connected(ba_graph)

        for node in ba_graph.nodes():
            node_list.append(node)

        degree_omega = []

        # 计算初始度一致性
        degree_list = [0 for i in range(N)]
        max_degree_list = []

        max_d = 0
        for node in ba_graph.nodes():
            max_d = max(max_d, ba_graph.degree(node))
        max_degree_list.append(max_d)

        # 给网络加点
        while node_number < N:
            M = nx.number_of_edges(ba_graph)
            sum_degree = 2 * M

            add_length = 0
            add_edge_list = []
            add_edge_node = []

            # 构建ba网络
            while add_length < m:
                for node in node_list:
                    if (node, node_number) not in add_edge_list and (node_number, node) not in add_edge_list:
                        r = random.random()
                        if r < ba_graph.degree(node) / sum_degree:
                            add_edge_list.append((node, node_number))
                            add_edge_node.append(node)
                            add_length += 1
                            if add_length >= m:
                                break
            node_list.append(node_number)

            for add_edge in add_edge_list:
                ba_graph.add_edge(add_edge[0], add_edge[1])

            # 计算度一致性
            max_d = 0
            for node in ba_graph.nodes():
                max_d = max(max_d, ba_graph.degree(node))
            max_degree_list.append(max_d)

            node_number += 1

        for node in ba_graph.nodes():
            degree_list[ba_graph.degree(node)] += 1
        for i1 in range(len(degree_list)):
            degree_list[i1] = degree_list[i1] / 1000

        for i1 in range(len(max_degree_list)):
            degree_max_list[i1] += max_degree_list[i1]
        for i1 in range(N):
            degree_dis[i1] += degree_list[i1]

    # 计算50次平均结果
    for i in range(len(degree_max_list)):
        degree_max_list[i] = degree_max_list[i] / 50

    for i in range(len(degree_dis)):
        degree_dis[i] = degree_dis[i] / 50

    f = open('F:/论文/lp-new/BA_new/average_degree' + str(2 * m) + '/deg_max.txt',
             'w')
    for value in degree_max_list:
        f.write(str(value) + '\n')
    f.close()

    f = open('F:/论文/lp-new/BA_new/average_degree' + str(2 * m) + '/deg_distribution.txt',
             'w')
    for value in degree_dis:
        f.write(str(value) + '\n')
    f.close()


def power_func(x, a, b):
    return a * np.power(x, b)

def fit_fun(m):
    degree_max_list = []
    degree_dis = []

    path = 'F:/论文/lp-new/BA_new/average_degree' + str(2 * m) + '/deg_distribution.txt'
    f = open(path, 'r')
    lines = f.readlines()
    f.close()

    for line in lines:
        item = line.strip('\n')
        degree_dis.append(float(item))

    path = 'F:/论文/lp-new/BA_new/average_degree' + str(2 * m) + '/deg_max.txt'
    f = open(path, 'r')
    lines = f.readlines()
    f.close()

    for line in lines:
        item = line.strip('\n')
        degree_max_list.append(float(item))

    x = [i for i in range(1, len(degree_max_list)+1)]
    popt1, pocv1 = curve_fit(fit_powerlaw, x, degree_max_list)
    # print(popt1)

    x2 = [i for i in range(m, len(degree_dis))]
    popt2, pocv2 = curve_fit(fit_powerlaw, x2, degree_dis[m:])
    print(popt2)

    y_fit = power_func(x2, *popt2)
    res = degree_dis[m:] - y_fit
    ss_res = np.sum(res ** 2)
    ss_tot = np.sum((degree_dis[m:] - np.mean(degree_dis[m:])) ** 2)
    r = 1 - (ss_res / ss_tot)
    print(r)


def scale_free_network_dc_10000(N, m, alpha):
    degree_average = [0 for i in range(N - 10 + 1)]
    distance_average = [0 for i in range(N - 10 + 1)]

    for i in range(50):
        node_number = 10
        node_list = []
        F = False
        while not F:
            ba_graph = nx.random_regular_graph(m * 2, 20)
            F = nx.is_connected(ba_graph)

        for node in ba_graph.nodes():
            node_list.append(node)

        degree_omega = []
        distance_omega = []

        # 计算初始度一致性
        degree_previous_list = [0 for i in range(N)]
        degree_now_list = [0 for i in range(N)]

        for node in node_list:
            degree_previous_list[ba_graph.degree(node)] += 1

        # 计算初始距离一致性
        distance_previous_list = [0 for i in range(N)]
        distance_now_list = [0 for i in range(N)]

        dis_dic = dict(nx.all_pairs_shortest_path_length(ba_graph))
        for node1 in node_list:
            for node2 in node_list:
                distance_previous_list[dis_dic[node1][node2]] += 1

        # 给网络加点
        while node_number < N:
            M = nx.number_of_edges(ba_graph)
            sum_degree = 2 * M

            add_length = 0
            add_edge_list = []
            add_edge_node = []

            # 构建ba网络
            while add_length < m:
                for node in node_list:
                    if (node, node_number) not in add_edge_list and (node_number, node) not in add_edge_list:
                        r = random.random()
                        if r < ba_graph.degree(node) / sum_degree:
                            add_edge_list.append((node, node_number))
                            add_edge_node.append(node)
                            add_length += 1
                            if add_length >= m:
                                break
            node_list.append(node_number)

            for add_edge in add_edge_list:
                ba_graph.add_edge(add_edge[0], add_edge[1])

            # 计算度一致性
            for node in node_list:
                degree_now_list[ba_graph.degree(node)] += 1

            degree_s = structural_distribution_consistency(degree_previous_list, degree_now_list, alpha, "deg", len(node_list))
            degree_omega.append(degree_s)

            # 计算距离一致性
            dis_dic = dict(nx.all_pairs_shortest_path_length(ba_graph))
            for node1 in node_list:
                for node2 in dis_dic[node1].keys():
                    distance_now_list[dis_dic[node1][node2]] += 1

            distance_s = structural_distribution_consistency(distance_previous_list, distance_now_list, alpha, "dis", len(node_list))
            distance_omega.append(distance_s)

            for item in range(N):
                degree_previous_list[item] = degree_now_list[item]
                distance_previous_list[item] = distance_now_list[item]

            node_number += 1

            degree_now_list = [0 for i in range(N)]
            distance_now_list = [0 for i in range(N)]

        for i1 in range(len(degree_omega)):
            degree_average[i1] += degree_omega[i1]
            distance_average[i1] += distance_omega[i1]

        path = 'F:/论文/lp-new/BA_new_10000/average_degree' + str(2 * m) + '/' + str(alpha) + '/number' + str(
            i)
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
    for i in range(N - 10 + 1):
        degree_average[i] = degree_average[i] / 50
        distance_average[i] = distance_average[i] / 50

    f = open('F:/论文/lp-new/BA_new_10000/average_degree' + str(2 * m) + '/' + str(alpha) + '/dis_avg.txt',
             'w')
    for value in distance_average:
        f.write(str(value) + '\n')
    f.close()
    f = open('F:/论文/lp-new/BA_new_10000/average_degree' + str(2 * m) + '/' + str(alpha) + '/deg_avg.txt',
             'w')
    for value in degree_average:
        f.write(str(value) + '\n')
    f.close()


if __name__ == '__main__':
    # m_list = [1, 2, 3, 4]
    # for i in range(len(m_list)):
    #     fit_fun(m_list[i])
        # scale_free_network_degree(1000, m_list[i])
    # m_list = [2]
    # alpha_list = [0, 2, 4, 6, 8]
    # for i in range(len(m_list)):
    #     for j in range(len(alpha_list)):
    #         print(alpha_list[j])
    #         scale_free_network_dc_10000(10000, m_list[i], alpha_list[j])