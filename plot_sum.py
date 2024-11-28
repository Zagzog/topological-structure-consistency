# -*- coding:utf-8 -*-
"""
created by zazo
2024.5.5
"""
import os
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import networkx as nx
from matplotlib.colors import LinearSegmentedColormap
from pandas import DataFrame
import random
from scipy import optimize
from scipy.optimize import curve_fit


#随着probsize变化的静态网络链路预测对比结果
def plot_fig7():
    graph_type = ['new FW1', 'new FW2', 'new FW3',  'sampled SmaGri', 'sampled Kohonen', 'sampled Delicious']
    link_type1 = ['deg', 'dis', 'cn', 'pa', 'lr', 'rwr_ver2']
    x = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45]
    lab_list = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)']

    color_list = ['#3d5db7', '#ba3d1c', '#84C2E4', '#BCBDC2', '#DDBBB3', '#E59A5B']
    # color_list = ['#525252', '#F24040', '#3d5db7', '#888A93', '#84C2E4', '#8db2e1']

    point_type = ['s', 'o', '^', 'v', 'D', '<']
    subplot_loc = [(2, 3, 1), (2, 3, 2), (2, 3, 3), (2, 3, 4), (2, 3, 5), (2, 3, 6)]
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    font = {
        'family': 'Times New Roman',
        'weight': 'bold',
        'size': 10,
    }
    font2 = {
        'family': 'Times New Roman',
        'weight': 'bold',
        'size': 30,
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

            plt.plot(x, y, color=color_list[j], marker=point_type[j], linestyle='-', linewidth=3)
            plt.xlim(0.04, 0.46)
            #plt.ylim(0.25, 1)
            if graph_type[i] == 'new FW2':
                plt.ylim(0.45, 1.)
            elif graph_type[i] == 'new FW3':
                plt.ylim(0.45, 1.)
            elif graph_type[i] == 'sampled SmaGri' or graph_type[i] == 'new FW1':
                plt.ylim(0.35, 0.95)
            elif graph_type[i] == 'sampled Delicious':
                plt.ylim(0.25, 1)
            else:
                plt.ylim(0.25, 1.0)
            plt.xticks([0.05, 0.15,  0.25,  0.35,  0.45])
            plt.tick_params(labelsize=20)

            plt.xlabel('p', font2, labelpad=8)
            plt.ylabel('AUC', font2, labelpad=10)
            plt.errorbar(x, y, color=color_list[j], marker=point_type[j], linestyle='-', yerr=yerror, capsize=3)
            # plt.title(lab_list[i], font2, x=-0.17)

    plt.subplots_adjust(wspace=0.3, hspace=0.5, left=0.07, right=0.98, bottom=0.09, top=0.95)
    plt.show()


def plot_fig10():
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    graph_type = ['contacts-prox-high-school-2013', 'contacts-prox-high-school-2013', 'contacts-prox-high-school-2013',
                  'contacts-prox-high-school-2013', 'SFHH-conf-sensor', 'SFHH-conf-sensor', 'SFHH-conf-sensor',
                  'SFHH-conf-sensor']

    link_type1 = ['deg', 'dis']
    link_type2 = ['cn', 'rwr']
    link_type3 = ['pa', 'lr']
    link_type = []

    lab_list = ['(a)', '(a)', '(a)', '(a)', '(b)', '(b)', '(b)', '(b)']

    color_list1 = ['#3d5db7', '#ba3d1c']
    color_list2 = ['#8db2e1', '#E3825D']
    color_list3 = ['#84C2E4', '#E59A5B']
    color_list4 = ['#BCBDC2', '#DDBBB3']
    color_list = []

    point_type1 = ['s', 'o']
    point_type2 = ['^', '<']
    point_type3 = ['v', 'D']
    point_type = []

    subplot_loc = [(2, 4, 1), (2, 4, 2), (2, 4, 3), (2, 4, 4), (2, 4, 5), (2, 4, 6), (2, 4, 7), (2, 4, 8)]

    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    font = {
        'family': 'Times New Roman',
        'weight': 'bold',
        'size': 10,
    }
    font2 = {
        'family': 'Times New Roman',
        'weight': 'bold',
        'size': 30,
    }
    a1 = plt.figure(figsize=(15, 8))
    for i in range(len(graph_type)):
        plt.subplot(subplot_loc[i][0], subplot_loc[i][1], subplot_loc[i][2])
        if i == 0 or i == 4:
            link_type = link_type1
            color_list = color_list1
            point_type = point_type1
        elif i == 1 or i == 5:
            link_type = link_type1
            color_list = color_list2
            point_type = point_type1
        elif i == 2 or i == 6:
            link_type = link_type2
            color_list = color_list3
            point_type = point_type2
        else:
            link_type = link_type3
            color_list = color_list4
            point_type = point_type3

        for j in range(len(link_type)):
            y = []
            ans = 0
            if i == 1 or i == 5:
                path = 'F:/论文/lp-new/dynamic_stablization_link_prediction/results/probe_size/' + graph_type[i] + '/' \
                       + link_type[j]
            else:
                path = 'F:/论文/lp-new/link-prediction-time-vary/results/probe_size/' + graph_type[i] + '/' + \
                       link_type[j]
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
            j_g = int(len(y) / 100)
            y1 = []
            yeer = []
            for y_value in range(100):
                y1.append(sum(y[y_value:y_value+j_g]) / j_g)
                yeer.append(np.std(y[y_value:y_value+j_g]))

            x1 = [_ for _ in range(100)]
            y_par = []
            x_par = []

            for k in range(100):
                y_par.append(y[int(k*(len(y) / 100))])
                x_par.append(x[int(k * (len(y) / 100))])
            plt.plot(x_par, y_par, color=color_list[j], marker=point_type[j], linestyle='-', markersize=3, linewidth=2)
            plt.ylim(-0.05, 1.05)

        if i == 0 or i == 4:
            plt.yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
            # plt.xticks([0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45])
            plt.tick_params(labelsize=20)

            plt.xlabel('t', font2, labelpad=8)
            plt.ylabel('AUC', font2, labelpad=10)
            # plt.errorbar(x, y, color=color_list[j], marker=point_type[j], linestyle='-', yerr=yerror, capsize=3)
            plt.title(lab_list[i], font2, x=-0.17)
        else:
            plt.yticks([])
            plt.tick_params(labelsize=20)
            plt.xlabel('t', font2, labelpad=8)

    plt.subplots_adjust(wspace=0, hspace=0.5, left=0.07, right=0.98, bottom=0.1, top=0.9)
    plt.show()


def plot_fig8and9(type):
    graph_type = ['new FW1', 'new FW2', 'new FW3', 'sampled SmaGri', 'sampled Kohonen', 'sampled Delicious']
    probe_list = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45]
    alpha_list = [8, 6, 4, 2, 0, -2, -4, -6, -8]
    lab_list = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)']
    subplot_loc = [(2, 3, 1), (2, 3, 2), (2, 3, 3), (2, 3, 4), (2, 3, 5), (2, 3, 6)]
    color_list3 = ['#84C2E4', '#E59A5B']
    # color2 = LinearSegmentedColormap.from_list("", ['#4791C5', '#FEE797', '#DE6C5B'])
    color2 = LinearSegmentedColormap.from_list("", ['#4791C5', '#FFFFFF', '#DE6C5B'])
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    font = {
        'family': 'Times New Roman',
        'weight': 'bold',
        'size': 10,
    }
    font2 = {
        'family': 'Times New Roman',
        'weight': 'bold',
        'size': 30,
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
        if type == 'deg' and i < 3:
            h_figure = sns.heatmap(data, robust=True, vmin=0.29, vmax=0.71, cmap=color2, annot_kws=font2, cbar=False)
        else:
            h_figure = sns.heatmap(data, robust=True, vmin=0.15, vmax=0.85, cmap=color2, annot_kws=font2, cbar=False)
        # cb = h_figure.figure.colorbar(h_figure.collections[0])  # 显示colorbar
        # cb.ax.tick_params(labelsize=17)
        # if type == "dis":
        #     sns.heatmap(data, robust=True, vmin=0.2, vmax=0.8, cmap=color2, annot_kws={"front_size": 11})
        # else:
        #     sns.heatmap(data, robust=True, cmap=color2, annot_kws={"front_size": 15})

        plt.xticks([i+0.5 for i in range(9)], [0.05, '', 0.15, '', 0.25, '', 0.35, '', 0.45])
        plt.yticks([i+0.5 for i in range(9)], [8, 6, 4, 2, 0, -2, -4, -6, -8], rotation='horizontal')
        plt.xlabel('p', font2, labelpad=8)
        plt.ylabel(r'$\alpha$', font2, labelpad=10)

        plt.tick_params(labelsize=20)
        # plt.title(lab_list[i], font2, x=-0.17)

    plt.subplots_adjust(wspace=0.3, hspace=0.4, left=0.05, right=0.98, bottom=0.09, top=0.95)
    plt.show()


def plot_fig11and12(type):
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'

    graph_type = ['contacts-prox-high-school-2013', 'SFHH-conf-sensor', 'contacts-prox-high-school-2013', 'SFHH-conf-sensor']
    alpha_list = [8, 6, 4, 2, 0, -2, -4, -6, -8]
    lab_list = ['(a)', '(b)', '(c)', '(d)']
    subplot_loc = [(2, 2, 1), (2, 2, 2), (2, 2, 3), (2, 2, 4)]

    # color2 = LinearSegmentedColormap.from_list("", ['#4791C5', '#FEE797', '#DE6C5B'])
    # color_list = ['#4791C5', '#D7E7F2', '#8EBBDB',  '#FEE7C6', '#F9E4E1', '#EBA69B', '#DE6C5B']
    # color2 = LinearSegmentedColormap.from_list("", color_list)
    color2 = LinearSegmentedColormap.from_list("", ['#4791C5', '#FFFFFF', '#DE6C5B'])

    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    font = {
        'family': 'Times New Roman',
        'weight': 'bold',
        'size': 25,
    }
    font2 = {
        'family': 'Times New Roman',
        'weight': 'bold',
        'size': 30,
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
        h_figure = sns.heatmap(data, robust=True, vmin=0, vmax=1, cmap=color2, annot_kws=font2, cbar=False)
        # cb = h_figure.figure.colorbar(h_figure.collections[0])  # 显示colorbar
        # cb.ax.tick_params(labelsize=17)

        if i == 0 or i == 2:
            plt.xticks(np.arange(0, 7001, 2000), np.arange(0, 7001, 2000), rotation=0)
        if i == 1 or i == 3:
            plt.xticks(np.arange(0, 3001, 1000), np.arange(0, 3001, 1000), rotation=0)
        plt.yticks([i+0.5 for i in range(9)], [8, 6, 4, 2, 0, -2, -4, -6, -8], rotation='horizontal')
        plt.xlabel('t', font2, labelpad=10)
        plt.ylabel(r'$\alpha$', font2, labelpad=10)
        plt.tick_params(labelsize=20)
        # plt.title(lab_list[i], font, x=-0.17)

    plt.subplots_adjust(wspace=0.3, hspace=0.4, left=0.1, right=0.98, bottom=0.1, top=0.95)
    plt.show()


def gini_coef(wealths):
    cum_wealths = np.cumsum(sorted(np.append(wealths, 0)))
    sum_wealths = cum_wealths[-1]
    xarray = np.array(range(0, len(cum_wealths))) / (len(cum_wealths)-1)
    yarray = cum_wealths / sum_wealths
    B = np.trapz(yarray, x=xarray)
    A = 0.5 - B
    return A / (A+B)


def plot_fig13():
    graph_type = ['new FW1', 'new FW2', 'new FW3', 'sampled SmaGri', 'sampled Kohonen', 'sampled Delicious']
    font = {
        'family': 'Times New Roman',
        'weight': 'bold',
        'size': 30,
    }
    font2 = {
        'family': 'Times New Roman',
        'weight': 'bold',
        'size': 30,
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
        d = nx.average_shortest_path_length(G)
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
        plt.plot(x, degree_distribution, color='#ba3d1c', linestyle='-', marker='o', markersize=5, linewidth=3)
        plt.plot(x, distance_distribution, color='#3d5db7', linestyle='-', marker='s', markersize=5, linewidth=3)

        plt.xlim(-5, 60)
        plt.tick_params(labelsize=20)
        plt.xlabel('x', font, labelpad=5)
        plt.ylabel('frequency', font2, labelpad=8)
        plt.title(lab_list[i], font2, x=-0.2)
    plt.subplots_adjust(wspace=0.35, hspace=0.5, left=0.07, right=0.98, bottom=0.1, top=0.95)
    plt.show()


def plot_fig5and6(type):
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    graph_list = ['contacts-prox-high-school-2013', 'SFHH-conf-sensor']
    subplot_loc = [(1, 2, 1), (1, 2, 2)]
    font2 = {
        'family': 'Times New Roman',
        'weight': 'bold',
        'size': 30,
    }
    lab_list = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)', '(g)', '(h)']
    # color_list = ['#D87DE5', '#F9A34E', '#038303', '#7F659D', '#676767', '#3176B8', '#D74B0C', '#42B497']

    color_list = ['#3d5db7', '#8db2e1', '#84C2E4', '#BCBDC2', '#DDBBB3', '#E59A5B', '#E3825D', '#ba3d1c']

    # color_list1 = ['#3d5db7', '#8db2e1', '#daedf7', '#D3D4D7']
    # color_list2 = ['#f5ebe9', '#f7d3ca', '#D37D66', '#ba3d1c']
    # alpha_list1 = [-8, -6, -4, -2]
    # alpha_list2 = [2, 4, 6, 8]

    alpha_list = [-8, -6, -4, -2, 2, 4, 6, 8]
    a1 = plt.figure(figsize=(12, 5))
    for j in range(2):
        avg_list = []
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
            avg_list.append(sum(y[:-1]) / len(y[:-1]))

        if type == 'deg':
            plt.xlabel('t', font2, labelpad=8)
            plt.ylabel(r'$\mathcal{D}_{\alpha}^t$', font2, labelpad=8)
            # plt.title(lab_list[j], font2, x=-0.2)
            plt.tick_params(labelsize=20)
        else:
            plt.xlabel('t', font2, labelpad=8)
            plt.ylabel(r'$\mathcal{L}_{\alpha}^t$', font2, labelpad=8)
            # plt.title(lab_list[j], font2, x=-0.2)
            plt.tick_params(labelsize=20)

        print(graph_list[j], avg_list)
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
    plt.subplots_adjust(wspace=0.4, hspace=0.4, left=0.13, right=0.99, bottom=0.15, top=0.9)
    plt.show()


def plot_fig1(powerlaw_type):
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    subplot_loc = [(1, 2, 1), (1, 2, 2)]
    font2 = {
        'family': 'Times New Roman',
        'weight': 'bold',
        'size': 70,
    }
    lab_list = ['(a)', '(b)', '(c)']
    # powerlaw_type = [2.5, 3.0, 4.0]
    m_a_type = [2, 4, 6]

    color_list1 = ['#3d5db7', '#8db2e1', '#84C2E4', '#BCBDC2']
    color_list2 = ['#DDBBB3', '#E59A5B', '#E3825D', '#ba3d1c']
    alpha_list1 = [-8, -6, -4, -2]
    alpha_list2 = [2, 4, 6, 8]

    for deg1 in range(len(m_a_type)):
        figure, ax = plt.subplots(figsize=(14, 6))
        for j in range(len(subplot_loc)):
            for i in range(4):
                if j < 1:
                    path = 'F:/论文/lp-new/new_ba_data/BA_new_10000/power_law_' + str(powerlaw_type) \
                           + '/m_a_' + str(m_a_type[deg1]) + '/' + str(alpha_list1[i])
                else:
                    path = 'F:/论文/lp-new/new_ba_data/BA_new_10000/power_law_' + str(powerlaw_type) \
                           + '/m_a_' + str(m_a_type[deg1]) + '/' + str(alpha_list2[i])
                f = open(path + '/deg_avg.txt', 'r')
                lines = f.readlines()
                f.close()

                y = []
                for line in lines:
                    item = line.strip('\n')
                    y.append(float(item))

                ax1 = plt.subplot(subplot_loc[j][0], subplot_loc[j][1], subplot_loc[j][2])

                x = [k for k in range(len(y) - 1)]
                if j < 1:
                    plt.plot(x, y[:-1], color=color_list1[i], linewidth=4)
                    plt.yscale('log')
                    plt.xscale('log')
                else:
                    plt.plot(x, y[:-1], color=color_list2[i], linewidth=4)
                    plt.xscale('log')
                    plt.yscale('log')

                # if j < 1:
                #     if powerlaw_type == 3.0:
                #         y_fit = [(m_a_type[deg1]) ** (- alpha_list1[i] - 2) * (k) ** (
                #                     (0.5) * (- alpha_list1[i] - 4)) for k in range(1, len(y))]
                #     else:
                #         y_fit = [(m_a_type[deg1]) ** (2 - alpha_list1[i] - powerlaw_type) * (k) ** (
                #                     (0.5) * (- alpha_list1[i] - powerlaw_type)) for k in range(1, len(y))]
                #     plt.plot(x, y_fit, color=color_list1[i], linestyle='--', linewidth=10)
                # else:
                #     if powerlaw_type == 3.0:
                #         y_fit = [3 * (m_a_type[deg1]) ** (- alpha_list2[i] - 2) * (k) ** (-1) for k in range(1, len(y))]
                #     else:
                #         y_fit = [
                #             abs(powerlaw_type - 3) * (m_a_type[deg1]) ** (2 - alpha_list2[i] - powerlaw_type) * (k) ** (-1) for k in range(1, len(y))]
                #     plt.plot(x, y_fit, color=color_list2[i], linestyle='--', linewidth=10)

            if j < 1:
                plt.xlim(5, 12000)
                plt.xlabel('t', font2, labelpad=5)
                plt.ylabel(r'$\mathcal{D}_{\alpha}^t$', font2, labelpad=5)
                # plt.title(lab_list[deg1], font2, x=-0.15)
                plt.tick_params(labelsize=50, pad=10)
            else:
                plt.xlim(5, 12000)
                plt.xlabel('t', font2, labelpad=5)
                plt.tick_params(labelsize=50, pad=10)
                ax1.yaxis.tick_right()
                ax1.yaxis.set_label_position("right")
                ax1.set_yticks([], minor=True)

        plt.subplots_adjust(wspace=0, hspace=0.2, left=0.17, right=0.9, bottom=0.18, top=0.9)
        plt.show()


def plot_fig2(powerlaw_type):
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    subplot_loc = [(1, 2, 1), (1, 2, 2)]
    font2 = {
        'family': 'Times New Roman',
        'weight': 'bold',
        'size': 70,
    }
    lab_list = ['(a)', '(b)', '(c)']
    # powerlaw_type = [2.5, 3.0, 4.0]
    m_a_type = [2, 4, 6]

    color_list1 = ['#3d5db7', '#8db2e1', '#84C2E4', '#BCBDC2']
    color_list2 = ['#DDBBB3', '#E59A5B', '#E3825D', '#ba3d1c']
    alpha_list1 = [-8, -6, -4, -2]
    alpha_list2 = [2, 4, 6, 8]

    for deg1 in range(len(m_a_type)):
        figure, ax = plt.subplots(figsize=(12, 5))
        for j in range(len(subplot_loc)):
            for i in range(4):
                if j < 1:
                    path = 'F:/论文/lp-new/new_ba_data/BA_new_1000/power_law_' + str(powerlaw_type) \
                           + '/m_a_' + str(m_a_type[deg1]) + '/' + str(alpha_list1[i])
                else:
                    path = 'F:/论文/lp-new/new_ba_data/BA_new_1000/power_law_' + str(powerlaw_type) \
                           + '/m_a_' + str(m_a_type[deg1]) + '/' + str(alpha_list2[i])
                f = open(path + '/dis_avg.txt', 'r')
                lines = f.readlines()
                f.close()

                y = []
                for line in lines:
                    item = line.strip('\n')
                    y.append(float(item))

                ax1 = plt.subplot(subplot_loc[j][0], subplot_loc[j][1], subplot_loc[j][2])

                x = [k for k in range(len(y) - 11)]
                if j < 1:
                    plt.plot(x, y[:-11], color=color_list1[i], linewidth=10)
                    plt.yscale('log')
                    plt.xscale('log')
                else:
                    plt.plot(x, y[:-11], color=color_list2[i], linewidth=10)
                    plt.xscale('log')
                    plt.yscale('log')

                # if j < 1:
                #     if powerlaw_type == 3.0:
                #         y_fit = [(m_a_type[deg1]) ** (- alpha_list1[i] - 2) * (k) ** (
                #                     (0.5) * (- alpha_list1[i] - 4)) for k in range(1, len(y))]
                #     else:
                #         y_fit = [(m_a_type[deg1]) ** (2 - alpha_list1[i] - powerlaw_type) * (k) ** (
                #                     (0.5) * (- alpha_list1[i] - powerlaw_type)) for k in range(1, len(y))]
                #     plt.plot(x, y_fit, color=color_list1[i], linestyle='--', linewidth=2)
                # else:
                #     if powerlaw_type == 3.0:
                #         y_fit = [3 * (m_a_type[deg1]) ** (- alpha_list2[i] - 2) * (k) ** (-1) for k in range(1, len(y))]
                #     else:
                #         y_fit = [
                #             abs(powerlaw_type - 3) * (m_a_type[deg1]) ** (2 - alpha_list2[i] - powerlaw_type) * (k) ** (-1) for k in range(1, len(y))]
                #     plt.plot(x, y_fit, color=color_list2[i], linestyle='--', linewidth=2)

            if j < 1:
                plt.xlim(5, 1500)
                plt.xlabel('t', font2, labelpad=5)
                plt.ylabel(r'$\mathcal{L}_{\alpha}^t$', font2, labelpad=5)
                # plt.title(lab_list[deg1], font2, x=-0.15)
                plt.tick_params(labelsize=50, pad=10)
            else:
                plt.xlim(5, 1500)
                plt.xlabel('t', font2, labelpad=8)
                plt.tick_params(labelsize=50, pad=10)
                ax1.yaxis.tick_right()
                ax1.yaxis.set_label_position("right")
                ax1.set_yticks([], minor=True)

        plt.subplots_adjust(wspace=0, hspace=0.2, left=0.18, right=0.9, bottom=0.18, top=0.9)
        plt.show()


def plot_fig3():
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    subplot_loc = [(1, 2, 1), (1, 2, 2)]
    font2 = {
        'family': 'Times New Roman',
        'weight': 'bold',
        'size': 60,
    }
    lab_list = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)', '(g)', '(h)']
    rep_type = [0.01, 0.02, 0.05, 0.1, 0.25, 0.5, 0.75, 1]

    # color_list = ['#D87DE5', '#F9A34E', '#038303', '#7F659D', '#676767', '#3176B8', '#D74B0C', '#42B497']

    color_list1 = ['#3d5db7', '#8db2e1', '#84C2E4', '#BCBDC2']
    color_list2 = ['#DDBBB3', '#E59A5B', '#E3825D', '#ba3d1c']
    alpha_list1 = [-8, -6, -4, -2]
    alpha_list2 = [2, 4, 6, 8]

    for rep in range(len(rep_type)):
        figure, ax = plt.subplots(figsize=(20, 8))
        for j in range(2):
            ax1 = plt.subplot(subplot_loc[j][0], subplot_loc[j][1], subplot_loc[j][2])
            for i in range(4):
                y = []
                if j < 1:
                    path = 'F:/论文/lp-new/WS_new/re_probability' + str(rep_type[rep]) + '/' + str(alpha_list1[i])
                else:
                    path = 'F:/论文/lp-new/WS_new/re_probability' + str(rep_type[rep]) + '/' + str(alpha_list2[i])
                f = open(path + '/deg_avg.txt', 'r')
                lines = f.readlines()
                f.close()
                for line in lines:
                    item = line.strip('\n')
                    y.append(float(item))
                # print(y)

                y1 = []
                x = []
                len1 = 0
                for item in y:
                    if item != 0.0:
                        y1.append(item)
                        x.append(len1)
                    len1 += 1

                # x = [k for k in range(len(y1))]
                if j < 1:
                    plt.plot(x[:2000], y1[:2000], color=color_list1[i])
                else:
                    plt.plot(x[:2000], y1[:2000], color=color_list2[i])

                plt.yscale('log')
                # plt.xscale('log')
                plt.xlim(-100, 1500)

            if j < 1:
                plt.xlabel('t', font2, labelpad=5)
                plt.ylabel(r'$\mathcal{D}_{\alpha}^t$', font2, labelpad=8)
                # plt.title(lab_list[rep], font2, x=-0.15)
                plt.tick_params(labelsize=40, pad=8)
                plt.xticks([0, 500, 1000])
            else:
                plt.xlabel('t', font2, labelpad=8)
                plt.tick_params(labelsize=40, pad=8)
                ax1.yaxis.tick_right()
                ax1.yaxis.set_label_position("right")
                ax1.set_yticks([], minor=True)
                plt.xticks([0, 500, 1000])

        plt.subplots_adjust(wspace=0, hspace=0.3, left=0.15, right=0.9, bottom=0.18, top=0.9)
        plt.show()


def plot_fig4():
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    subplot_loc = [(1, 2, 1), (1, 2, 2)]
    font2 = {
        'family': 'Times New Roman',
        'weight': 'bold',
        'size': 60,
    }
    lab_list = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)', '(g)', '(h)']
    rep_type = [0.01, 0.02, 0.05, 0.1, 0.25, 0.5, 0.75, 1]

    # color_list = ['#D87DE5', '#F9A34E', '#038303', '#7F659D', '#676767', '#3176B8', '#D74B0C', '#42B497']

    color_list1 = ['#3d5db7', '#8db2e1', '#84C2E4', '#BCBDC2']
    color_list2 = ['#DDBBB3', '#E59A5B', '#E3825D', '#ba3d1c']

    # color_list1 = ['#ab4d41', '#c65a4c', '#f9e1d7', '#f3dc97']

    # color_list1 = ['#D87DE5', '#F9A34E', '#038303', '#7F659D']
    # color_list2 = ['#676767', '#42B497', '#D74B0C', '#3176B8']
    alpha_list1 = [-8, -6, -4, -2]
    alpha_list2 = [8, 6, 4, 2]

    for rep in range(len(rep_type)):
        figure, ax = plt.subplots(figsize=(20, 8))
        for j in range(2):
            ax1 = plt.subplot(subplot_loc[j][0], subplot_loc[j][1], subplot_loc[j][2])
            for i in range(4):
                y = []
                if j < 1:
                    path = 'F:/论文/lp-new/WS_new/re_probability' + str(rep_type[rep]) + '/' + str(alpha_list1[i])
                else:
                    path = 'F:/论文/lp-new/WS_new/re_probability' + str(rep_type[rep]) + '/' + str(alpha_list2[i])
                f = open(path + '/dis_avg.txt', 'r')
                lines = f.readlines()
                f.close()
                for line in lines:
                    item = line.strip('\n')
                    y.append(float(item))
                # print(y)

                y1 = []
                x = []
                len1 = 0
                for item in y:
                    if item != 0.0:
                        y1.append(item)
                        x.append(len1)
                    len1 += 1

                # x = [k for k in range(len(y1))]
                if j < 1:
                    plt.plot(x[:3000], y1[:3000], color=color_list1[i])
                else:
                    plt.plot(x[:3000], y1[:3000], color=color_list2[i])

                plt.yscale('log')
                # plt.xscale('log')
                plt.xlim(-100, 3100)

            if j < 1:
                plt.xlabel('t', font2, labelpad=5)
                plt.ylabel(r'$\mathcal{L}_{\alpha}^t$', font2, labelpad=8)
                # plt.title(lab_list[rep], font2, x=-0.15)
                plt.tick_params(labelsize=40, pad=8)
                plt.xticks([0, 1000, 2000])
            else:
                plt.xlabel('t', font2, labelpad=8)
                plt.tick_params(labelsize=40, pad=8)
                ax1.yaxis.tick_right()
                ax1.yaxis.set_label_position("right")
                ax1.set_yticks([], minor=True)
                plt.xticks([0, 1000, 2000])

        plt.subplots_adjust(wspace=0, hspace=0.3, left=0.15, right=0.9, bottom=0.18, top=0.9)
        plt.show()


def plot_empty_colorbar():
    font = {
        'weight': 'bold',
        'size': 25,
    }
    font2 = {
        'weight': 'normal',
        'size': 20,
    }
    figure, ax = plt.subplots(figsize=(5, 5))
    color = LinearSegmentedColormap.from_list("", ['#4791C5', '#FFFFFF', '#DE6C5B'])
    data = {}
    df = DataFrame(data)
    ax = sns.heatmap(df, center=0.5, cbar=False, vmin=0, vmax=1, cmap=color)
    cb = plt.colorbar(ax.collections[0], orientation='vertical')
    cb.ax.tick_params(labelsize=15, labelrotation=0)
    plt.subplots_adjust(left=0.2, right=0.9)
    cb.outline.set_visible(False)
    cb.set_ticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
    cb.set_ticklabels([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
    #plt.savefig('F:/desktop/正文图/figure2/ver2/colorbar.png')
    plt.show()


def plot_fig10_ver2():
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    graph_type = ['contacts-prox-high-school-2013', 'contacts-prox-high-school-2013', 'contacts-prox-high-school-2013',
                  'SFHH-conf-sensor', 'SFHH-conf-sensor', 'SFHH-conf-sensor']

    link_type1 = ['deg', 'dis']
    link_type2 = ['cn', 'rwr']
    link_type3 = ['pa', 'lr']
    link_type = []

    lab_list = ['(a)', '(a)', '(a)', '(a)', '(b)', '(b)', '(b)', '(b)']

    color_list1 = ['#3d5db7', '#ba3d1c']
    # color_list2 = ['#8db2e1', '#E3825D']
    color_list3 = ['#84C2E4', '#E59A5B']
    color_list4 = ['#BCBDC2', '#DDBBB3']
    color_list = []

    point_type1 = ['s', 'o']
    point_type2 = ['^', '<']
    point_type3 = ['v', 'D']
    point_type = []

    subplot_loc = [(2, 3, 1), (2, 3, 2), (2, 3, 3), (2, 3, 4), (2, 3, 5), (2, 3, 6)]

    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    font = {
        'family': 'Times New Roman',
        'weight': 'bold',
        'size': 10,
    }
    font2 = {
        'family': 'Times New Roman',
        'weight': 'bold',
        'size': 30,
    }
    a1 = plt.figure(figsize=(15, 8))
    for i in range(len(graph_type)):
        plt.subplot(subplot_loc[i][0], subplot_loc[i][1], subplot_loc[i][2])
        if i == 0 or i == 3:
            link_type = link_type1
            color_list = color_list1
            point_type = point_type1
        elif i == 1 or i == 4:
            link_type = link_type2
            color_list = color_list3
            point_type = point_type2
        else:
            link_type = link_type3
            color_list = color_list4
            point_type = point_type3

        for j in range(len(link_type)):
            y = []
            ans = 0
            path = 'F:/论文/lp-new/link-prediction-time-vary/results/probe_size/' + graph_type[i] + '/' + \
                   link_type[j]
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
            j_g = int(len(y) / 100)
            y1 = []
            yeer = []
            for y_value in range(100):
                y1.append(sum(y[y_value:y_value+j_g]) / j_g)
                yeer.append(np.std(y[y_value:y_value+j_g]))

            x1 = [_ for _ in range(100)]
            y_par = []
            x_par = []

            for k in range(100):
                y_par.append(y[int(k*(len(y) / 100))])
                x_par.append(x[int(k * (len(y) / 100))])
            plt.plot(x_par, y_par, color=color_list[j], marker=point_type[j], linestyle='-', markersize=3, linewidth=2)
            plt.ylim(-0.05, 1.05)

        if i == 0 or i == 3:
            plt.yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
            # plt.xticks([0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45])
            plt.tick_params(labelsize=20)

            plt.xlabel('t', font2, labelpad=8)
            plt.ylabel('AUC', font2, labelpad=10)
            # plt.errorbar(x, y, color=color_list[j], marker=point_type[j], linestyle='-', yerr=yerror, capsize=3)
            # plt.title(lab_list[i], font2, x=-0.17)
        else:
            plt.yticks([])
            plt.tick_params(labelsize=20)
            plt.xlabel('t', font2, labelpad=8)

    plt.subplots_adjust(wspace=0, hspace=0.5, left=0.07, right=0.98, bottom=0.1, top=0.9)
    plt.show()


def plot_fig11and12_ver2():
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'

    graph_type = ['contacts-prox-high-school-2013', 'SFHH-conf-sensor', 'contacts-prox-high-school-2013', 'SFHH-conf-sensor']
    alpha_list = [8, 6, 4, 2, 0, -2, -4, -6, -8]
    lab_list = ['(a)', '(b)', '(c)', '(d)']
    subplot_loc = [(2, 2, 1), (2, 2, 2), (2, 2, 3), (2, 2, 4)]

    # color2 = LinearSegmentedColormap.from_list("", ['#4791C5', '#FEE797', '#DE6C5B'])
    # color_list = ['#4791C5', '#D7E7F2', '#8EBBDB',  '#FEE7C6', '#F9E4E1', '#EBA69B', '#DE6C5B']
    # color2 = LinearSegmentedColormap.from_list("", color_list)
    color2 = LinearSegmentedColormap.from_list("", ['#4791C5', '#FFFFFF', '#DE6C5B'])

    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    font = {
        'family': 'Times New Roman',
        'weight': 'bold',
        'size': 25,
    }
    font2 = {
        'family': 'Times New Roman',
        'weight': 'bold',
        'size': 30,
    }

    a1 = plt.figure(figsize=(12, 10))
    for i in range(len(graph_type)):
        plt.subplot(subplot_loc[i][0], subplot_loc[i][1], subplot_loc[i][2])
        data = []
        for j in range(len(alpha_list)):
            store = []
            if i < 2:
                path = 'F:/论文/lp-new/link-prediction-time-vary/' + graph_type[i] + '/alpha' + str(alpha_list[j]) + '/deg'
            else:
                path = 'F:/论文/lp-new/link-prediction-time-vary/' + graph_type[i] + '/alpha' + str(alpha_list[j]) + '/dis'
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
        h_figure = sns.heatmap(data, robust=True, vmin=0, vmax=1, cmap=color2, annot_kws=font2, cbar=False)
        # cb = h_figure.figure.colorbar(h_figure.collections[0])  # 显示colorbar
        # cb.ax.tick_params(labelsize=17)

        if i == 0 or i == 2:
            plt.xticks(np.arange(0, 7001, 2000), np.arange(0, 7001, 2000), rotation=0)
        if i == 1 or i == 3:
            plt.xticks(np.arange(0, 3001, 1000), np.arange(0, 3001, 1000), rotation=0)
        plt.yticks([i+0.5 for i in range(9)], [8, 6, 4, 2, 0, -2, -4, -6, -8], rotation='horizontal')
        plt.xlabel('t', font2, labelpad=10)
        plt.ylabel(r'$\alpha$', font2, labelpad=10)
        plt.tick_params(labelsize=20)
        # plt.title(lab_list[i], font, x=-0.17)

    plt.subplots_adjust(wspace=0.2, hspace=0.4, left=0.1, right=0.98, bottom=0.1, top=0.95)
    plt.show()


def plot_fig11and12_dynamic():
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'

    graph_type = ['contacts-prox-high-school-2013', 'SFHH-conf-sensor', 'contacts-prox-high-school-2013', 'SFHH-conf-sensor']
    alpha_list = [8, 6, 4, 2, 0, -2, -4, -6, -8]
    lab_list = ['(a)', '(b)', '(c)', '(d)']
    subplot_loc = [(2, 2, 1), (2, 2, 2), (2, 2, 3), (2, 2, 4)]

    # color2 = LinearSegmentedColormap.from_list("", ['#4791C5', '#FEE797', '#DE6C5B'])
    # color_list = ['#4791C5', '#D7E7F2', '#8EBBDB',  '#FEE7C6', '#F9E4E1', '#EBA69B', '#DE6C5B']
    # color2 = LinearSegmentedColormap.from_list("", color_list)
    color2 = LinearSegmentedColormap.from_list("", ['#4791C5', '#FFFFFF', '#DE6C5B'])

    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    font = {
        'family': 'Times New Roman',
        'weight': 'bold',
        'size': 25,
    }
    font2 = {
        'family': 'Times New Roman',
        'weight': 'bold',
        'size': 30,
    }

    a1 = plt.figure(figsize=(12, 10))
    for i in range(len(graph_type)):
        plt.subplot(subplot_loc[i][0], subplot_loc[i][1], subplot_loc[i][2])
        data = []
        for j in range(len(alpha_list)):
            store = []
            if i < 2:
                path = 'F:/论文/lp-new/dynamic_stablization_link_prediction/' + graph_type[i] + '/alpha' + str(alpha_list[j]) + '/deg'
            else:
                path = 'F:/论文/lp-new/dynamic_stablization_link_prediction/' + graph_type[i] + '/alpha' + str(alpha_list[j]) + '/dis'
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
        h_figure = sns.heatmap(data, robust=True, vmin=0, vmax=1, cmap=color2, annot_kws=font2, cbar=False)
        # cb = h_figure.figure.colorbar(h_figure.collections[0])  # 显示colorbar
        # cb.ax.tick_params(labelsize=17)

        if i == 0 or i == 2:
            plt.xticks(np.arange(0, 7001, 2000), np.arange(0, 7001, 2000), rotation=0)
        if i == 1 or i == 3:
            plt.xticks(np.arange(0, 3001, 1000), np.arange(0, 3001, 1000), rotation=0)
        plt.yticks([i+0.5 for i in range(9)], [8, 6, 4, 2, 0, -2, -4, -6, -8], rotation='horizontal')
        plt.xlabel('t', font2, labelpad=10)
        plt.ylabel(r'$\alpha$', font2, labelpad=10)
        plt.tick_params(labelsize=20)
        # plt.title(lab_list[i], font, x=-0.17)

    plt.subplots_adjust(wspace=0.2, hspace=0.4, left=0.1, right=0.98, bottom=0.1, top=0.95)
    plt.show()


def plot_fig10_dynamic():
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    graph_type = ['contacts-prox-high-school-2013', 'SFHH-conf-sensor']

    link_type1 = ['deg', 'dis']

    lab_list = ['(a)', '(a)', '(a)', '(a)', '(b)', '(b)', '(b)', '(b)']

    color_list1 = ['#3d5db7', '#ba3d1c']
    # color_list2 = ['#8db2e1', '#E3825D']
    color_list3 = ['#84C2E4', '#E59A5B']
    color_list4 = ['#BCBDC2', '#DDBBB3']
    color_list = []

    point_type1 = ['s', 'o']
    point_type2 = ['^', '<']
    point_type3 = ['v', 'D']
    point_type = []

    subplot_loc = [(1, 2, 1), (1, 2, 2)]

    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    font = {
        'family': 'Times New Roman',
        'weight': 'bold',
        'size': 10,
    }
    font2 = {
        'family': 'Times New Roman',
        'weight': 'bold',
        'size': 30,
    }
    a1 = plt.figure(figsize=(12, 6))
    for i in range(len(graph_type)):
        plt.subplot(subplot_loc[i][0], subplot_loc[i][1], subplot_loc[i][2])

        for j in range(len(link_type1)):
            y = []
            ans = 0
            path = 'F:/论文/lp-new/dynamic_stablization_link_prediction/results/probe_size/' + graph_type[i] + '/' \
                   + link_type1[j]
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
            j_g = int(len(y) / 100)
            y1 = []
            yeer = []
            for y_value in range(100):
                y1.append(sum(y[y_value:y_value+j_g]) / j_g)
                yeer.append(np.std(y[y_value:y_value+j_g]))

            x1 = [_ for _ in range(100)]
            y_par = []
            x_par = []

            for k in range(100):
                y_par.append(y[int(k*(len(y) / 100))])
                x_par.append(x[int(k * (len(y) / 100))])
            plt.plot(x_par, y_par, color=color_list1[j], marker=point_type1[j], linestyle='-', markersize=3, linewidth=2)
            plt.ylim(-0.05, 1.05)


            plt.yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
            # plt.xticks([0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45])
            plt.tick_params(labelsize=20)

            plt.xlabel('t', font2, labelpad=8)
            plt.ylabel('AUC', font2, labelpad=10)

    plt.subplots_adjust(wspace=0.4, hspace=0.5, left=0.1, right=0.98, bottom=0.15, top=0.9)
    plt.show()


def plot_fig8and9_solofigure(type):
    graph_type = ['new FW1', 'new FW2', 'new FW3', 'sampled SmaGri', 'sampled Kohonen', 'sampled Delicious']
    probe_list = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45]
    alpha_list = [8, 6, 4, 2, 0, -2, -4, -6, -8]
    lab_list = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)']
    # subplot_loc = [(2, 3, 1), (2, 3, 2), (2, 3, 3), (2, 3, 4), (2, 3, 5), (2, 3, 6)]
    color_list3 = ['#84C2E4', '#E59A5B']
    # color2 = LinearSegmentedColormap.from_list("", ['#4791C5', '#FEE797', '#DE6C5B'])
    color2 = LinearSegmentedColormap.from_list("", ['#4791C5', '#FFFFFF', '#DE6C5B'])
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    font = {
        'family': 'Times New Roman',
        'weight': 'bold',
        'size': 10,
    }
    font2 = {
        'family': 'Times New Roman',
        'weight': 'bold',
        'size': 30,
    }

    # a1 = plt.figure(figsize=(8, 7))
    for i in range(len(graph_type)):
        a1 = plt.figure(figsize=(8, 7))
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
        if type == 'deg' and i < 3:
            h_figure = sns.heatmap(data, robust=True, vmin=0.29, vmax=0.71, cmap=color2, annot_kws=font2, cbar=False)
        else:
            h_figure = sns.heatmap(data, robust=True, vmin=0.15, vmax=0.85, cmap=color2, annot_kws=font2, cbar=False)
        # cb = h_figure.figure.colorbar(h_figure.collections[0])  # 显示colorbar
        # cb.ax.tick_params(labelsize=17)
        # if type == "dis":
        #     sns.heatmap(data, robust=True, vmin=0.2, vmax=0.8, cmap=color2, annot_kws={"front_size": 11})
        # else:
        #     sns.heatmap(data, robust=True, cmap=color2, annot_kws={"front_size": 15})

        plt.xticks([i+0.5 for i in range(9)], [0.05, '', 0.15, '', 0.25, '', 0.35, '', 0.45])
        plt.yticks([i+0.5 for i in range(9)], [8, 6, 4, 2, 0, -2, -4, -6, -8], rotation='horizontal')
        plt.xlabel('p', font2, labelpad=8)
        plt.ylabel(r'$\alpha$', font2, labelpad=10)

        plt.tick_params(labelsize=20)
        # plt.title(lab_list[i], font2, x=-0.17)

        plt.subplots_adjust(bottom=0.15)
        plt.show()


if __name__ == '__main__':
    # plot_fig8and9_solofigure('dis')
    # plot_fig10_dynamic()
    # plot_fig11and12_dynamic()
    # plot_fig11and12_ver2()
    # plot_fig10_ver2()
    # plot_empty_colorbar()
    # plot_fig4()
    # plot_fig3()
    # power_law_type = [2.5, 3.0, 4.0]
    # for i in range(3):
    #     plot_fig2(power_law_type[i])
        # plot_fig1(power_law_type[i])
    # plot_fig5and6('dis')
    # plot_fig5and6('deg')
    # plot_fig13()
    # plot_fig11and12('deg')
    # plot_fig11and12('dis')
    # plot_fig8and9('dis')
    # plot_fig8and9('deg')
    plot_fig10()
    # plot_fig7()

