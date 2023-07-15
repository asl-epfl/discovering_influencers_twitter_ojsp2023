from itertools import combinations
from itertools import islice
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors
import networkx as nx
import seaborn as sns
from matplotlib import cm
import matplotlib as mpl

LIGHT_BLUE = "#8ECAE6"
BLUE = "#3A7CB8"#"#219EBC"
DARK_BLUE = "#3A7CB8" #"#035177"
YELLOW = "#FFB703"
ORANGE = "#E5AD02" #"#FB8500"
RED = "#F04532" #"#F00314"
GREEN = "#75A43A" #"#14B37D"
colors = sns.color_palette("tab10")
colors_dark = sns.color_palette("dark")

def plot_weighted_graphs(combination_matrix, combination_matrix_learned,
                         adjacency_matrix, by_side=True, adj_regime=1,path=None,show=True):
    combination_matrix = combination_matrix + combination_matrix.T
    combination_matrix_learned = combination_matrix_learned + combination_matrix_learned.T

    pos = nx.spring_layout(nx.from_numpy_matrix(adjacency_matrix))
    if adj_regime == 5:
        N = combination_matrix.shape[0]
        pos = {}
        pos[0] = np.array([-1, 0])
        pos[1] = np.array([1, 0])
        for k in range(2, N):
            pos[k] = np.array([0, -1 + 2*(k-2) / (N-3)])
    G = nx.from_numpy_matrix(combination_matrix)
    edges = G.edges()
    weights = [G[u][v]['weight']*5 for u, v in edges]

    G_learned = nx.from_numpy_matrix(combination_matrix_learned)
    edges_learned = G_learned.edges()
    weights_learned = [G_learned[u][v]['weight']*5 for u, v in edges_learned]
    if by_side:
        fig, axs = plt.subplots(ncols=2, gridspec_kw=dict(width_ratios=[4, 4]), figsize=(10, 4))

        axs[0].title.set_text('true graph')
        nx.draw_networkx_nodes(G, pos=pos, node_color=BLUE, node_size=100, ax=axs[0])
        nx.draw_networkx_edges(G, pos=pos, edge_color=DARK_BLUE, width=weights, ax=axs[0])

        axs[1].title.set_text('estimated graph')
        nx.draw_networkx_nodes(G_learned, pos=pos, node_color=BLUE, node_size=100, ax=axs[1])
        nx.draw_networkx_edges(G_learned, pos=pos, edge_color=DARK_BLUE, width=weights_learned, ax=axs[1])
        if path is not None:
            plt.savefig(path)
        if show==True:
            plt.show()
            
    else:
        plt.figure(figsize=(8, 7))
        nx.draw_networkx_nodes(G, pos=pos, node_color=ORANGE, node_size=350)
        nx.draw_networkx_edges(G, pos=pos, edge_color=DARK_BLUE, width=weights)
        if show==True:
            plt.show()

        plt.figure(figsize=(8, 7))
        nx.draw_networkx_nodes(G_learned, pos=pos, node_color=ORANGE, node_size=350)
        nx.draw_networkx_edges(G_learned, pos=pos, edge_color=DARK_BLUE, width=weights_learned)
        if path is not None:
            plt.savefig(path)
        if show==True:
            plt.show()

def plot_centrality(influences_learn, influences_true,show=False, window="adaptive",path=None):
    plt.figure()
    agents = influences_learn.shape[0]
    plt.scatter(np.arange(agents), influences_true, s=80, color=colors[1], alpha=0.8,
                label="true")
    plt.vlines(np.arange(agents), 0, influences_true, color=colors[1], alpha=0.8,)
    plt.scatter(np.arange(agents), influences_learn, s=80, color=colors[0], alpha=0.8,
                label="$M = " + str(window) + '$')
    plt.vlines(np.arange(agents), 0, influences_learn, color=colors[0], alpha=0.8,)
    #plt.xticks(np.arange(agents))
    #plt.title("Centrality")
    plt.grid()
    plt.xlabel('agent')
    plt.legend()
    if path!=None:
        plt.savefig(path)
    if show:
        plt.show()

def plot_influences(influences_learn, influences_true,show=False, window="learned",path=None,xticks = None,xlabels=None):
    plt.figure()
    agents = influences_learn.shape[0]
    plt.scatter(np.arange(agents), influences_true, s=80, color=colors[1], alpha=0.8,
                label="true")
    plt.vlines(np.arange(agents), 0, influences_true, color=colors[1], alpha=0.8,)
    plt.scatter(np.arange(agents), influences_learn, s=80, color=colors[0], alpha=0.8,
                label="learned")
    plt.vlines(np.arange(agents), 0, influences_learn, color=colors[0], alpha=0.8,)
    #plt.xticks(np.arange(agents))
    #plt.title("Influneces")
    plt.grid()
    plt.xlabel('agent')
    plt.ylabel('centrality')
    if xticks!=None:
        plt.xticks(ticks=xticks, labels=xlabels)
    plt.legend()
    if path!=None:
        plt.savefig(path)
    if show:
        plt.show()

def plot_follower_count(influences_learn, influences_true,show=False, window="adaptive",path=None):
    agents = influences_learn.shape[0]
    plt.figure()
    plt.scatter(np.arange(agents), influences_true, s=80, color=colors[1], alpha=0.8)
    plt.vlines(np.arange(agents), 0, influences_true, color=colors[1], alpha=0.8,)
    plt.scatter(np.arange(agents), influences_learn, s=80, color=colors[0], alpha=0.8)
    plt.vlines(np.arange(agents), 0, influences_learn, color=colors[0], alpha=0.8,)
    #plt.xticks(np.arange(agents))
    plt.title("Follower Count")
    plt.grid()
    plt.xlabel('agent')
    plt.legend()
    if path!=None:
        plt.savefig(path)
    if show:
        plt.show()

def plot_influences_single_label(influences_learn,show=False, window="adaptive",path=None,xticks = None,xlabels=None):
    plt.figure()
    agents = influences_learn.shape[0]
    plt.scatter(np.arange(agents), influences_learn, s=80, color=colors[0], alpha=0.8)
    plt.vlines(np.arange(agents), 0, influences_learn, color=colors[0], alpha=0.8,)
    #plt.xticks(np.arange(agents))
    plt.grid()
    plt.xlabel('agent')
    plt.ylabel("influence")
    if xticks!=None:
        plt.xticks(ticks=xticks, labels=xlabels)
    plt.legend()
    if path!=None:
        plt.savefig(path)
    if show:
        plt.show()


def metropolis_rule(matrix):

    combination_matrix = np.zeros(matrix.shape)

    degs = np.zeros(matrix.shape[0])
    for i in range(matrix.shape[0]):
        degs[i] = np.sum(matrix[:,i])

    for l in range(matrix.shape[0]):
        for k in range(matrix.shape[1]):
            if k!=l and matrix[l,k]!=0:
                combination_matrix[l,k] = 1/max(degs[l],degs[k])
    
    for l in range(matrix.shape[0]):
        for k in range(matrix.shape[1]):
            if k==l:
                combination_matrix[l,k] = 1 - np.sum(combination_matrix[:,k])
    
    return combination_matrix

def averaging_rule(matrix,A_weight=1):

    combination_matrix = matrix
    A = combination_matrix /  np.sum(combination_matrix,axis=0)
    I = np.eye(A.shape[0])
    return A_weight*A+(1-A_weight)*I
    #return A

def e_vector(pos,length):
    vec = np.zeros((length,1))
    vec[pos] = 1
    return vec

def grouper(n, iterable):
    it = iter(iterable)
    while True:
        chunk = tuple(islice(it, n))
        if not chunk:
            return
        yield chunk

def optimize(adj_matrix_prev,cur_lambda,prev_lambda,lr,delta,L_hat,reg):
    multiplier_1 = 1-delta
    multiplier_2 = delta
    kl_div = L_hat
    
    adj = adj_matrix_prev.T + lr*(
                multiplier_1*(cur_lambda - multiplier_1*adj_matrix_prev.T@prev_lambda - multiplier_2*kl_div)@prev_lambda.T -\
                reg*(adj_matrix_prev.T / np.abs(adj_matrix_prev.T))
        )
    adj = adj.T

    return adj

def return_grad(adj_matrix_prev,cur_lambda,prev_lambda,lr,delta,L_hat,reg,absence_vector,lambda_matrix,i=0):
    multiplier_1 = 1-delta
    multiplier_2 = delta

    inner_gradient = prev_lambda - (np.sum(lambda_matrix[:i,:],axis=0,keepdims=True)/i).T


    first_grad = multiplier_1*((-inner_gradient)@(cur_lambda - multiplier_1*adj_matrix_prev.T@prev_lambda - multiplier_2*L_hat).T)
    
    my_grad = np.zeros(adj_matrix_prev.shape)
    for l in range(my_grad.shape[0]):
                if absence_vector[l] == 0:
                    my_grad[l,:] = first_grad[l,:]
                elif absence_vector[l] == 1:
                    for k in range(my_grad.shape[0]):
                        my_grad[l,k] = first_grad[k,k]

    return first_grad,my_grad