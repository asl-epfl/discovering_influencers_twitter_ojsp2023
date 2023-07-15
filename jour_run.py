import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
import pandas as pd
import networkx as nx
import time
from sklearn.cluster import KMeans
from scipy.stats import hmean
from utils.utils import metropolis_rule,averaging_rule,grouper,plot_weighted_graphs,plot_influences,plot_influences_single_label
import pickle
import os 
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
import tweepy
from community import community_louvain
import requests
from tqdm import tqdm
np.random.seed(45)


mu = 0.0003
delta = 0.0001
reg = 0.006
repeat_time = 150
project = False
learning_choice = 0
normalization_step = True
BATCH_SIZE=30
verbose = True

data_path = "twitter_data_ff/journalist_beliefs.txt"

user_data_file = open(data_path)
user_data_read = user_data_file.read()
user_data_strings = user_data_read.split("\n")

user_data_matrix = []
for user in user_data_strings:
    user_sentiments = user.split(",")
    l_user_sentiments = []
    for se in user_sentiments:
        l_user_sentiments.append(float(se))
    user_data_matrix.append(np.array(l_user_sentiments))

lambda_matrix = np.array(user_data_matrix).T

org_combination_matrix = np.load("twitter_data_ff/journalist_original_combination_matrix.npy")

def expected_to_A(matrix,p_of_abs):
    EA = np.zeros(matrix.shape)
    for l in range(EA.shape[0]):
        for k in range(EA.shape[1]):
            if l!=k:
                EA[l,k] = matrix[l,k] / (1-p_of_abs[l])
    for l in range(EA.shape[0]):
        for k in range(EA.shape[1]):
            if l==k:
                EA[l,k] = 1-(np.sum(EA[:,k])-EA[l,k])
    return EA

def check_left_stochastic(matrix):
    col_sums = np.sum(matrix,axis=0,keepdims=True)
    return col_sums

def get_order(arr):
    
    return np.flip(np.argsort(arr))

bin_count = lambda_matrix.shape[0]

A = np.random.random((lambda_matrix.shape[1],lambda_matrix.shape[1]))
A = A / np.sum(A,axis=0)

#A = np.ones((67,67)) * 1 / 67

loss_adj = []
loss_combination = []

progress = []



#offset = np.random.randint(bin_count//2,bin_count-3)

#offset = bin_count - 5
offset = 0
print(f"OFFSET: {offset}")

one_user = []

#print(A)
#assert 1==0



print(lambda_matrix.shape)

print(lambda_matrix[0,:].shape)

L_hat_history = []
#time.sleep(2)

one_agent_L_hat = []
indices = np.arange(1,lambda_matrix.shape[0])
batches = list(grouper(BATCH_SIZE,indices))
if verbose:
    outer_loop = tqdm(range(repeat_time))
else:
    outer_loop = range(repeat_time)
for repeating in outer_loop:
    for batch in batches:
        combined_gradient = np.zeros(A.shape)
        prev_A = A.copy()
        for time_stamp in batch:
            #print(repeating,time_stamp)
            i = time_stamp + offset
            prev_lambda = lambda_matrix[i-1].reshape((A.shape[0],1))
            cur_lambda = lambda_matrix[i].reshape((A.shape[0],1))
            
            L_hat = np.zeros(cur_lambda.shape)
            if i != 1:
                for j in range(1,i):
                    L_hat += lambda_matrix[j].reshape((A.shape[0],1)) - (1-delta)* A @ lambda_matrix[j-1].reshape((A.shape[0],1))
                L_hat = L_hat / (delta*(i-1))
            L_hat_history.append(np.linalg.norm(L_hat))
            if i != 1:
                inner_gradient = prev_lambda - (np.sum(lambda_matrix[:i-1,:],axis=0,keepdims=True)/(i-1)).T
            else:
                inner_gradient = prev_lambda

            one_agent_L_hat.append(L_hat[0])
            outer_gradient = cur_lambda.T - (1-delta)* prev_lambda.T@A - delta * L_hat.T
            total_gradient = ( -(1-delta)*(inner_gradient@outer_gradient) )
            if learning_choice == 0:
                new_gradient = total_gradient
                
            elif learning_choice == 1:
                new_gradient = np.zeros(total_gradient.shape)

                activity_vector = activity_matrix[i]
                #print("activity_vector:",activity_vector.shape) prints as desired
                for l in range(new_gradient.shape[0]):
                    if activity_vector[l] == 1:
                        print(f"agent {l} is active")
                        new_gradient[l,:] = total_gradient[l,:]
                    elif activity_vector[l] == 0:
                        for k in range(new_gradient.shape[0]):
                            new_gradient[l,k] = total_gradient[k,k]
                        print(f"agent {l} is NOT active")
            else:
                new_gradient = np.zeros(total_gradient.shape)
            combined_gradient += new_gradient
        
        combined_gradient = combined_gradient/len(batch)
        #print(np.linalg.norm(A))
        #time.sleep(0.05)
        A = A - mu* (combined_gradient +  reg * np.sign(A))
        if project:
            A = A*((A>0).astype(int))
            A = A /  np.sum(A,axis=0)
            #A = (A + A.T) / 2
            pass
                

        one_user.append(A[10])
        new_A = A.copy()
        #print(new_A.shape)
        #print(org_combination_matrix.shape)
        progress.append(np.linalg.norm((new_A-prev_A)))
        loss_combination.append(np.linalg.norm(new_A-org_combination_matrix))

L_hat = np.zeros(cur_lambda.shape)
for j in range(1,i):
     L_hat += lambda_matrix[j].reshape((A.shape[0],1)) - (1-delta)* A @ lambda_matrix[j-1].reshape((A.shape[0],1))
L_hat = L_hat / (delta*(i-1))

normalization_step = True
if normalization_step:
    A = A /  np.sum(A,axis=0)

final_A = A.copy()

perron_eig = np.linalg.eig(org_combination_matrix)[1][:,np.argmax(np.linalg.eig(org_combination_matrix)[0])]

perron_eig = perron_eig/perron_eig.sum()

perron_eig_learned = np.abs(np.linalg.eig(final_A)[1][:,np.argmax(np.linalg.eig(final_A)[0])]) 
perron_eig_learned = perron_eig_learned / perron_eig_learned.sum()

perron_eig_sorted = np.flip(np.sort(perron_eig))
perron_eig_ordering = get_order(perron_eig)
perron_eig_learned_sorted = perron_eig_learned[perron_eig_ordering]

L_hat = -L_hat/np.linalg.norm(L_hat)

influence_original = perron_eig * L_hat.T
influence_learned =  perron_eig_learned * L_hat.T
influence_original = influence_original[0]
influence_learned = influence_learned[0]

influence_original_sorted = np.flip(np.sort(influence_original))
influence_ordering = get_order(influence_original)
influence_learned_sorted = influence_learned[influence_ordering]

show_binary = True
plot_influences(perron_eig_learned_sorted,perron_eig_sorted,show=show_binary)
influences_sorted = influence_learned_sorted*((influence_learned_sorted>0).astype(int))
plot_influences_single_label(influences_sorted,show=show_binary)