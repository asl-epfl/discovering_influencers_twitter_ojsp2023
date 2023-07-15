import tweepy
import pandas as pd
import requests
from typing import List
import statistics
import networkx as nx
import matplotlib.pyplot as plt
import time
import seaborn as sns
from community import community_louvain
from math import isnan

bearer_token = "INSERT HERE"

client = tweepy.Client( bearer_token=bearer_token,  
                        return_type = requests.Response,
                        wait_on_rate_limit=True)

me = client.get_user(username="HeimishCon")
me = me.json()
me_data = me["data"]
me_id = me_data["id"]

ben_shapiro = client.get_user(username="benshapiro")
ben_shapiro = ben_shapiro.json()
ben_shapiro_data = ben_shapiro["data"]
ben_shapiroid = ben_shapiro_data["id"]

df = pd.read_csv("network_of_followers_ben_newester.csv") #Read into a df _added.csv
G = nx.from_pandas_edgelist(df, "source", "target")
print(len(G.nodes))

#time.sleep(5)
G_sorted = pd.DataFrame(sorted(G.degree, key=lambda x: x[1], reverse=True))
G_sorted.columns = ["nconst","degree"]
G_sorted.to_csv("ben_shapiro_network_degrees.csv")
print(G_sorted.head())
G_tmp = nx.k_core(G, 5) 
#G_tmp = G
print(len(G_tmp.nodes))

skewed_file = open("ben_skewed.txt","r")
skewed_file = skewed_file.read()
skewed_users = skewed_file.split("\n")
users_to_keep = skewed_users[:50]

refined_nodes = list(G_tmp.nodes)
nodes = refined_nodes.copy()
#for user in refined_nodes:
for user in nodes:
  if str(user) not in users_to_keep:
    G_tmp.remove_node(user)
  else:
    pass

G_tmp = nx.k_core(G_tmp, 2) 
ben_shapiroid_int= 17995040

new_edge_list = [(1181971448811986944,4235795894),
(222887440,1243930993259679744),
(928607826586374146,936592805660712960),
(1243930993259679744,1128332943061991426),
(51312363,1181971448811986944),
(749657035374034948,1243930993259679744),
(1181971448811986944,69684040),
(1181971448811986944,51312363),
(1181971448811986944,4235795894)]
G_tmp.add_edges_from(new_edge_list) 

new_edge_list = [(780014810591363072,17995040),
(1242077411900096513,17995040),
(51312363,17995040),
(69684040,17995040),
(237111933,17995040),
(1322491899979157509,17995040)
]
G_tmp.add_edges_from(new_edge_list) 
print(len(G_tmp.nodes))

partition = community_louvain.best_partition(G_tmp)
#Turn partition into dataframe
partition1 = pd.DataFrame([partition]).T
partition1 = partition1.reset_index()
partition1.columns = ['names','group']

G_sorted = pd.DataFrame(sorted(G_tmp.degree, key=lambda x: x[1], reverse=True))
print(G_sorted.head())
G_sorted.columns = ['names','degree']
G_sorted.head()
dc = G_sorted

combined = pd.merge(dc,partition1, how='left', left_on="names",right_on="names")
pos = nx.spring_layout(G_tmp)
f, ax = plt.subplots(figsize=(10, 10))
plt.style.use('ggplot')

#cc = nx.betweenness_centrality(G2)
nodes = nx.draw_networkx_nodes(G_tmp, pos,
                               cmap=plt.cm.Set1,
                               node_color=combined['group'],
                               alpha=0.8)
                               
nodes.set_edgecolor('k')
nx.draw_networkx_labels(G_tmp, pos, font_size=8)

nx.draw_networkx_edges(G_tmp, pos, width=1.0, alpha=0.2)

plt.savefig('ben_graph_23.png')

node_file = open("filtered_ben_skewed.txt","w")
nodes = list(G_tmp.nodes)

for node in nodes:
  node_file.write(str(node))
  if node != nodes[-1]:
    node_file.write("\n")

total_reqs = 0

def get_tweets(keyword,start_time,end_time="2022-04-30T00:00:00.000Z"):
  query =f"{keyword} lang:en"
  #time.sleep(3)
  tweets = client.search_all_tweets(query=query, max_results = 10, start_time=start_time,end_time = end_time, tweet_fields = ['text','created_at'])  
  time.sleep(3)
  tweets_dict = tweets.json() 
  global problematics
  global total_reqs
  if tweets_dict["meta"]["result_count"] != 0 : 
    tweets_data = tweets_dict['data'] 
    df = pd.json_normalize(tweets_data) 
    all_tweets = df["text"].tolist()
    tweet_dates = df["created_at"].tolist()
  else:
    print("returned empty lists")
    all_tweets = []
    tweet_dates = []
  
  total_reqs += 1
  print(f"Total reqs = {total_reqs}")
  return all_tweets,tweet_dates

node_file = open("filtered_ben_skewed.txt","r")
node_file_r = node_file.read()
remaining_nodes = node_file_r.split("\n")

total_activites = []
print(len(remaining_nodes))


date_file = open("date_search_2017.txt","r")
date_file_r = date_file.read()
dates = date_file_r.split("\n")

start_dates = dates[:-1]
end_dates = dates[1:]
assert len(start_dates)==len(end_dates)

mode_name = ["ben_skewed"]

for bl in mode_name:

    print(bl)
    
    trump_start = dates[:1440]
    trump_end = dates[1:1441]
    biden_start = dates[1440:-1]
    biden_end = dates[1441:]
    
    bin_no = (len(trump_end)+len(biden_end))
    print(bin_no)
    
    activities = {}
    bin_tweets = []
    ext_tweets = {}

    sayac = 0

    for userid in remaining_nodes:

        print("Doing:",userid)
        sayac += 1
        print(sayac)
        
        user_activity = [0]*bin_no
        user_tweets = []
        #inner_sayac = 0

        k = 0
    
        for l in range(len(trump_end)):
            print(k+l)
            print(trump_start[l])
            trump_tweets,_ = get_tweets(f"from:{userid} trump",trump_start[l],trump_end[l])
            #time.sleep(3)
            my_str = ""
            
            for tw in trump_tweets:
                my_str += tw
                if tw != trump_tweets[-1]:
                    my_str += "STOP"
            user_tweets.append(my_str)
            
            if trump_tweets != [] :
                user_activity[k+l]=1
            l += 1
    
        for m in range(len(biden_end)):
            print(k+l+m)
            print(biden_start[m])
            biden_tweets,_ = get_tweets(f"from:{userid} biden",biden_start[m],biden_end[m])
            #time.sleep(3)
            my_str = ""
            for tw in biden_tweets:
                my_str += tw
                if tw != biden_tweets[-1]:
                    my_str += "STOP"
            user_tweets.append(my_str)
    
            if biden_tweets != [] :
                user_activity[k+l+m]=1
      
        ext_tweets[userid]=user_tweets
        activities[userid]=user_activity
  
        df_ext = pd.DataFrame.from_dict(ext_tweets)
        df_ext.to_csv(f"ben_tweets_extracted.csv",index=False)
        
        df_act = pd.DataFrame.from_dict(activities)
        df_act.to_csv(f"ben_activity_matrix.csv",index=False)

    df_ext = pd.DataFrame.from_dict(ext_tweets)
    df_ext.to_csv(f"ben_tweets_extracted.csv",index=False)
        
    df_act = pd.DataFrame.from_dict(activities)
    df_act.to_csv(f"ben_activity_matrix.csv",index=False)
