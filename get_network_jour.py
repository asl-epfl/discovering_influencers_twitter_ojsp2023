import tweepy
import pandas as pd
import numpy as np
import requests
from typing import List
import statistics
import networkx as nx
import matplotlib.pyplot as plt
import time
import seaborn as sns
from community import community_louvain
from math import isnan
from utils.utils import metropolis_rule
import seaborn as sns

bearer_token = "INSERT HERE"

client = tweepy.Client( bearer_token=bearer_token,  
                        return_type = requests.Response,
                        wait_on_rate_limit=True)


me = client.get_user(username="retripp")
me = me.json()
me_data = me["data"]
me_id = me_data["id"]

jour = client.get_user(username="maggieNYT")
jour = jour.json()
jour_data = jour["data"]
jourid = jour_data["id"]


user_list = [me_id]
follower_list = []
for user in user_list:
    followers = []
    followers_response = client.get_users_followers(id = user, max_results = 500)
    fols = followers_response.json()
    data = fols["data"]
    followers = []
    for i in data:
      followers.append(i["id"])
    if jourid not in followers:
        followers.append(jourid)
    #followers.append(user)
    print(len(followers))
    follower_list.append(followers)
    
 
df = pd.DataFrame(columns=['source','target']) #Empty DataFrame
df['target'] = follower_list[0] #Set the list of followers as the target column
df['source'] = int(me_id)


print(df)

user_list = list(df['target']) #Use the list of followers we extracted in the code above i.e. my 450 followers

time.sleep(5)


for userID in user_list:

    print(userID)
    followers = []
    follower_list = []
    
    count += 1
    print(count)
    try:
        followers_response = client.get_users_followers(id = userID, max_results = 500)
        fols = followers_response.json()
        data = fols["data"]
        time.sleep(5)
        for i in data:
            followers.append(i["id"])
        print(len(followers))
        
    except:
        print("error")
        continue

    #follower_list.append(followers)
    temp = pd.DataFrame(columns=['source', 'target'])
    #temp['target'] = follower_list[0]
    temp['target'] = followers
    temp['source'] = userID
    df = df.append(temp)
    df.to_csv("network_of_followers_jour.csv")

### SECOND STAGE

df = pd.read_csv("network_of_followers_jour.csv") #Read into a df
print(df.head())
G = nx.from_pandas_edgelist(df, "source", "target")

G_sorted = pd.DataFrame(sorted(G.degree, key=lambda x: x[1], reverse=True))
G_sorted.columns = ["nconst","degree"]


G_tmp = nx.k_core(G, 4) 
adjacency_matrix = nx.adjacency_matrix(G_tmp)
#print(G_tmp.degree)
nodes = G_tmp.nodes
activity = []
max_results = 500
iteration = 0
offset = 0
print(len(list(nodes)))
for i in range(len(nodes)):
    iteration = i + offset
    userID = list(nodes)[iteration]
    iteration += 1

    print(iteration)

    keyword = "biden"
    query =f"{keyword} from:{userID} lang:en"
    tweets = client.search_all_tweets(query=query, max_results = max_results, start_time='2021-01-01T00:00:00.000Z',end_time='2021-07-01T00:00:00.000Z', tweet_fields = ['text']) 
    time.sleep(2)
    json_response001 = tweets_dict = tweets.json()


    keyword = "biden"
    query =f"{keyword} from:{userID} lang:en"
    tweets = client.search_all_tweets(query=query, max_results = max_results, start_time='2021-07-01T00:00:00.000Z',end_time='2022-01-01T00:00:00.000Z', tweet_fields = ['text']) 
    time.sleep(2)
    json_response01 = tweets_dict = tweets.json()


    keyword = "biden"
    query =f"{keyword} from:{userID} lang:en"
    tweets = client.search_all_tweets(query=query, max_results = max_results, start_time='2022-01-01T00:00:00.000Z',end_time='2022-07-01T00:00:00.000Z',tweet_fields = ['text']) 
    time.sleep(2)
    json_response1 = tweets_dict = tweets.json()

    keyword = "biden"
    query =f"{keyword} from:{userID} lang:en"
    tweets = client.search_all_tweets(query=query, max_results = max_results, start_time='2022-07-01T00:00:00.000Z',tweet_fields = ['text']) 
    time.sleep(2)
    json_response10= tweets_dict = tweets.json()
    
    
    keyword = "trump"
    query =f"{keyword} from:{userID} lang:en"
    tweets2 = client.search_all_tweets(query=query, max_results = max_results, start_time='2017-01-01T00:00:00.000Z', end_time='2017-07-01T00:00:00.000Z', tweet_fields = ['text']) 
    time.sleep(2)
    json_response00000002 = tweets_dict2 = tweets2.json()

    keyword = "trump"
    query =f"{keyword} from:{userID} lang:en"
    tweets2 = client.search_all_tweets(query=query, max_results = max_results, start_time='2017-07-01T00:00:00.000Z', end_time='2018-01-01T00:00:00.000Z', tweet_fields = ['text']) 
    time.sleep(2)
    json_response0000002 = tweets_dict2 = tweets2.json()

    keyword = "trump"
    query =f"{keyword} from:{userID} lang:en"
    tweets2 = client.search_all_tweets(query=query, max_results = max_results, start_time='2018-01-01T00:00:00.000Z', end_time='2018-07-01T00:00:00.000Z', tweet_fields = ['text']) 
    time.sleep(2)
    json_response000002 = tweets_dict2 = tweets2.json()
    
    
    keyword = "trump"
    query =f"{keyword} from:{userID} lang:en"
    tweets2 = client.search_all_tweets(query=query, max_results = max_results, start_time='2018-07-01T00:00:00.000Z', end_time='2019-01-01T00:00:00.000Z', tweet_fields = ['text']) 
    time.sleep(2)
    json_response00002 = tweets_dict2 = tweets2.json()

    
    keyword = "trump"
    query =f"{keyword} from:{userID} lang:en"
    tweets2 = client.search_all_tweets(query=query, max_results = max_results, start_time='2019-01-01T00:00:00.000Z', end_time='2019-07-01T00:00:00.000Z', tweet_fields = ['text']) 
    time.sleep(2)
    json_response0002 = tweets_dict2 = tweets2.json()


    keyword = "trump"
    query =f"{keyword} from:{userID} lang:en"
    tweets2 = client.search_all_tweets(query=query, max_results = max_results, start_time='2019-07-01T00:00:00.000Z', end_time='2020-01-01T00:00:00.000Z', tweet_fields = ['text']) 
    time.sleep(2)
    json_response002 = tweets_dict2 = tweets2.json()
    
    keyword = "trump"
    query =f"{keyword} from:{userID} lang:en"
    tweets2 = client.search_all_tweets(query=query, max_results = max_results, start_time='2020-01-01T00:00:00.000Z', end_time='2020-07-01T00:00:00.000Z', tweet_fields = ['text']) 
    time.sleep(2)
    json_response02 = tweets_dict2 = tweets2.json()

    keyword = "trump"
    query =f"{keyword} from:{userID} lang:en"
    tweets2 = client.search_all_tweets(query=query, max_results = max_results, start_time='2020-07-01T00:00:00.000Z', end_time='2021-01-01T00:00:00.000Z', tweet_fields = ['text']) 
    time.sleep(2)
    json_response2 = tweets_dict2 = tweets2.json()
    
    result_count = json_response10['meta']['result_count'] + json_response1['meta']['result_count'] + json_response01['meta']['result_count'] + json_response001['meta']['result_count'] + json_response2['meta']['result_count'] + json_response02['meta']['result_count'] +\
    json_response002['meta']['result_count'] + json_response0002['meta']['result_count'] + json_response00002['meta']['result_count'] + json_response000002['meta']['result_count'] + json_response0000002['meta']['result_count'] +\
    json_response00000002['meta']['result_count']

  
  
    if result_count is not None and result_count > 0 :
      count = result_count
    else:
      count = 0
    activity.append((userID,count))
    df_activity = pd.DataFrame(sorted(activity, key=lambda x: x[1], reverse=True), columns =['UserID', 'Count'])
    df_activity.to_csv('df_activity_jour_users.csv')

