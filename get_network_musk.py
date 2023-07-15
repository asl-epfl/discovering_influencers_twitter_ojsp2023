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

me = client.get_user(username="MrLexton")
me = me.json()
me_data = me["data"]
me_id = me_data["id"]

elon_musk = client.get_user(username="elonmusk")
elon_musk = elon_musk.json()
elon_musk_data = elon_musk["data"]
elon_muskid = elon_musk_data["id"]

user_list = [me_id]
follower_list = []
for user in user_list:
    followers = []
    followers_response = client.get_users_followers(id = user, max_results = 1000)
    fols = followers_response.json()
    data = fols["data"]
    followers = []
    for i in data:
      followers.append(i["id"])
    if elon_muskid not in followers:
        followers.append(elon_muskid)
    if EvaFoxUid not in followers:
        followers.append(EvaFoxUid)
    if westcoastbillid not in followers:
        followers.append(westcoastbillid)
    #followers.append(user)
    print(len(followers))
    follower_list.append(followers)
    
 
df = pd.DataFrame(columns=['source','target']) #Empty DataFrame
df['target'] = follower_list[0] #Set the list of followers as the target column
df['source'] = int(me_id)


print(df)

time.sleep(3)

count = 0

user_list = list(df['target']) #Use the list of followers we extracted in the code above i.e. my 450 followers


for userID in user_list:

    print(userID)
    followers = []
    follower_list = []
    
    identified_followers = []

    if userID == elon_muskid:
        
        flag = True
        pagination = True
        next_token = None

        while flag:
                if len(identified_followers) > 10:
                    break
                print("Total Identified followers:",len(identified_followers))
                # Check if max_count reached
                print("-------------------")
                print("Token: ", next_token)
                ##url = create_url(keyword, start_list[i],end_list[i], max_results)
                followers_response = client.get_users_followers(id = userID, pagination_token =next_token, max_results = 1000)

                #json_response = connect_to_endpoint(url[0], headers, url[1], next_token)
                time.sleep(3)
                json_response = fols = followers_response.json()
                
                data = fols["data"]
                
                for i in data:
                    if i["id"] in user_list:
                        identified_followers.append(i["id"])

                # If no next token exists

                if pagination:
                    if 'next_token' in json_response['meta']:
                    # Save the token to use for next call
                        next_token = json_response['meta']['next_token']
                        print("Next Token: ", next_token)
                    else:
                        flag=False
                        next_token = None
                else:
                    #Since this is the final request, turn flag to false to move to the next time period.
                    flag = False
                    next_token = None
                time.sleep(1)

        df = pd.DataFrame(columns=['source','target']) #Empty DataFrame
        df['target'] = identified_followers #Set the list of followers as the target column
        df['source'] = int(me_id)

        df.to_csv("network_of_followers_musk.csv") ###network_of_followers_musk_to
        break

############ SEE THEIR ACTIVITY

df = pd.read_csv("network_of_followers_musk.csv") #Read into a df
print(df.head())
G = nx.from_pandas_edgelist(df, "source", "target")

G_sorted = pd.DataFrame(sorted(G.degree, key=lambda x: x[1], reverse=True))
G_sorted.columns = ["nconst","degree"]

G_tmp = nx.k_core(G, 5) 
adjacency_matrix = nx.adjacency_matrix(G_tmp)
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

    keyword = "coin"
    query =f"{keyword} from:{userID} lang:en"
    tweets = client.search_all_tweets(query=query, max_results = max_results, start_time='2021-01-01T00:00:00.000Z',end_time='2021-07-01T00:00:00.000Z', tweet_fields = ['text']) 
    time.sleep(2)
    json_response001 = tweets_dict = tweets.json()

    keyword = "coin"
    query =f"{keyword} from:{userID} lang:en"
    tweets = client.search_all_tweets(query=query, max_results = max_results, start_time='2021-07-01T00:00:00.000Z',end_time='2022-01-01T00:00:00.000Z', tweet_fields = ['text']) 
    time.sleep(2)
    json_response01 = tweets_dict = tweets.json()

    keyword = "coin"
    query =f"{keyword} from:{userID} lang:en"
    tweets = client.search_all_tweets(query=query, max_results = max_results, start_time='2022-01-01T00:00:00.000Z', tweet_fields = ['text']) 
    time.sleep(2)
    json_response1 = tweets_dict = tweets.json()
    
    keyword = "coin"
    query =f"{keyword} from:{userID} lang:en"
    tweets2 = client.search_all_tweets(query=query, max_results = max_results, start_time='2017-01-01T00:00:00.000Z', end_time='2017-07-01T00:00:00.000Z', tweet_fields = ['text']) 
    time.sleep(2)
    json_response00000002 = tweets_dict2 = tweets2.json()

    keyword = "coin"
    query =f"{keyword} from:{userID} lang:en"
    tweets2 = client.search_all_tweets(query=query, max_results = max_results, start_time='2017-07-01T00:00:00.000Z', end_time='2018-01-01T00:00:00.000Z', tweet_fields = ['text']) 
    time.sleep(2)
    json_response0000002 = tweets_dict2 = tweets2.json()

    keyword = "coin"
    query =f"{keyword} from:{userID} lang:en"
    tweets2 = client.search_all_tweets(query=query, max_results = max_results, start_time='2018-01-01T00:00:00.000Z', end_time='2018-07-01T00:00:00.000Z', tweet_fields = ['text']) 
    time.sleep(2)
    json_response000002 = tweets_dict2 = tweets2.json()
    
    keyword = "coin"
    query =f"{keyword} from:{userID} lang:en"
    tweets2 = client.search_all_tweets(query=query, max_results = max_results, start_time='2018-07-01T00:00:00.000Z', end_time='2019-01-01T00:00:00.000Z', tweet_fields = ['text']) 
    time.sleep(2)
    json_response00002 = tweets_dict2 = tweets2.json()

    keyword = "coin"
    query =f"{keyword} from:{userID} lang:en"
    tweets2 = client.search_all_tweets(query=query, max_results = max_results, start_time='2019-01-01T00:00:00.000Z', end_time='2019-07-01T00:00:00.000Z', tweet_fields = ['text']) 
    time.sleep(2)
    json_response0002 = tweets_dict2 = tweets2.json()

    keyword = "coin"
    query =f"{keyword} from:{userID} lang:en"
    tweets2 = client.search_all_tweets(query=query, max_results = max_results, start_time='2019-07-01T00:00:00.000Z', end_time='2020-01-01T00:00:00.000Z', tweet_fields = ['text']) 
    time.sleep(2)
    json_response002 = tweets_dict2 = tweets2.json()
    
    keyword = "coin"
    query =f"{keyword} from:{userID} lang:en"
    tweets2 = client.search_all_tweets(query=query, max_results = max_results, start_time='2020-01-01T00:00:00.000Z', end_time='2020-07-01T00:00:00.000Z', tweet_fields = ['text']) 
    time.sleep(2)
    json_response02 = tweets_dict2 = tweets2.json()

    keyword = "coin"
    query =f"{keyword} from:{userID} lang:en"
    tweets2 = client.search_all_tweets(query=query, max_results = max_results, start_time='2020-07-01T00:00:00.000Z', end_time='2021-01-01T00:00:00.000Z', tweet_fields = ['text']) 
    time.sleep(2)
    json_response2 = tweets_dict2 = tweets2.json()
    
    result_count = json_response1['meta']['result_count'] + json_response01['meta']['result_count'] + json_response001['meta']['result_count'] + json_response2['meta']['result_count'] + json_response02['meta']['result_count'] +\
    json_response002['meta']['result_count'] + json_response0002['meta']['result_count'] + json_response00002['meta']['result_count'] + json_response000002['meta']['result_count'] + json_response0000002['meta']['result_count'] +\
    json_response00000002['meta']['result_count']

    if result_count is not None and result_count > 0 :
      count = result_count
    else:
      count = 0
    activity.append((userID,count))
    df_activity = pd.DataFrame(sorted(activity, key=lambda x: x[1], reverse=True), columns =['UserID', 'Count'])
    df_activity.to_csv('df_activity_musk_users_4.csv')
print(df_activity)