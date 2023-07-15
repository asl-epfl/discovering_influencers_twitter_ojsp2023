from utils import roberta_plus_sent_anal

import tweepy
import requests
from textblob import TextBlob
import preprocessor as p
import statistics
from typing import List
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from math import isnan

def Average(array):
    avg = np.sum(array) / array.shape[0]
    rounded = round(avg,2)
    return rounded

def clean_tweets(all_tweets):
    if type(all_tweets) != str:
        tweets_clean = []
        for tweet in all_tweets:
            tweets_clean.append(p.clean(tweet)) 
        return tweets_clean
    else:
        tweet = all_tweets
        return p.clean(tweet)

if __name__ == "__main__":

    df_tweets = pd.read_csv("ben_skewed_tweets_merged.csv",dtype=str)
    direct = "/ben_shapiro"
    user_node_file = open("./filtered_ben_skewed.txt","r")
    nr = user_node_file.read()
    user_nodes = nr.split("\n")
    print(len(user_nodes))

    print(df_tweets.shape)
    sayac = 0

    for UserID in user_nodes:
    
        print(UserID)
        
        print(sayac)
        sayac += 1
        
        user_tweets = df_tweets[UserID][:]
        
        current_belief = 0
        
        scor_file = open(f"{direct}/ben_beliefs.txt","a")
        activity_file = open(f"{direct}/ben_activity.txt","a")

        inner_sayac = 0
        
        for tweet in user_tweets:
            
            print(inner_sayac)
            inner_sayac += 1
            
            try:
                assert isnan(tweet) is True
                print("Current belief NOT updated")
                
                activity_file.write("0")
                if tweet != list(user_tweets)[-1]:
                    activity_file.write(",")
                
            except:
                segmented_tweet = tweet.split("STOP")
                tweets_clean = clean_tweets(segmented_tweet)
                sentiment_scores = roberta_plus_sent_anal.eval_sentences_positive(tweets_clean)
                
                log_sentiment_scores = np.log(np.array(sentiment_scores)/(1-np.array(sentiment_scores)))
                current_belief = Average(log_sentiment_scores)
                
                print("Current belief updated")
                
                activity_file.write("1")
                if tweet != list(user_tweets)[-1]:
                    activity_file.write(",")
                


            print("Roberta Score: ",current_belief)
            score_string = str(current_belief)
            scor_file.write(score_string)
            
            if tweet != list(user_tweets)[-1]:
                scor_file.write(",")

        scor_file.write("\n")
        activity_file.write("\n")

        scor_file.close()
        activity_file.close()