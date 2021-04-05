import json
import numpy as np
from typing import Dict, List

ALM = 0
BALTIMORE = 1
BLM = 2
DAVIDSON = 3
ELECTION = 4
METOO = 5
SANDY = 6

NUM_CORPUS = 7
factors = ["subversion", "authority", "cheating", "fairness", "harm", "care", "betrayal", \
    "loyalty", "purity", "degradation", "non-moral"]

def get_data():
    f = open("./MFTC_V4.json", "r")
    data = json.load(f)
    alm_tweets = vectorize_tweets(data[ALM]["Tweets"])
    baltimore_tweets = vectorize_tweets(data[BALTIMORE]["Tweets"])
    blm_tweets = vectorize_tweets(data[BLM]["Tweets"])
    davidson_tweets = vectorize_tweets(data[DAVIDSON]["Tweets"])
    election_tweets = vectorize_tweets(data[ELECTION]["Tweets"])
    metoo_tweets = vectorize_tweets(data[METOO]["Tweets"])
    sandy_tweets = vectorize_tweets(data[SANDY]["Tweets"])

    f.close()
    return alm_tweets, baltimore_tweets, blm_tweets, davidson_tweets, election_tweets, metoo_tweets, sandy_tweets

def vectorize_tweets(tweets: List[Dict]):
    v_tweets = np.zeros((len(tweets), len(factors)))

    for i in range(len(tweets)):
        annotations = []

        # Collect all unique annotations applied to a single tweet.
        for j in range(len(tweets[i]["annotations"])):
            annotations = list(set(annotations + tweets[i]["annotations"][j]["annotation"].split(",")))
       
        # Update vector representation of tweet according to its annotations.
        for j in range(len(factors)):
            if factors[j] in annotations:
                v_tweets[i][j] = 1

    return v_tweets



if __name__ == "__main__":
    alm_tweets, baltimore_tweets, blm_tweets, davidson_tweets, election_tweets, metoo_tweets, sandy_tweets = get_data()

