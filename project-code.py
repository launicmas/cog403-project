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
labels = {ALM: 0, BALTIMORE: 0.75, BLM: 1, DAVIDSON: 0.25, ELECTION: 0.5, METOO: 0.75, SANDY: 0.5}

def get_data():
    f = open("./MFTC_V4.json", "r")
    data = json.load(f)
    alm_X, alm_Y = vectorize_tweets(data[ALM]["Tweets"], ALM)
    baltimore_X, baltimore_Y = vectorize_tweets(data[BALTIMORE]["Tweets"], BALTIMORE)
    blm_X, blm_Y = vectorize_tweets(data[BLM]["Tweets"], BLM)
    davidson_X, davidson_Y = vectorize_tweets(data[DAVIDSON]["Tweets"], DAVIDSON)
    election_X, election_Y = vectorize_tweets(data[ELECTION]["Tweets"], ELECTION)
    metoo_X, metoo_Y = vectorize_tweets(data[METOO]["Tweets"], METOO)
    sandy_X, sandy_Y = vectorize_tweets(data[SANDY]["Tweets"], SANDY)

    X = np.concatenate((alm_X, baltimore_X, blm_X, davidson_X, election_X, metoo_X, sandy_X), axis=0)
    Y = np.concatenate((alm_Y, baltimore_Y, blm_Y, davidson_Y, election_Y, metoo_Y, sandy_Y))
    f.close()
    
    return X, Y

def vectorize_tweets(tweets: List[Dict], label: int):
    X = np.zeros((len(tweets), len(factors)))
    Y = np.full((len(tweets)), labels[label])

    for i in range(len(tweets)):
        annotations = []

        # Collect all unique annotations applied to a single tweet.
        for j in range(len(tweets[i]["annotations"])):
            annotations = list(set(annotations + tweets[i]["annotations"][j]["annotation"].split(",")))
       
        # Update vector representation of tweet according to its annotations.
        for j in range(len(factors)):
            if factors[j] in annotations:
                X[i][j] = 1

    return X, Y



if __name__ == "__main__":
    X, Y = get_data()
    print(X.shape, Y.shape)
