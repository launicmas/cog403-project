import json
from typing import Dict, List
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV, train_test_split

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
labels = {ALM: 0, BALTIMORE: 1, BLM: 1, DAVIDSON: 0, ELECTION: 0, METOO: 1, SANDY: 1}

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

    f.close()

    return alm_X, alm_Y, baltimore_X, baltimore_Y, blm_X, blm_Y, davidson_X, davidson_Y, \
        election_X, election_Y, metoo_X, metoo_Y, sandy_X, sandy_Y


def vectorize_tweets(tweets: List[Dict], label: int):
    X = np.zeros((len(tweets), len(factors)))
    Y = np.full((len(tweets)), labels[label])

    for i in range(len(tweets)):
        annotations = []

        # Collect all annotations applied to a single tweet.
        for j in range(len(tweets[i]["annotations"])):
            annotations = annotations + tweets[i]["annotations"][j]["annotation"].split(",")
       
        # Update vector representation of tweet according to its annotations.
        for j in range(len(factors)):
            for k in range(len(annotations)):
                if factors[j] == annotations[k]:
                    X[i][j] += 1

    return X, Y


def run_log_regression(X, Y):
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33, random_state=42)

    # Find optimal parameters for logistic regression on dataset
    parameters = {"penalty": ["l1", "l2"], "C": np.logspace(-4, 4, 20), "solver": ["liblinear"]}
    log_reg = LogisticRegression(class_weight="balanced")
    clf = GridSearchCV(log_reg, parameters)
    clf.fit(X_train, Y_train)

    conf_m = confusion_matrix(Y_test, clf.predict(X_test))

    return clf.score(X_train, Y_train), clf.score(X_test, Y_test), conf_m


if __name__ == "__main__":
    alm_X, alm_Y, baltimore_X, baltimore_Y, blm_X, blm_Y, davidson_X, davidson_Y, \
        election_X, election_Y, metoo_X, metoo_Y, sandy_X, sandy_Y = get_data()

    # Group 1: All data
    all_X = np.concatenate((alm_X, baltimore_X, blm_X, davidson_X, election_X, metoo_X, sandy_X), axis=0)
    all_Y = np.concatenate((alm_Y, baltimore_Y, blm_Y, davidson_Y, election_Y, metoo_Y, sandy_Y))
    train_acc, test_acc, conf_m = run_log_regression(all_X, all_Y)
    print("Group 1 training accuracy: ", train_acc)
    print("Group 1 testing accuracy: ", test_acc)
    print("Group 1 Confusion Matrix:\n", conf_m)

    # Group 2: ALM and BLM only
    X = np.concatenate((alm_X, blm_X), axis=0)
    Y = np.concatenate((alm_Y, blm_Y))
    train_acc, test_acc, conf_m = run_log_regression(X, Y)
    print("Group 2 training accuracy: ", train_acc)
    print("Group 2 testing accuracy: ", test_acc)
    print("Group 2 Confusion Matrix:\n", conf_m)

    # Group 3: All data except Sandy and Elections
    X = np.concatenate((alm_X, baltimore_X, blm_X, davidson_X, metoo_X), axis=0)
    Y = np.concatenate((alm_Y, baltimore_Y, blm_Y, davidson_Y, metoo_Y))
    train_acc, test_acc, conf_m = run_log_regression(X, Y)
    print("Group 3 training accuracy: ", train_acc)
    print("Group 3 testing accuracy: ", test_acc)
    print("Group 3 Confusion Matrix:\n", conf_m)

    # Group 4: Permute Election and Sandy labels
    sandy_0 = np.zeros(sandy_Y.shape)
    election_1 = np.full(election_Y.shape, 1)

    Y = np.concatenate((alm_Y, baltimore_Y, blm_Y, davidson_Y, election_1, metoo_Y, sandy_Y))
    train_acc, test_acc, conf_m = run_log_regression(all_X, Y)
    print("Group 4 training accuracy (E=1, S=1): ", train_acc)
    print("Group 4 testing accuracy (E=1, S=1): ", test_acc)
    print("Group 4 Confusion Matrix (E=1, S=1):\n", conf_m)

    Y = np.concatenate((alm_Y, baltimore_Y, blm_Y, davidson_Y, election_Y, metoo_Y, sandy_0))
    train_acc, test_acc, conf_m = run_log_regression(all_X, Y)
    print("Group 4 training accuracy (E=0, S=0): ", train_acc)
    print("Group 4 testing accuracy (E=0, S=0): ", test_acc)
    print("Group 4 Confusion Matrix (E=0, S=0):\n", conf_m)

    Y = np.concatenate((alm_Y, baltimore_Y, blm_Y, davidson_Y, election_1, metoo_Y, sandy_0))
    train_acc, test_acc, conf_m = run_log_regression(all_X, Y)
    print("Group 4 training accuracy (E=1, S=0): ", train_acc)
    print("Group 4 testing accuracy (E=1, S=0): ", test_acc)
    print("Group 4 Confusion Matrix (E=1, S=0):\n", conf_m)


    

    
   


