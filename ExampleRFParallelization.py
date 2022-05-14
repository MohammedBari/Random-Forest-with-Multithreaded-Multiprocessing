from matplotlib import pyplot
from numpy import arange
from numpy import mean
from numpy import std
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
import _thread
import time
import os
from random import random, choice
import string
from numpy import mean
from numpy import std
from sklearn.datasets import load_iris
from sklearn import tree
from sklearn.tree import export_graphviz
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import export_text
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pydot as pyd
import threading
import logging
from queue import Queue
import matplotlib.pyplot as plt

trainData = pd.read_csv('DataFiles/train.csv')
trainData.head()

testData = pd.read_csv('DataFiles/test.csv')
testData.head()

feature = pd.read_csv('DataFiles/train.csv')
feature.head()

feature_list = list(feature.columns)
features = np.array(feature)

def get_dataset1():
    # yt = trainData["Exists"]
    # featuresl = ['bio_1', 'bio_2', 'bio_3', 'bio_4', 'bio_5', 'bio_6', 'bio_7', 'bio_8', 'bio_9', 'bio_10', 'bio_11',
    #              'bio_12', 'bio_13', 'bio_14', 'bio_15', 'bio_16', 'bio_17', 'bio_18', 'bio_19']
    # Xt = pd.get_dummies(trainData[featuresl])
    # Xte = pd.get_dummies(testData[featuresl])
#
    # X_train, X_test, y_train, y_test = train_test_split(Xt, yt, test_size=0.3)
    # X_train2, X_test2, y_train2, y_test2 = train_test_split(X_train, y_train, test_size=0.3)
    # return X_train, y_train
    X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, n_redundant=5, random_state=3)
    return X, y


def get_dataset2():
    # yt = trainData["Exists"]
    # featuresl = ['bio_1', 'bio_2', 'bio_3', 'bio_4', 'bio_5', 'bio_6', 'bio_7', 'bio_8', 'bio_9', 'bio_10', 'bio_11',
    #              'bio_12', 'bio_13', 'bio_14', 'bio_15', 'bio_16', 'bio_17', 'bio_18', 'bio_19']
    # Xt = pd.get_dummies(trainData[featuresl])
    # Xte = pd.get_dummies(testData[featuresl])
    # X_train2, X_test2, y_train2, y_test2 = train_test_split(Xt, yt, test_size=0.3)
    # return X_train2, y_train2
    X2, y2 = make_classification(n_samples=1000, n_features=20, n_informative=15, n_redundant=5, random_state=3)
    return X2, y2


class threads(threading.Thread):
    def __init__(self, wait):
        threading.Thread.__init__(self)
        self.wait = wait

    def run(self):
        i = 0
        runPredictions(self.wait)
        runPredictions2(self.wait)
        print("%s finished executing" % self.wait)

# get a list of models to evaluate
def get_models():
    models = dict()
    # explore ratios from 10% to 100% in 10% increments
    for i in arange(0.1, 1.1, 0.1):
        key = '%.1f' % i
        # set max_samples=None to use 100%
        if i == 1.0:
            i = None
        models[key] = RandomForestClassifier(max_samples=i)
    return models


# evaluate a given model using cross-validation
def evaluate_model(model, X, y):
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
    return scores

def runPredictions(wait):
    time.sleep(wait)
    # define dataset
    X, y = get_dataset1()
    models = get_models()

    results, names, results2 = list(), list(), list()

    for name, model in models.items():
        scores = evaluate_model(model, X, y)
        results.append(scores)
        names.append(name)
        print('>%s %.3f (%.3f)' % (name, mean(scores), std(scores)))
    pyplot.boxplot(results, labels=names, showmeans=True)
    pyplot.show()

def runPredictions2(wait):
    time.sleep(wait)
    # define dataset
    X2, y2 = get_dataset2()
    models = get_models()

    results, names, results2 = list(), list(), list()

    for name, model in models.items():
        scores = evaluate_model(model, X2, y2)
        results.append(scores)
        names.append(name)
        print('>%s %.3f (%.3f)' % (name, mean(scores), std(scores)))
    pyplot.boxplot(results, labels=names, showmeans=True)
    pyplot.show()

if __name__ == "__main__":
    tic = time.time()

    thread1 = threads(1)
    thread2 = threads(2)
    # thread3 = threads(3, X_train3, X_test3, y_train3, y_test3)

    thread1.start()
    thread2.start()
    # thread3.start()
    thread1.join()
    thread2.join()
    # thread3.join()

    toc = time.time()
    print('\nDone in {:.4f} seconds'.format(toc - tic))