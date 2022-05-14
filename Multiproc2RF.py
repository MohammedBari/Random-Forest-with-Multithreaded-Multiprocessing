import time
import os
from random import random, choice
import string
from sklearn.datasets import load_iris
from sklearn import tree
from sklearn.tree import export_graphviz
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import export_text
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pydot as pyd
import multiprocessing
from multiprocessing import Process
from multiprocessing import pool

trainData = pd.read_csv('DataFiles/train.csv')
trainData.head()

testData = pd.read_csv('DataFiles/test.csv')
testData.head()


def runPredictions1():
    i = 1
    time.sleep(1)

    y_train = trainData["Exists"]
    features = ['bio_1', 'bio_2', 'bio_3', 'bio_4', 'bio_5', 'bio_6', 'bio_7', 'bio_8', 'bio_9', 'bio_10', 'bio_11',
                'bio_12', 'bio_13', 'bio_14', 'bio_15', 'bio_16', 'bio_17', 'bio_18', 'bio_19']
    X_train = pd.get_dummies(trainData[features])
    X_test = pd.get_dummies(testData[features])

    print('Training Features Shape:', X_train.shape)
    print('Training Labels Shape:', y_train.shape)
    print('Testing Features Shape:', X_test.shape)

    rf = RandomForestClassifier(n_estimators=1500, max_depth=50, random_state=1, n_jobs=4)
    rf.fit(X_train, y_train)
    prediction = rf.predict(X_test)

    # Print one tree to file
    # print('\n')
    # rftree = rf.estimators_[5]
    # export_graphviz(rftree, out_file='RFOutput/tree.dot', feature_names=features, rounded=True, precision=1)
    # (graph,) = pyd.graph_from_dot_file('RFOutput/tree.dot')
    # graph.write_png('RFOutput/tree.png')

    output = pd.DataFrame({'pointid': trainData.pointid, 'Exists': prediction})
    output.to_csv('RFOutput/outp.csv', index=False)
    print("outputted")

    # Variable importance
    importance = list(rf.feature_importances_)
    feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(features, importance)]
    feature_importances = sorted(feature_importances, key=lambda x: x[1], reverse=True)
    [print('Variable:  {:20} Importance: {}'.format(*pair)) for pair in feature_importances]


tic = time.time()
runPredictions1()
toc = time.time()
print('\nDone in {:.4f} seconds'.format(toc - tic))
