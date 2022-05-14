import _thread
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

yt = trainData["Exists"]
featuresl = ['bio_1', 'bio_2', 'bio_3', 'bio_4', 'bio_5', 'bio_6', 'bio_7', 'bio_8', 'bio_9', 'bio_10', 'bio_11',
             'bio_12', 'bio_13', 'bio_14', 'bio_15', 'bio_16', 'bio_17', 'bio_18', 'bio_19']
Xt = pd.get_dummies(trainData[featuresl])
Xte = pd.get_dummies(testData[featuresl])

X_train, X_test, y_train, y_test = train_test_split(Xt, yt, test_size=0.3)
X_train2, X_test2, y_train2, y_test2 = train_test_split(X_train, y_train, test_size=0.3)
X_train3, X_test3, y_train3, y_test3 = train_test_split(X_train2, y_train2, test_size=0.3)
# X_train4, X_test4, y_train4, y_test4 = train_test_split(X_train, y_train, test_size=0.3)

print('Training Features Shape:', X_train.shape)
print('Training Labels Shape:', y_train.shape)
print('Testing Features Shape:', X_test.shape)
global preds


class threads(threading.Thread):
    def __init__(self, wait, X_train, X_test, y_train, y_test):
        threading.Thread.__init__(self)
        self.y_test = y_test
        self.y_train = y_train
        self.X_test = X_test
        self.X_train = X_train
        self.wait = wait


    def run(self):
        i = 0
        runPredictions(self.wait, self.X_train, self.X_test, self.y_train, self.y_test, featuresl)
        print("%s finished executing" % self.wait)


def runPredictions(wait, Xtr, Xtes, ytr, yte, feats):
    i = 0
    print('\n')
    while i < 1:
        print('\n')
        time.sleep(wait)

        rf = RandomForestClassifier(n_estimators=1500, random_state=1)
        # rf = RandomForestClassifier(n_estimators=1500, max_depth=50, random_state=1, n_jobs=4)
        rf.fit(Xtr, ytr)
        prediction = rf.predict(Xtes)
        prediction = pd.DataFrame(prediction)
        # df_out["Prediction"] = prediction.reset_index()[0]
        yte['Exists'] = prediction
        df_out = pd.merge(yt, yte[['Exists']], how='left', left_index=True, right_index=True)

        output = pd.DataFrame(df_out)
        output.to_csv('ThreadOutput/outp%s.csv' % i, index=False)
        print("outputted %s" % i)
        print(prediction)

        output = pd.DataFrame({'Exists': yte})
        temp = i+ 1
        output.to_csv('ThreadOutput/outp%s.csv' % temp , index=False)
        print("outputted %s" % temp )
        print(prediction)

        # Variable importance
        importance = list(rf.feature_importances_)
        feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(feats, importance)]
        feature_importances = sorted(feature_importances, key=lambda x: x[1], reverse=True)
        [print('Variable:  {:20} Importance: {}'.format(*pair)) for pair in feature_importances]
        i = i + 1


if __name__ == "__main__":
    tic = time.time()

    thread1 = threads(1, X_train, X_test, y_train, y_test)
    thread2 = threads(2, X_train2, X_test2, y_train2, y_test2)
    thread3 = threads(3, X_train3, X_test3, y_train3, y_test3)
    # thread4 = threads(4, X_train4, X_test4, y_train4, y_test4)

    thread1.start()
    thread2.start()
    thread3.start()
    # thread4.start()
    thread1.join()
    thread2.join()
    thread3.join()
    # thread4.join()

    toc = time.time()
    print('\nDone in {:.4f} seconds'.format(toc - tic))
