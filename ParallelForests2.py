# Mohammed Bari

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

yt = trainData["Exists"]
featuresl = ['bio_1', 'bio_2', 'bio_3', 'bio_4', 'bio_5', 'bio_6', 'bio_7', 'bio_8', 'bio_9', 'bio_10', 'bio_11',
             'bio_12', 'bio_13', 'bio_14', 'bio_15', 'bio_16', 'bio_17', 'bio_18', 'bio_19']
Xt = pd.get_dummies(trainData[featuresl])
Xte = pd.get_dummies(testData[featuresl])

X_train, X_test, y_train, y_test = train_test_split(Xt, yt, test_size=0.3)
X_train2, X_test2, y_train2, y_test2 = train_test_split(X_train, y_train, test_size=0.3)
# X_train3, X_test3, y_train3, y_test3 = train_test_split(X_train2, y_train2, test_size=0.3)
# X_train4, X_test4, y_train4, y_test4 = train_test_split(X_train3, y_train3, test_size=0.3)

print('Training Features Shape:', X_train.shape)
print('Training Labels Shape:', y_train.shape)
print('Testing Features Shape:', X_test.shape)


def runPredictions1(wait, Xtr, Xtes, ytr, yte, feats):
    i = 0
    print('\n')
    while i < 1:
        print('\n')
        time.sleep(wait)

        rf = RandomForestClassifier(n_estimators=1500, random_state=1)
        rf.fit(Xtr, ytr)
        prediction = rf.predict(Xtes)

        print('Predicted Class: %d' % prediction[0])

        # cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
        # n_scores = cross_val_score(rf.fit(Xtr, ytr), X_test, y_test, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')
        # print('Accuracy: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))

        prediction = pd.DataFrame(prediction)
        yte['Exists'] = prediction
        df_out = pd.merge(yt, yte[['Exists']], how='left', left_index=True, right_index=True)

        output = pd.DataFrame(df_out)
        output.to_csv('ParallelOut/outp%s.csv' % i, index=False)
        print("outputted %s" % i)
        print(prediction)

        output = pd.DataFrame({'Exists': yte})
        temp = i+ 1
        output.to_csv('ParallelOut/outp%s.csv' % temp , index=False)
        print("outputted %s" % temp )
        print(prediction)

        # Variable importance
        importance = list(rf.feature_importances_)
        feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(feats, importance)]
        feature_importances = sorted(feature_importances, key=lambda x: x[1], reverse=True)
        [print('Variable:  {:20} Importance: {}'.format(*pair)) for pair in feature_importances]
        i = i + 1


def runPredictions2():
    i = 1
    time.sleep(1)
    test = pd.read_csv('DataFiles/test.csv')
    test.head()
    features = pd.read_csv('DataFiles/train.csv')
    features.head()

    # Assigning, removing, saving labels (target)
    labels = np.array(features['Exists'])
    features = features.drop('Exists', axis=1)
    features = features.drop('pointid', axis=1)
    features = features.drop('index', axis=1)
    feature_list = list(features.columns)
    features = np.array(features)

    # Create training, test sets
    X_train, X_test, y_train, y_test = \
        train_test_split(features, labels, random_state=0)
    print('Training Features Shape:', X_train.shape)
    print('Training Labels Shape:', y_train.shape)
    print('Testing Features Shape:', X_test.shape)
    print('Testing Labels Shape:', y_test.shape)

    # Running model, 1000 trees
    rf = RandomForestClassifier(n_estimators=1500)
    rf.fit(X_train, y_train)
    predictions = rf.predict(X_test)
    print(predictions)

    # Print one tree to file
    # print('\n')
    # rftree = rf.estimators_[5]
    # export_graphviz(rftree, out_file='RFOutput/tree2.dot', feature_names=feature_list, rounded=True, precision=1)
    # (graph,) = pyd.graph_from_dot_file('RFOutput/tree2.dot')
    # graph.write_png('RFOutput/tree2.png')

    output = pd.DataFrame({'Exists': predictions})
    output.to_csv('RFOutput/outp2.csv', index=False)
    print("outputted")

    # Variable importance
    importance = list(rf.feature_importances_)
    feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(feature_list, importance)]
    feature_importances = sorted(feature_importances, key=lambda x: x[1], reverse=True)
    [print('Variable:  {:20} Importance: {}'.format(*pair)) for pair in feature_importances]


# Start time, and assign processes to the prediction function
tic = time.time()
pr1 = multiprocessing.Process(target=runPredictions1(1, X_train, X_test, y_train, y_test, featuresl))
pr2 = multiprocessing.Process(target=runPredictions1(1, X_train2, X_test2, y_train2, y_test2, featuresl))
# pr3 = multiprocessing.Process(target=runPredictions1(1, X_train3, X_test3, y_train3, y_test3, featuresl))
# pr4 = multiprocessing.Process(target=runPredictions1(1, X_train4, X_test4, y_train4, y_test4, featuresl))

if __name__ == '__main__':
    pr1.start()
    pr1.join()
    pr2.start()
    pr2.join()
    # pr3.start()
    # pr3.join()
    # pr4.start()
    # pr4.join()

toc = time.time()
print('\nDone in {:.4f} seconds'.format(toc - tic))
