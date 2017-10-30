#! /usr/bin/env python

import sys
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier

from sklearn.model_selection import cross_val_score


X = np.loadtxt("aa_w3_a3.dat")
Y = np.loadtxt("ss_a3.dat")

clf = svm.SVC()
scores = cross_val_score(clf, X, Y)
print scores.mean()

clf = AdaBoostClassifier(n_estimators=100)
scores = cross_val_score(clf, X, Y)
print scores.mean()

clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,max_depth=1, random_state=0)
scores = cross_val_score(clf, X, Y)
print scores.mean()

clf = SGDClassifier(loss="hinge", penalty="l2")
scores = cross_val_score(clf, X, Y)
print scores.mean()


clf = GaussianNB()
scores = cross_val_score(clf, X, Y)
print scores.mean()

clf1 = LogisticRegression(random_state=1)
clf2 = RandomForestClassifier(random_state=1)
clf3 = GaussianNB()
eclf = VotingClassifier(estimators=[('lr', clf1), ('rf', clf2), ('gnb', clf3)], voting='hard')


for clf, label in zip([clf1, clf2, clf3, eclf], ['Logistic Regression', 'Random Forest', 'naive Bayes', 'Ensemble']):
    scores = cross_val_score(clf, X, Y, cv=5, scoring='accuracy')
    print("Accuracy: %0.2f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), label))




