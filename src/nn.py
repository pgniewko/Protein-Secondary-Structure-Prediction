#! /usr/bin/env python
# BioE 134, Fall 2017
# Author: Pawel Gniewek (pawel.gniewek@berkeley.edu)
# License: BSD
#
# Point to an input file (../data/db/aa_w5_a3.dat), and sec.str. classes file (../data/db/ss_a3.dat)
# Usage: ./optimize_clf.py ../data/db/aa_w5_a3.dat ../data/db/ss_a3.dat

from __future__ import print_function


import sys
import numpy as np

from sklearn.neural_network import MLPClassifier

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report


if __name__ == "__main__":

# Read the data
    X = np.loadtxt(sys.argv[1])
    Y = np.loadtxt(sys.argv[2])

    clf = MLPClassifier(activation='relu',solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(25,11,3), random_state=1, verbose=10)
    scores = cross_val_score(clf, X, Y, cv=5, scoring='accuracy')
    print("Accuracy: %0.2f (+/- %0.2f) [ %s ]" % (scores.mean(), scores.std(), "MLPClassifier: CV=5"))

    sys.exit(1)
# Set the parameters by cross-validation
    parameters = [ {'penalty':['l2'], \
    'C':[0.25, 0.5, 0.75, 1.0, 1.5, 2.0], \
    'solver':['newton-cg', 'lbfgs', 'sag'], \
    'intercept_scaling':[0.5, 1.0, 2.5, 5.0] },\
    {'penalty':['l1'], \
     'C':[0.25, 0.5, 0.75, 1.0, 1.5, 2.0], \
     'solver':['liblinear'], \
     'intercept_scaling':[0.5, 1.0, 2.5, 5.0] } ]


# Split the dataset in two equal parts
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.5, random_state=0)

    scores = ['precision', 'recall']

    for score in scores:
        print("# Tuning hyper-parameters for %s" % score)
        print()

        clf = GridSearchCV( MLPClassifier() , parameters,\
                            cv=2, scoring='%s_macro' % score)

        clf.fit(X_train, y_train)

        print("Best parameters set found on development set:")
        print()
        print(clf.best_params_)
        print()
        print("Grid scores on development set:")
        print()
        means = clf.cv_results_['mean_test_score']
        stds = clf.cv_results_['std_test_score']
        for mean, std, params in zip(means, stds, clf.cv_results_['params']):
            print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))
        print()

        print("Detailed classification report:")
        print()
        print("The model is trained on the full development set.")
        print("The scores are computed on the full evaluation set.")
        print()
        y_true, y_pred = y_test, clf.predict(X_test)
        print(classification_report(y_true, y_pred))
        print()


