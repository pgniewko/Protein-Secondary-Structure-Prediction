#! /usr/bin/env python
# BioE 134, Fall 2017
# Author: Pawel Gniewek (pawel.gniewek@berkeley.edu)
# License: BSD
#
# Point to an input file (../data/db/aa_w5_a3.dat), and sec.str. classes file (../data/db/ss_a3.dat)
# Usage: ./simple_ml.py ../data/db/aa_w5_a3.dat ../data/db/ss_a3.dat


import sys
import itertools
import numpy as np
from scipy import interp
import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix

from sklearn.dummy import DummyClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier
from sklearn.svm import SVC

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold


def plot_ROC_for_clf(clf, X, Y, cv_fold=5):
    """
    """
    
    n_classes = len(set(Y))
    Y_bin = []
    for cl in set(Y):
        arr_ = []
        for el in Y:
            if el == cl:
                arr_.append(1)
            else:
                arr_.append(0)
        Y_bin.append( np.array( arr_ ) )

    cv = StratifiedKFold(n_splits=cv_fold)

    for class_idx in range(n_classes):
        class_array = np.array([])
        class_probs = np.array([])

        for train, test in cv.split(X, Y_bin[class_idx]):
            probas_ = clf.fit(X[train], Y_bin[class_idx][train]).predict_proba(X[test])

            class_array = np.concatenate( ( class_array, Y_bin[class_idx][test]), axis=0)           
            class_probs = np.concatenate( ( class_probs, probas_[:,1]), axis=0) 

        fpr, tpr, thresholds = roc_curve(class_array, class_probs)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, alpha=0.6, label='ROC for class %d (AUC = %0.2f)' % (class_idx, roc_auc))

    
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Random', alpha=.8)   

    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate', fontsize=20)
    plt.ylabel('True Positive Rate', fontsize=20)
    plt.title('Receiver operating characteristic', fontsize=25)
    plt.legend(loc="lower right", fontsize=15)


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

#    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


if __name__ == "__main__":

    ## Read the data
    #X = np.loadtxt("../data/db/aa_w10_a3.dat")
    #Y = np.loadtxt("../data/db/ss_a3.dat")
    X = np.loadtxt(sys.argv[1])
    Y = np.loadtxt(sys.argv[2])


    ###  (Random) Baseline
    clf = DummyClassifier(strategy="stratified")
    scores = cross_val_score(clf, X, Y, cv=5, scoring='accuracy')
    print( "Accuracy: %0.2f (+/- %0.2f) [ %s ]" % (scores.mean(), scores.std(), "Dummy (stratified)") )
    
    clf = DummyClassifier(strategy="stratified")
    clf.fit(X, Y)
    Y_pred = clf.predict(X)
    print( "Accuracy: %0.2f (+/- %0.2f) [ %s ]" % (np.mean(Y_pred == Y), np.std(Y_pred == Y), "DummyClassifier") )

    clf = DummyClassifier(strategy="uniform")
    scores = cross_val_score(clf, X, Y, cv=5, scoring='accuracy')
    print( "Accuracy: %0.2f (+/- %0.2f) [ %s ]" % (scores.mean(), scores.std(), "Dummy (uniform)") )


    ## Decision Tree - overfitting
    clf = DecisionTreeClassifier()
    clf.fit(X, Y)
    Y_pred = clf.predict(X)
    print( "Accuracy: %0.2f (+/- %0.2f) [ %s ]" % (np.mean(Y_pred == Y), np.std(Y_pred == Y), "DecisionTreeClassifier*") )

    # Decision Tree - cross validation
    clf = DecisionTreeClassifier()
    scores = cross_val_score(clf, X, Y, cv=5, scoring='accuracy')
    print("Accuracy: %0.2f (+/- %0.2f) [ %s ]" % (scores.mean(), scores.std(),"DecisionTreeClassifier: CV=5, max_depth=None"))

    # RandomForestClassifier
    clf = RandomForestClassifier(random_state=1, max_depth=50)
    scores = cross_val_score(clf, X, Y, cv=5, scoring='accuracy')
    print("Accuracy: %0.2f (+/- %0.2f) [ %s ]" % (scores.mean(), scores.std(),"RandomForestClassifier: CV=5, max_depth=50"))
    
    # RandomForestClassifier
    clf = RandomForestClassifier(random_state=1, max_depth=50, criterion='entropy')
    scores = cross_val_score(clf, X, Y, cv=5, scoring='accuracy')
    print("Accuracy: %0.2f (+/- %0.2f) [ %s ]" % (scores.mean(), scores.std(),"RandomForestClassifier,entropy: CV=5, max_depth=50"))

    clf1 = DecisionTreeClassifier(max_depth=5)
    clf2 = RandomForestClassifier(random_state=1, max_depth=10)
    clf3 = GaussianNB()
    clf4 = LogisticRegression()
    eclf = VotingClassifier(estimators=[('dt', clf1), ('rf', clf2), ('mnb', clf3), ('lr', clf4)], voting='hard')

    for clf, label in zip([clf1, clf2, clf3, clf4,  eclf], \
            ['Decision Tree', 'Random Forest', 'Naive Bayes', 'Logistic Regression', 'Ensemble']):
        scores = cross_val_score(clf, X, Y, cv=5, scoring='accuracy')
        print("Accuracy: %0.2f (+/- %0.2f) [ %s ]" % (scores.mean(), scores.std(), label))


    plot_flag = True
    plot_flag = False
    if plot_flag:
        plt.figure( figsize=(7,7) )
        plot_ROC_for_clf( RandomForestClassifier(random_state=1, max_depth=10), X, Y, cv_fold=5)

        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=0)
        Y_pred = eclf.fit(X_train, Y_train).predict(X_test)

# Compute confusion matrix
        cnf_matrix = confusion_matrix(Y_test, Y_pred)
        np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
        plt.figure()
        plot_confusion_matrix(cnf_matrix, classes=['H','E','C'],\
                             title='Confusion matrix, without normalization')

# Plot normalized confusion matrix
        plt.figure()
        plot_confusion_matrix(cnf_matrix, classes=['H','E','C'], normalize=True,
                      title='Normalized confusion matrix')

        plt.show()

    sys.exit(1)

    clf = SVC(kernel="linear", C=0.025)
    scores = cross_val_score(clf, X, Y, cv=5)
    print("Accuracy: %0.2f (+/- %0.2f) [ %s ]" % (scores.mean(), scores.std(),"SVC(kernel=linear, C=0.025): CV=5"))

    clf = SVC(kernel="rbf", gamma=2, C=1)
    scores = cross_val_score(clf, X, Y, cv=5)
    print("Accuracy: %0.2f (+/- %0.2f) [ %s ]" % (scores.mean(), scores.std(),"SVC(gamma=2, C=1): CV=5"))



