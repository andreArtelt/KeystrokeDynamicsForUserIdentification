#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function
import sys
import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split, KFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.tree import export_graphviz
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, cohen_kappa_score, confusion_matrix, roc_auc_score
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from analyze_data import perform_pca, perform_lda, perform_pca_sparse
from cluster_user import cluster
from sklearn_lvq import GrmlvqModel


# Important params for evaluation
NUM_EVAL_RUNS = 25
NUM_KFOLD_SPLITS = 2
TEST_SIZE = 0.5
USE_KFOLD = True
USE_CLUSTERS = False
NUM_CLUSTERS = 5
MOUSE_ONLY = False


# Only set one of them True
USE_PCA = False
USE_LDA = False


cur_model = None


def get_model():
    if cur_model == "randomforest":
        return RandomForestClassifier(n_estimators=400, max_depth=6, class_weight="balanced")
    elif cur_model == "adaboost":
        return AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=7), n_estimators=400, learning_rate=0.01)
    elif cur_model == "decisiontree":
        return DecisionTreeClassifier(max_depth=7)
    elif cur_model == "mlp":
        return MLPClassifier(hidden_layer_sizes=(100,), activation="logistic", solver="lbfgs")
    elif cur_model == "grmlvq":
        return GrmlvqModel(prototypes_per_class=8)
    elif cur_model == "logreg":
        return LogisticRegression(C=0.01, penalty='l2')
    elif cur_model == "svm":
        return SVC(C=400.0, kernel='rbf', gamma=0.05)
    elif cur_model == "knn":
        return KNeighborsClassifier(n_neighbors=3)
    elif cur_model == "naivebayes":
        return GaussianNB()
    else:
        raise "Invalid model!"


def analyse_feature_importance(model, feature_usage):
    usage = model.feature_importances_
    for i in range(len(usage)):
        if usage[i] > 0.001:
            feature_usage[i] += 1
    return feature_usage


def do_work(X_, y_, accuracy_scores, recall_scores, precision_scores, f1_scores, kappa_scores, confusion_matricies, roc_auc_scores):
    feature_usage_ = np.zeros(X_.shape[1])

    # Shuffle data
    X_, y_ = shuffle(X_, y_)

    # Resampling (because data is highly unbalanced)
    X_, y_ = RandomUnderSampler().fit_sample(X_, y_)

    def fit_eval_model(X_train, X_test, y_train, y_test):
        # Preprocessing
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        if USE_PCA:
            X_train, X_test = perform_pca(X_train, X_test, n_components=15)

        if USE_LDA:
            X_train, X_test = perform_lda(X_train, X_test, y_train, n_components=15)

        # Fit model (order = ranking of models)
        model = get_model()
        model.fit(X_train, y_train)

        y_test_pred = model.predict(X_test)

        if isinstance(model, DecisionTreeClassifier):    # Visualize the decision tree
            export_graphviz(model, "tree.dot", feature_names=None)
            #dot -Tpng tree.dot -o tree.png

        # Feature importance
        feature_usage_local = np.zeros(X_.shape[1])
        if hasattr(model, "feature_importances_"):
            feature_usage_local = analyse_feature_importance(model, feature_usage_local)

        # Evaluate model
        accuracy_scores.append(accuracy_score(y_test, y_test_pred))
        recall_scores.append(recall_score(y_test, y_test_pred))
        precision_scores.append(precision_score(y_test, y_test_pred))
        f1_scores.append(f1_score(y_test, y_test_pred))
        kappa_scores.append(cohen_kappa_score(y_test, y_test_pred))
        confusion_matricies.append(confusion_matrix(y_test, y_test_pred))
        roc_auc_scores.append(roc_auc_score(y_test, y_test_pred))
        return feature_usage_local

    # Split data by either using kfold or a simple train-test split
    if USE_KFOLD == True:
        kf = KFold(n_splits=NUM_KFOLD_SPLITS, shuffle=True)
        for train_index, test_index in kf.split(X_):
            X_train, X_test = X_[train_index], X_[test_index]
            y_train, y_test = y_[train_index], y_[test_index]

            feature_usage_ += fit_eval_model(X_train, X_test, y_train, y_test)
    else:
        X_train, X_test, y_train, y_test = train_test_split(X_, y_, test_size=TEST_SIZE, shuffle=True)
        feature_usage_ += fit_eval_model(X_train, X_test, y_train, y_test)

    return feature_usage_

def classify(X, y):
    # Select features
    if MOUSE_ONLY is False:
        features = None
        if cur_model in ["logreg", "svm"]:
            features = [2021, 1993, 1992, 2001, 2003, 2002, 1998, 1995, 108, 1996, 110, 1997, 2008, 1999, 111]
        else:
            features = [2118, 2121, 2021, 1999, 3924, 3913, 2233, 1993, 2002, 1992, 2234, 2120, 2103, 2204, 2119, 2186, 2106, 3917, 2184]

        if features is not None:
            X = X[:, features]

    # Cluster/Group user
    if USE_CLUSTERS:
        X_new = []
        for label in range(len(np.unique(y))):
            X_new.append(np.mean(X[y == label,:], axis=0))
        X_new = np.array(X_new)

        y_new_clusters = cluster(X_new, n_clusters=NUM_CLUSTERS)

        y_new = np.copy(y)
        for i in range(X.shape[0]):
            y_new[i] = y_new_clusters[y[i]]

        y = y_new

    # Array for results (metric-scores per user)
    user_metrics = []

    # Placeholder for feature usage
    feature_usage = np.zeros(X.shape[1])

    # Fit a classifier for each user
    num_users = len(np.unique(y))
    for label in range(num_users):
        print("User: {0}".format(label))

        accuracy_scores = []
        precision_scores = []
        recall_scores = []
        f1_scores = []
        kappa_scores = []
        confusion_matricies = []
        roc_auc_scores = []

        # One vs. Rest
        #y_ = np.array(y == label).astype(np.int) # Predict if the user is itself
        y_ = np.array(y != label).astype(np.int)  # Predict if it is somebody else
        X_ = X

        print("Number of samples: {0}".format(X_.shape[0]))

        # Run several rounds of splitting, fitting and evaluation
        feature_usage += np.sum(np.vstack([do_work(X_, y_, accuracy_scores, precision_scores, recall_scores,
                                                           f1_scores, kappa_scores, confusion_matricies, roc_auc_scores)
                                            for _ in range(NUM_EVAL_RUNS)]), axis=0)

        # Save results
        user_metrics.append([accuracy_scores, precision_scores, recall_scores, f1_scores, kappa_scores, roc_auc_scores])

        # Compute overall statistics
        print("F1-score:\n{0}\nMean={1} Var={2}\n".format(str(f1_scores), np.mean(f1_scores), np.var(f1_scores)))
        print("ROC-AUC-score:\n{0}\nMean={1} Var={2}\n".format(str(roc_auc_scores), np.mean(roc_auc_scores),
                                                                np.var(roc_auc_scores)))

        if label == num_users - 1:
            print("Feature usage:")
            print(feature_usage)
    
    user_metrics = np.array(user_metrics)
    return user_metrics


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: <data.npz> <scores_out.npz> <model> <n_cluster=-1>")
    else:
        f_in = sys.argv[1]
        scores_f_out = sys.argv[2]
        cur_model = sys.argv[3]
        if len(sys.argv) > 4:
            USE_CLUSTERS = True
            NUM_CLUSTERS = int(sys.argv[4])

        # Load data
        data = np.load(f_in)
        X = data["X"]
        y = data["y"]

        if MOUSE_ONLY:
            a = X[:, 1992:2028]
            b = X[:, 3912:]
            X = np.concatenate((a, b), axis=1)

        results = classify(X, y)
        user_metrics = results

        if scores_f_out is not None:
            np.savez(scores_f_out, X=user_metrics)
