#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function
import sys
import numpy as np
from itertools import repeat

from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split, KFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, cohen_kappa_score, \
    confusion_matrix, roc_auc_score
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from sklearn_lvq import GrmlvqModel

from analyze_data import perform_pca, perform_lda


# Important params for evaluation
NUM_EVAL_RUNS = 25
NUM_KFOLD_SPLITS = 2
TEST_SIZE = 0.5
USE_KFOLD = True
NUM_LOCAL_WINDOWS = 3

# Only set one of them True
USE_PCA = False
USE_LDA = False

SOFT_MAJORITY_VOTE = True

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
        return KNeighborsClassifier(n_neighbors=5)
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


def build_global_windows(X, y):
    X_global_windows = []
    y_global_windows = []

    i = 0
    while i < X.shape[0] - NUM_LOCAL_WINDOWS:
        # "Span" a local window
        x = [z for z in range(i, i + NUM_LOCAL_WINDOWS)]

        # Check if all data points in the local window have the same label
        cur_label = y[i]
        if all([y[t] == cur_label for t in x]):
            X_global_windows.append(x)
            y_global_windows.append(cur_label)

            i += NUM_LOCAL_WINDOWS  # No overlapping windows!
        else:
            i += 1

    X_global_windows = np.array(X_global_windows)
    y_global_windows = np.array(y_global_windows)

    return X_global_windows, y_global_windows


def predict_on_global_windows(y_pred_proba):
    y_pred_final = []
    if SOFT_MAJORITY_VOTE:
        for i in range(0, y_pred_proba.shape[0], NUM_LOCAL_WINDOWS):
            # "Soft" majority vote (sequential bayesian hypothesis testing)
            h0 = 0.5
            h1 = 0.5
            # normalization = y_pred_proba[0][0] * 0.5 + y_pred_proba[0][1] * 0.5
            # y_final_soft = (y_pred_proba[0][1] * h1) / normalization
            for j in range(i, i + NUM_LOCAL_WINDOWS):
                normalization = y_pred_proba[j][0] * h0 + y_pred_proba[j][1] * h1
                y_final = (y_pred_proba[j][1] * h1) / normalization
                h1 = y_final
                h0 = 1 - h1
            y_final = 1 if y_final >= 0.5 else 0
            y_pred_final.append(y_final)
    else:  # Hard Majority vote
        for i in range(0, y_pred_proba.shape[0], NUM_LOCAL_WINDOWS):
            y_final = sum(y_pred_proba[range(i, i + NUM_LOCAL_WINDOWS)]) * 1.0 / NUM_LOCAL_WINDOWS
            y_final = 1 if y_final >= 0.5 else 0
            y_pred_final.append(y_final)

    return y_pred_final


def classify(X, y):
    if cur_model in ["logreg", "svm"]:
        features = [2118, 3925, 2106, 2237, 3929, 2109, 2108, 3927, 2233, 2121, 2235, 2184, 3928, 3913, 3924]
    elif cur_model == "randomforest":
        features = [2187, 2186, 2185, 2120, 2204, 2102, 2121, 2103, 2118, 2184, 2205, 2119, 2101, 2174, 2168]
    else:
        features = [2118, 2121, 2120, 2187, 2186, 2184, 2185, 2204, 2102, 2103, 3925, 2237, 2106, 2205, 3929]

    if features is not None:
        X = X[:, features]

    feature_usage = np.zeros(X.shape[1])

    user_metrics = []

    # Build global windows
    X_global, y_global = build_global_windows(X, y)

    # Fit a classifier for each user
    num_users = len(np.unique(y_global))
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
        y_ = np.array(y_global == label).astype(np.int)
        X_ = X_global

        # Use multiple runs for estimating the performance
        for i in range(NUM_EVAL_RUNS):
            # Shuffle data
            X_, y_ = shuffle(X_, y_)

            # Resampling (because data is highly unbalanced)
            X_, y_ = RandomUnderSampler().fit_sample(X_, y_)
            if i == 0:
                print("Number of samples: {0}".format(X_.shape[0]))

            def fit_eval_model(X_train, X_test, y_train, y_test, feature_usage):
                # Get data from global windows
                X_train_ = np.vstack([X[X_train[k, :], :] for k in range(X_train.shape[0])])
                y_train_ = np.vstack([list(repeat(l, NUM_LOCAL_WINDOWS)) for l in y_train]).flatten()
                X_test_ = np.vstack([X[X_test[k, :], :] for k in range(X_test.shape[0])])

                # Preprocessing
                scaler = StandardScaler()
                X_train_ = scaler.fit_transform(X_train_)
                X_test_ = scaler.transform(X_test_)
                
                if USE_PCA:
                    X_train_, X_test_ = perform_pca(X_train_, X_test_)

                if USE_LDA:
                    X_train_, X_test_ = perform_lda(X_train_, X_test_, y_train_)

                # Fit model
                model = get_model()
                model.fit(X_train_, y_train_)

                if hasattr(model, "feature_importances_"):
                    feature_usage =  analyse_feature_importance(model, feature_usage)

                if SOFT_MAJORITY_VOTE:
                    y_test_pred = model.predict_proba(X_test_)
                else:
                    y_test_pred = model.predict(X_test_)

                y_test_pred = predict_on_global_windows(y_test_pred)

                # Evaluate model
                accuracy_scores.append(accuracy_score(y_test, y_test_pred))
                recall_scores.append(recall_score(y_test, y_test_pred))
                precision_scores.append(precision_score(y_test, y_test_pred))
                f1_scores.append(f1_score(y_test, y_test_pred))
                kappa_scores.append(cohen_kappa_score(y_test, y_test_pred))
                confusion_matricies.append(confusion_matrix(y_test, y_test_pred))
                roc_auc_scores.append(roc_auc_score(y_test, y_test_pred))
                return feature_usage

            # Split data by either using kfold or a simple train-test split
            if USE_KFOLD == True:
                kf = KFold(n_splits=NUM_KFOLD_SPLITS, shuffle=True)
                for train_index, test_index in kf.split(X_):
                    X_train, X_test = X_[train_index], X_[test_index]
                    y_train, y_test = y_[train_index], y_[test_index]

                    feature_usage = fit_eval_model(X_train, X_test, y_train, y_test, feature_usage)
            else:
                X_train, X_test, y_train, y_test = train_test_split(X_, y_, test_size=TEST_SIZE,
                                                                    shuffle=True)  # , stratify=y_)
                feature_usage = fit_eval_model(X_train, X_test, y_train, y_test, feature_usage)

        # Save results
        user_metrics.append(
               [accuracy_scores, precision_scores, recall_scores, f1_scores, kappa_scores, roc_auc_scores])

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
        print("Usage: <data.npz> <scores_out.npz> <model>")
    else:
        f_in = sys.argv[1]
        scores_f_out = sys.argv[2]
        cur_model = sys.argv[3]

        # Load data
        data = np.load(f_in)
        X = data["X"]
        y = data["y"]

        results = classify(X, y)
        user_metrics = results

        if scores_f_out is not None:
            np.savez(scores_f_out, X=user_metrics)



