#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function
import sys
import numpy as np
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn_lvq import GrmlvqModel, MrslvqModel, LgmlvqModel, LmrslvqModel
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, cohen_kappa_score, confusion_matrix, make_scorer
from imblearn.under_sampling import RandomUnderSampler
import GPy
import GPyOpt


NUM_EVAL_RUNS = 5
NUM_KFOLD_SPLITS = 2
TEST_SIZE = 0.5

scoring_function = f1_score


def bayesopt_logisticregression(X, y):
    domain =[{'name': 'C', 'type': 'continuous', 'domain': (0.0, 10.0)}]

    def fit_model(x):
        x = np.atleast_2d(np.exp(x))
        fs = np.zeros((x.shape[0],1))

        for i in range(x.shape[0]):
            fs[i] = 0

            kf = KFold(n_splits=NUM_KFOLD_SPLITS, shuffle=True)
            for train_index, test_index in kf.split(X_):
                X_train, X_test = X_[train_index], X_[test_index]
                y_train, y_test = y_[train_index], y_[test_index]
            
                scaler = StandardScaler()
                X_train = scaler.fit_transform(X_train)
                X_test = scaler.transform(X_test)

                model = LogisticRegression(C=x[i, 0])
                model.fit(X_train, y_train)
                
                y_test_pred = model.predict(X_test)
                score = -1.0 * scoring_function(y_test, y_test_pred)

                fs[i] += score

            fs[i] *= 1.0 / NUM_KFOLD_SPLITS
        
        return fs

    # Create otimizer
    opt = GPyOpt.methods.BayesianOptimization(f=fit_model, domain=domain, acquisition_type='EI', acquisition_weight=0.1)

    opt.run_optimization(max_iter=20)
    params = np.exp(opt.X[np.argmin(opt.Y)])

    return {'C': params[0]}

def gridsearch_logisticregression(X, y):
    params = {'C': np.logspace(-2, 6, 10)}
    model = LogisticRegression()

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    grid = GridSearchCV(model, params, scoring=make_scorer(scoring_function), cv=3, n_jobs=-1)
    grid.fit(X, y)

    return grid.best_params_


def gridsearch_grmlvq(X, y):
    params = {'prototypes_per_class': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}
    model = GrmlvqModel()

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    grid = GridSearchCV(model, params, scoring=make_scorer(scoring_function), cv=3, n_jobs=-1)
    grid.fit(X, y)

    return grid.best_params_

def gridsearch_mrslvq(X, y):
    params = {'prototypes_per_class': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}
    model = MrslvqModel()

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    grid = GridSearchCV(model, params, scoring=make_scorer(scoring_function), cv=3, n_jobs=-1)
    grid.fit(X, y)

    return grid.best_params_

def gridsearch_lgmlvq(X, y):
    params = {'prototypes_per_class': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}
    model = LgmlvqModel()

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    grid = GridSearchCV(model, params, scoring=make_scorer(scoring_function), cv=3, n_jobs=-1)
    grid.fit(X, y)

    return grid.best_params_

def gridsearch_lmrslvq(X, y):
    params = {'prototypes_per_class': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}
    model = LmrslvqModel()

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    grid = GridSearchCV(model, params, scoring=make_scorer(scoring_function), cv=3, n_jobs=-1)
    grid.fit(X, y)

    return grid.best_params_


def gridsearch_knn(X, y):
    params = {'n_neighbors': [2, 3, 4, 5, 6, 7, 8, 9, 10]}
    model = KNeighborsClassifier()

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    grid = GridSearchCV(model, params, scoring=make_scorer(scoring_function), cv=3, n_jobs=-1)
    grid.fit(X, y)

    return grid.best_params_

def bayesopt_knn(X, y):
    domain =[{'name': 'n_neighbors', 'type': 'discrete', 'domain': [2, 3, 4, 5, 6, 7, 8, 9, 10]}]

    def fit_model(x):
        x = np.atleast_2d(x)
        fs = np.zeros((x.shape[0],1))

        for i in range(x.shape[0]):
            fs[i] = 0

            kf = KFold(n_splits=NUM_KFOLD_SPLITS, shuffle=True)
            for train_index, test_index in kf.split(X_):
                X_train, X_test = X_[train_index], X_[test_index]
                y_train, y_test = y_[train_index], y_[test_index]
            
                model = KNeighborsClassifier(n_neighbors=int(x[i, 0]))
                model.fit(X_train, y_train)
                
                y_test_pred = model.predict(X_test)
                score = -1.0 * scoring_function(y_test, y_test_pred)

                fs[i] += score

            fs[i] *= 1.0 / NUM_KFOLD_SPLITS
        
        return fs

    # Create otimizer
    opt = GPyOpt.methods.BayesianOptimization(f=fit_model, domain=domain, acquisition_type='EI', acquisition_weight=0.1)

    opt.run_optimization(max_iter=50)
    params = opt.X[np.argmin(opt.Y)]
    
    return {'n_neighbors': int(params[0])}


def bayesopt_decisiontree(X, y):
    domain =[{'name': 'max_depth', 'type': 'discrete', 'domain': np.arange(5, 15, 1)}]

    def fit_model(x):
        x = np.atleast_2d(x)
        fs = np.zeros((x.shape[0],1))

        for i in range(x.shape[0]):
            fs[i] = 0

            kf = KFold(n_splits=NUM_KFOLD_SPLITS, shuffle=True)
            for train_index, test_index in kf.split(X_):
                X_train, X_test = X_[train_index], X_[test_index]
                y_train, y_test = y_[train_index], y_[test_index]
            
                model = DecisionTreeClassifier(max_depth=x[i, 0])
                model.fit(X_train, y_train)
                
                y_test_pred = model.predict(X_test)
                score = -1.0 * scoring_function(y_test, y_test_pred)

                fs[i] += score

            fs[i] *= 1.0 / NUM_KFOLD_SPLITS
        
        return fs

    # Create otimizer
    opt = GPyOpt.methods.BayesianOptimization(f=fit_model, domain=domain, acquisition_type='EI', acquisition_weight=0.1)

    opt.run_optimization(max_iter=50)
    params = opt.X[np.argmin(opt.Y)]
    
    return {'max_depth': params[0]}

def gridsearch_decisiontree(X, y):
    params = {'max_depth': np.arange(5, 15, 1)}
    model = DecisionTreeClassifier()

    grid = GridSearchCV(model, params, scoring=make_scorer(scoring_function), cv=3, n_jobs=-1)
    grid.fit(X, y)

    return grid.best_params_

def bayesopt_adaboost(X, y):
    domain =[{'name': 'n_estimators', 'type': 'discrete', 'domain': np.arange(100, 1000, 100)},
             {'name': 'learning_rate', 'type': 'discrete', 'domain': np.arange(0.0001, 1.0, 0.005)}]

    def fit_model(x):
        x = np.atleast_2d(x)
        fs = np.zeros((x.shape[0],1))

        for i in range(x.shape[0]):
            fs[i] = 0

            kf = KFold(n_splits=NUM_KFOLD_SPLITS, shuffle=True)
            for train_index, test_index in kf.split(X_):
                X_train, X_test = X_[train_index], X_[test_index]
                y_train, y_test = y_[train_index], y_[test_index]
            
                model = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=7), n_estimators=int(x[i, 0]), learning_rate=x[i, 1])
                model.fit(X_train, y_train)
                
                y_test_pred = model.predict(X_test)
                score = -1.0 * scoring_function(y_test, y_test_pred)

                fs[i] += score

            fs[i] *= 1.0 / NUM_KFOLD_SPLITS
        
        return fs

    # Create otimizer
    opt = GPyOpt.methods.BayesianOptimization(f=fit_model, domain=domain, acquisition_type='EI', acquisition_weight=0.1)

    opt.run_optimization(max_iter=50)
    params = opt.X[np.argmin(opt.Y)]
    
    return {'n_estimators': params[0], 'learning_rate': params[1]}

def gridsearch_adaboost(X, y):
    params = {'n_estimators': np.arange(100, 1000, 100), 'learning_rate': np.arange(0.0001, 1.0, 0.005)}
    model = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=7))

    grid = GridSearchCV(model, params, scoring=make_scorer(scoring_function), cv=3, n_jobs=-1)
    grid.fit(X, y)

    return grid.best_params_

def bayesopt_randomforest(X, y):
    domain =[{'name': 'n_estimators', 'type': 'discrete', 'domain': np.arange(100, 1000, 100)},
             {'name': 'max_depth', 'type': 'discrete', 'domain': np.arange(5, 15, 1)}]

    def fit_model(x):
        x = np.atleast_2d(x)
        fs = np.zeros((x.shape[0],1))

        for i in range(x.shape[0]):
            fs[i] = 0

            kf = KFold(n_splits=NUM_KFOLD_SPLITS, shuffle=True)
            for train_index, test_index in kf.split(X_):
                X_train, X_test = X_[train_index], X_[test_index]
                y_train, y_test = y_[train_index], y_[test_index]
            
                model = RandomForestClassifier(n_estimators=int(x[i, 0]), max_depth=int(x[i, 1]))
                model.fit(X_train, y_train)
                
                y_test_pred = model.predict(X_test)
                score = -1.0 * scoring_function(y_test, y_test_pred)

                fs[i] += score

            fs[i] *= 1.0 / NUM_KFOLD_SPLITS
        
        return fs

    # Create otimizer
    opt = GPyOpt.methods.BayesianOptimization(f=fit_model, domain=domain, acquisition_type='EI', acquisition_weight=0.1)

    opt.run_optimization(max_iter=50)
    params = opt.X[np.argmin(opt.Y)]
    
    return {'n_estimators': params[0], 'max_depth': params[1]}

def gridsearch_randomforest(X, y):
    params = {'n_estimators': np.arange(100, 1000, 100), 'max_depth': np.arange(5, 15, 1)}
    model = RandomForestClassifier()

    grid = GridSearchCV(model, params, scoring=make_scorer(scoring_function), cv=3, n_jobs=-1)
    grid.fit(X, y)

    return grid.best_params_

def bayesopt_svm(X, y):
    domain =[{'name': 'C', 'type': 'continuous', 'domain': (0.0, 7.0)},
	         {'name': 'gamma', 'type': 'continuous', 'domain': (-12.0, -2.0)}]

    def fit_model(x):
        x = np.atleast_2d(np.exp(x))
        fs = np.zeros((x.shape[0],1))

        for i in range(x.shape[0]):
            fs[i] = 0

            kf = KFold(n_splits=NUM_KFOLD_SPLITS, shuffle=True)
            for train_index, test_index in kf.split(X_):
                X_train, X_test = X_[train_index], X_[test_index]
                y_train, y_test = y_[train_index], y_[test_index]
            
                scaler = StandardScaler()
                X_train = scaler.fit_transform(X_train)
                X_test = scaler.transform(X_test)

                model = SVC(C=x[i, 0], gamma=x[i, 1])
                model.fit(X_train, y_train)
                
                y_test_pred = model.predict(X_test)
                score = -1.0 * scoring_function(y_test, y_test_pred)

                fs[i] += score

            fs[i] *= 1.0 / NUM_KFOLD_SPLITS
        
        return fs

    # Create otimizer
    opt = GPyOpt.methods.BayesianOptimization(f=fit_model, domain=domain, acquisition_type='EI', acquisition_weight=0.1)

    opt.run_optimization(max_iter=50)
    params = np.exp(opt.X[np.argmin(opt.Y)])
    
    return {'C': params[0], 'Gamma': params[1]}

def gridsearch_svm(X, y):
    params = {'C': np.logspace(-2, 6, 10), 'gamma': np.logspace(-15, -2, 10)}
    model = SVC(kernel='rbf')

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    grid = GridSearchCV(model, params, scoring=make_scorer(scoring_function), cv=3, n_jobs=-1)
    grid.fit(X, y)

    return grid.best_params_


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: <data.npz>")
    else:
        f_in = sys.argv[1]

        # Load data
        data = np.load(f_in)
        X = data["X"]
        y = data["y"]

        # Select features
        features = [2118, 2121, 2021, 3924, 3913, 1999, 2233, 1993, 2002, 1992, 2234, 2120, 2103, 2204, 2119, 2186, 2106, 3917]
        X = X[:, features]

        print(X.shape)

        # Fit a classifier for each user
        num_users = len(np.unique(y))
        for label in range(num_users):
            # One vs. Rest
            y_ = np.array(y == label).astype(np.int)
            X_ = X

            for _ in range(NUM_EVAL_RUNS):
                # Shuffle data
                X_, y_ = shuffle(X_, y_)

                # Resampling (because data is highly unbalanced)
                X_, y_ = RandomUnderSampler().fit_sample(X_, y_)

                # Find good parameters
                print("AdaBoost")
                print("User: {0} Params: {1}".format(label, str(gridsearch_adaboost(X_, y_))))
                print("User: {0} Params: {1}".format(label, str(bayesopt_adaboost(X_, y_))))

                print("LVQ")
                print("GRMLVQ - User: {0} Params: {1}".format(label, str(gridsearch_grmlvq(X_, y_))))
                print("LGMLVQ - User: {0} Params: {1}".format(label, str(gridsearch_lgmlvq(X_, y_))))
                print("LMRSLVQ - User: {0} Params: {1}".format(label, str(gridsearch_lmrslvq(X_, y_))))
                print("MRSLVQ - User: {0} Params: {1}".format(label, str(gridsearch_mrslvq(X_, y_))))

                print("DecisionTree")
                print("User: {0} Params: {1}".format(label, str(bayesopt_decisiontree(X_, y_))))
                print("User: {0} Params: {1}".format(label, str(gridsearch_decisiontree(X_, y_))))

                print("RandomForest")
                print("User: {0} Params: {1}".format(label, str(gridsearch_randomforest(X_, y_))))
                print("User: {0} Params: {1}".format(label, str(bayesopt_randomforest(X_, y_))))

                print("LogisiticRegression")
                print("User: {0} Params: {1}".format(label, str(bayesopt_logisticregression(X_, y_))))
                print("User: {0} Params: {1}".format(label, str(gridsearch_logisticregression(X_, y_))))
                
                print("KNN")
                print("User: {0} Params: {1}".format(label, str(bayesopt_knn(X_, y_))))
                print("User: {0} Params: {1}".format(label, str(gridsearch_knn(X_, y_))))

                print("SVM")
                print("User: {0} Params: {1}".format(label, str(bayesopt_svm(X_, y_))))
                print("User: {0} Params: {1}".format(label, str(gridsearch_svm(X_, y_))))
