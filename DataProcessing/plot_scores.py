#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function
import sys
import numpy as np
import matplotlib.pyplot as plt


metric_names = ["Accuracy", "Precision", "Recall", "F1", "Kappa", "ROC-AUC"]


def scatterplot_helper(X, score_id, title):
    x = X[:, score_id, :]
    mean = np.mean(x, axis=1)
    var = np.var(x, axis=1)
    mean_var_pos = mean + var
    mean_var_neg = mean - var
    
    y = np.concatenate((mean, mean_var_pos, mean_var_neg))
    xticks = np.array([i for i in range(1, x.shape[0] + 1)])
    x = np.concatenate((xticks, xticks, xticks))
    
    plt.scatter(x, y)
    plt.axhline(y=0.5, color='r', linestyle='-')
    plt.xticks(xticks)
    plt.yticks(np.arange(0.4, 1.0, 0.05))
    plt.ylabel("Score")
    plt.xlabel("User")
    plt.title("{0} - Metric: {1}".format(title, metric_names[score_id]))


def boxplot_helper(X, score_id, title):
    plt.boxplot(np.transpose(X[:, score_id, :]))
    plt.axhline(y=0.5, color='r', linestyle='-')
    plt.yticks(np.arange(0.4, 1.0, 0.05))
    plt.ylabel("Score")
    plt.xlabel("User")
    plt.title("{0} - Metric: {1}".format(title, metric_names[score_id]))


def create_boxplots(X, score_id, title, file_out=None):
    plt.figure()
    boxplot_helper(X, score_id, title)

    if file_out is None:
        plt.show()
    else:
        plt.savefig(file_out)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: <data.npz> <plot_out>")
    else:
        f_in = sys.argv[1]

        f_out = None
        if len(sys.argv) > 2:
            f_out = sys.argv[2]

        data = np.load(f_in)
        X = data["X"]

        for i in [3, 5]:
            for j in range(X.shape[0]):
                print("User={0}\n".format(j+1))
                scores = X[j, i, :].flatten()
                print("{0}:\n{1}\nMean={2} Var={3} Min={4} Max={5}\n".format(metric_names[i], str(scores),
                                                            np.mean(scores), np.var(scores), np.min(scores), np.max(scores)))

            create_boxplots(X, score_id=i, title=f_in, file_out=f_out + metric_names[i] + ".png" if f_out is not None else None)
