#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import numpy as np

metric_names = ["Accuracy", "Precision", "Recall", "F1", "Kappa", "ROC-AUC"]


def calculate_metrics(X, score_id):
    return np.mean(X[:, score_id, :]), np.std(X[:, score_id, :]), np.min(X[:, score_id, :]), np.max(X[:, score_id, :])


def write_csv(f_out, data):
    with open(f_out, 'w') as out:
        out.write(", mean, std, min, max\n")
        for metric in metric_names:
            mean, std, min, max = calculate_metrics(data, metric_names.index(metric))
            out.write("{}, {}, {}, {}, {}\n".format(metric, mean, std, min, max))


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: <input.npz> <output.csv>")
    else:
        f_in = sys.argv[1]
        f_out = sys.argv[2]

        data = np.load(f_in)["X"]

        write_csv(f_out, data)
