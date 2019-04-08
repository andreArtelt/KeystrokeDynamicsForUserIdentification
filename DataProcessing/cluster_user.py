#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function
import sys
import numpy as np
from sklearn.cluster import KMeans, SpectralClustering, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler


def cluster(X, n_clusters, X_test=None):
    model = KMeans(n_clusters=n_clusters, max_iter=500)
    #model = SpectralClustering(n_clusters=n_clusters)
    #model = AgglomerativeClustering(n_clusters=n_clusters)

    scaler = StandardScaler()

    model.fit(scaler.fit_transform(X))

    if isinstance(model, KMeans) and X_test is not None:
        return model.labels_, model.predict(X_test)

    return model.labels_


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: <data_in.npz> <data_out.npz> <num_clusters>")
    else:
        f_in = sys.argv[1]
        f_out = sys.argv[2]
        n_clusters = int(sys.argv[3])

        # Load data
        data = np.load(f_in)
        X = data["X"]

        # Cluster user
        y_new = cluster(X, n_clusters=n_clusters)

        # Save result
        np.savez(f_out, X=X, y=y_new)
