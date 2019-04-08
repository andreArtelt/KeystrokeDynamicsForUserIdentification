#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function
import sys
import numpy as np
import cPickle
from os import listdir
from os.path import join, isdir
from joblib import Parallel, delayed
from create_features import create_data, build_features_name_dict
from utils import list_dirs_in_directory as list_all_dirs


def process_dir(d, i, n_split, window_size, window_shift, time_col_id):
    return (create_data(d, n_split, window_size, window_shift, time_col_id), i)


def create_dataset(in_dirs, output_file, n_split=None, window_size=50, window_shift=50, time_col_id='time', n_jobs=-1):
    """
    Creates a data set from "raw" data. For this purpose the raw data (which is a time series) is split into serval batches
    and from each batch a fixed nubmer of features is extracted/computed.
    NOTE: If you use a sliding window (see `window_size` and `window_shift`) be aware of strongly overlapping windows!
          (Strongly overlapping windows produces highly correlated data points, which makes estimating the model performance really challenging)
    Parameters
    ----------
    in_dirs : list of str
        Specifies a list of directories containg data
    output_file : str
        Specifies the file where the resulting numpy arrays (X and y) are saved
    n_split : int, optional
        Specifies the number of parts/data points the data is divided into (the default is None which means that `window_size` and `window_shift` are used to get batches from the data).
    window_size : int, optional
        Specifies the size/width of the window in s (the default is 50s). NOTE: This quantity is ignored if `n_split` != None
    window_shift : int, optional
        Specifies how much (in s) the window gets shifted each step (the default is 50s). NOTE: This quantity is ignored if `n_split` != None
    time_col_id : str, optional
        Specifies which time column to use (the default is 'time' which corresponds to the time measured in the browser). Use 'time2' for using the time measured in the server.
    n_jobs: int, optional
        Specifies the number cpu cores used for processing the data (the default is -1). Use `n_jobs=-1` to use all available cpu cores.
    """
    X = []
    y = []

    # Process each directory (each user has it's own directory!)
    data_set = Parallel(n_jobs=n_jobs)(delayed(process_dir)(d, i, n_split, window_size, window_shift, time_col_id) for i, d in enumerate(in_dirs))
    for x_data, i in data_set:
        for x in x_data:
            X.append(x)
            y.append(i)
    
    X = np.array(X)
    y = np.array(y)

    # Compute a mapping between feature ids and names/descs
    feature_names = build_features_name_dict(X.shape[1])

    # Save data
    np.savez(output_file, X=X, y=y)

    with open(output_file + ".pkl", 'w') as f_out:
        cPickle.dump(feature_names, f_out)


# python create_dataset.py "../Data/" "../Data/data.npz"
if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: <data_root_dir> <data_out.npz> <time_id='time'> <window_size=20>")
    else:
        # Parse arguments
        root_dir = sys.argv[1]
        output_file = sys.argv[2]
        time_col_id = "time"
        if len(sys.argv) > 3:
            time_col_id = sys.argv[3]
        window_size = 20
        if len(sys.argv) > 4:
            window_size = int(sys.argv[4])

        dirs = list_all_dirs(root_dir)

        # Create data set
        create_dataset(dirs, output_file, time_col_id=time_col_id, n_jobs=-1, window_size=window_size, window_shift=window_size)
