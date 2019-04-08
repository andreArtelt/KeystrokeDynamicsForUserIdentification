#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function
import sys
from joblib import Parallel, delayed
from utils import list_dirs_in_directory, list_files_in_directory
from db2df import database_to_dataframe


def handle_dir(d, time_col_id):
    files = filter(lambda x: x.endswith(".db"), list_files_in_directory(d))

    for f in files:
        database_to_dataframe(f, d, time_col_id)

def process_directories(root_dir, time_col_id):
    subdirs = list_dirs_in_directory(root_dir)

    Parallel(n_jobs=-1)(delayed(handle_dir)(d, time_col_id) for d in subdirs)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: <data_root_dir> <time_col_id=time>")
    else:
        data_root_dir = sys.argv[1]
        time_col_id = "time"
        if len(sys.argv) > 2:
            time_col_id = sys.argv[2]

        process_directories(data_root_dir, time_col_id)
