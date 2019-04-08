#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function
import sys
import os
import pandas as pd
import numpy as np
from joblib import Parallel, delayed
from utils import list_dirs_in_directory, list_files_in_directory, create_dir_if_not_exists


def apply_defence_mechanisms(d_in, d_out):
    files = filter(lambda x: x.endswith(".csv"), list_files_in_directory(d_in))

    for f in files:
        df = pd.read_csv(f)
        
        # Rounding (remove exactness from timestamps, e.g. remove ms)
        rf = 1000.0 # TODO: Hope that the server is unable to compute precise/useful time stamps (order of ms)
        df["time"] = df["time"] / rf
        df = df.round({"time": 0})
        df["time"] = df["time"] * rf
        df["time"] = df["time"].astype(int)

        # Save result
        df.to_csv(os.path.join(d_out, os.path.basename(f)))


def process_dir(d, dir_out):
    d_out = os.path.join(dir_out, os.path.basename(d))

    create_dir_if_not_exists(d_out)
    apply_defence_mechanisms(d, d_out)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: <data_root_dir> <data_out_dir>")
    else:
        dir_in = sys.argv[1]
        dir_out = sys.argv[2]

        dirs = list_dirs_in_directory(dir_in)
        Parallel(n_jobs=-1)(delayed(process_dir)(d, dir_out) for d in dirs)
