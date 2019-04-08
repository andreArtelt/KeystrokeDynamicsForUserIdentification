#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function
import sqlite3
import json
import sys
import pandas as pd
import os


def get_hdr(t, time_col_id):
    return {0: [time_col_id, "key"],
            1: [time_col_id, "key"],
            2: [time_col_id, "x", "y"],
            3: [time_col_id, "x", "y", "b"],
            4: [time_col_id, "x", "y", "b"],
            5: [time_col_id, "x", "y", "b"],
            6: [time_col_id, "x", "y", "dx", "dy", "dz", "m"]}[t]


def process_row(r, t):
    if t == 0 or t == 1:
        return [r[0], json.loads(r[1])]
    elif t == 2:
        values = json.loads(r[1])
        return [r[0], values["x"], values["y"]]
    elif t == 3 or t == 4 or t == 5:
        values = json.loads(r[1])
        return [r[0], values["x"], values["y"], values["b"]]
    elif t == 6:
        values = json.loads(r[1])
        return [r[0], values["x"], values["y"], values["dx"], values["dy"], values["dz"], values["m"]]


def database_to_dataframe(db_file, df_frames_path="", time_col_id="time"):
    # Open database
    db_con = sqlite3.connect(db_file)

    # Read/Parse data
    cmd = db_con.cursor()

    # One dataframe for each type of action
    for t in range(0, 7):
        # Get data
        data = []
        hdr = get_hdr(t, time_col_id)
        for row in cmd.execute("SELECT {0}, value FROM recordings WHERE type = ? ORDER BY {0}".format(time_col_id), (t,)):
            data.append(process_row(row, t))

        # Create and save dataframe
        pd.DataFrame(data, columns=hdr).to_csv(os.path.join(df_frames_path, str(t) + ".csv"), encoding="utf-8")

    # Close database
    db_con.close()


if __name__ == "__main__":
    # Parse command line arguments
    db_file = sys.argv[1] if len(sys.argv) > 1 else None
    df_frames_path = sys.argv[2] if len(sys.argv) > 2 else ""
    time_col_id = sys.argv[3] if len(sys.argv) > 3 else "time"

    if db_file is None:
        print("Usage: <data.db> <df_output_dir> <time_col_id=time>")
    else:    
        # Export database content to dataframes (.csv file)
        database_to_dataframe(db_file, df_frames_path)
