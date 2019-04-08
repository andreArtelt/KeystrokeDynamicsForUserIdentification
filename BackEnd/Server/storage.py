# -*- coding: utf-8 -*-
import sqlite3
import time


class Storage(object):
    def __init__(self, db_file):
        self.db_file = db_file
        self.connection = sqlite3.connect(self.db_file)

    def close(self):
        if self.connection is not None:
            self.connection.commit()
            self.connection.close()

    def __enter__(self):
        return self

    def __exit__(self, type, val, tb):
        self.close()

    def create(self):
        cmd = self.connection.cursor()
        cmd.execute("CREATE TABLE IF NOT EXISTS recordings (type INTEGER NOT NULL, time INTEGER NOT NULL, value TEXT NOT NULL, time2 INTEGER NOT NULL)")

    def write(self, event_type, time1, value):
        query = "INSERT INTO recordings (type, time, value, time2) VALUES (?, ?, ?, ?)"

        cmd = self.connection.cursor()

        time2 = int(round(time.time() * 1000))
        cmd.execute(query, (event_type, time1, value, time2))
