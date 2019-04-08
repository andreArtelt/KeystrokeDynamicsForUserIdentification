# -*- coding: utf-8 -*-
import os


def list_dirs_in_directory(path):
    return [os.path.join(path, f) for f in os.listdir(path) if not os.path.isfile(os.path.join(path, f))]

def list_files_in_directory(path):
    return [os.path.join(path, f) for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]

def create_dir_if_not_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)
