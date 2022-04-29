"""Data module."""

import os
import pickle

class Data(dict):
    """
    Data is a dictionary-like object that exposes its keys as attributes.

    Heavily inspired by sklearn's `Bunch` class.
    """
    def __init__(self, **kwargs):
        dict.__init__(self, kwargs)

    def __setattr__(self, key, value):
        self[key] = value

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(key)


def dump(data, path):
    """
    Dump some data to disk with pickle.
    """
    # Create destination path if it doesn't exist
    dirname = os.path.dirname(path)
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    # Dump data to destination file
    with open(path, 'wb') as file:
        pickle.dump(data, file)


def load(path):
    """
    Load some data from pickle file.
    """
    with open(path, 'rb') as file:
        data = pickle.load(file)
    return data
