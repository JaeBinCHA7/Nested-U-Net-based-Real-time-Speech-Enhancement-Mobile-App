import os

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)