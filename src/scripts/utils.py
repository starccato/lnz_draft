import numpy as np

def get_timestamps()->np.ndarray:
    fs = 1024
    nd = 256
    return np.arange(0, nd) / fs