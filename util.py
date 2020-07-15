import matplotlib.pyplot as plt
import numpy as np
from typing import List


def split_data(_x: List, _y: List, ratio: List[float] = None):
    if ratio is None:
        ratio = [0.5, 0.3]
    if len(_x) != len(_y):
        raise AssertionError('dataset lists are not the same length')
    if len(ratio) != 2:
        raise AssertionError('The length of the ratio must be 2')
    if np.sum(ratio) > 1:
        raise AssertionError('The sum of the ratios must not exceed 1')

    idx = np.multiply(np.cumsum(ratio), np.array([len(_x), len(_y)])).astype(np.int32)

    train = (_x[:idx[0]], _y[:idx[0]])
    val = (_x[idx[0]:idx[1]], _y[idx[0]:idx[1]])
    test = (_x[idx[1]:], _y[idx[1]:])

    return train, val, test


def plot_model(h, validation: bool = False, keys: List[str] = None):
    if keys is None:
        keys = ['loss']
    fig, axes = plt.subplots(nrows=1, ncols=len(keys), sharex='all')

    for idx, key in enumerate(keys):
        axes[idx].set_title('Model ' + key)
        axes[idx].plot(h.history[key], label=key)
        if validation:
            axes[idx].plot(h.history['val_' + key], label='val_' + key)
        axes[idx].set_xlabel('Epoch')
        axes[idx].set_ylabel(key.capitalize())
        axes[idx].legend()

    fig.show()
