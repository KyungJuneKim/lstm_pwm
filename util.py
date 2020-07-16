import matplotlib.pyplot as plt
import numpy as np
from typing import List
from warnings import warn


def split_data(x: List, y: List, ratio: List[float] = None):
    if ratio is None:
        ratio = [0.5, 0.3]
    if len(x) != len(y):
        raise AssertionError('dataset lists are not the same length')
    if np.sum(ratio) > 1:
        raise AssertionError('The sum of the ratios must not exceed 1')
    if any(r*len(x) < 1 for r in ratio):
        warn('Too little ratio')

    indices = np.multiply(np.cumsum(ratio), len(x)).astype(np.int32)
    splited = [(x[:indices[0]], y[:indices[0]])]
    for i, j in zip(indices[:-1], indices[1:]):
        splited.append((x[i:j], y[i:j]))
    splited.append((x[indices[-1]:], y[indices[-1]:]))

    return splited


def plot_model(h, validation: bool = False, keys: List[str] = None):
    if keys is None:
        keys = ['loss']
    fig, axes = plt.subplots(nrows=1, ncols=len(keys), sharex='all', figsize=(15, 6))

    for idx, key in enumerate(keys):
        axes[idx].set_title('Model ' + key)
        axes[idx].plot(h.history[key], label=key)
        if validation:
            axes[idx].plot(h.history['val_' + key], label='val_' + key)
        axes[idx].set_xlabel('Epoch')
        axes[idx].set_ylabel(key.capitalize())
        axes[idx].legend()

    plt.tight_layout()
    plt.show()
