import numpy as np
import contextlib
# import torch

# Configures numpy print options
@contextlib.contextmanager

def _printoptions(*args, **kwargs):
    original = np.get_printoptions()
    np.set_printoptions(*args, **kwargs)
    try:
        yield
    finally: 
        np.set_printoptions(**original)