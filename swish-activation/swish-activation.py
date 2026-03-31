import numpy as np

def swish(x):
    """
    Implement Swish activation function.
    """
    # Write code here
    x = np.array(x, dtype = float)
    for a in x:
        a = 1 / (1 + np.exp(-x))

    return a*x
    pass