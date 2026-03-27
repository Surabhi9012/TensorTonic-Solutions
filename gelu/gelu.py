import numpy as np
import math

def gelu(x):
    """
    Compute the Gaussian Error Linear Unit (exact version using erf).
    x: list or np.ndarray
    Return: np.ndarray of same shape (dtype=float)
    """
    # Write code here
    x = np.array(x, dtype = float)

    # return 0.5 * x (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))
    erf_vec = np.vectorize(math.erf)
    
    return 0.5 * x * (1 + erf_vec(x / np.sqrt(2)))
    
    pass

