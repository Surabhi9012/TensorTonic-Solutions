import numpy as np

def mean_squared_error(y_pred, y_true):
    """
    Returns: float MSE
    """
    # Write code here
    y_pred = np.array(y_pred)
    y_true = np.array(y_true)
    N = y_pred.shape[0]


    mse = np.sum((y_pred - y_true) ** 2)/N
    return float(mse)

    
    
    # return np.mean((y_pred-y_true)**2)
    
    pass
