# import numpy as np

# def cross_entropy_loss(y_true, y_pred):
#     """
#     Compute average cross-entropy loss for multi-class classification.
#     """
#     # Write code here
#     y_true = np.array(y_true)
#     y_pred = np.array(y_pred)

#     N = y_true.size[0]
    
#     correct_p = y_pred[np.arange(N), y_true]
#     loss = -np.sum(np.log(correct_p))/N
#     return loss
#     pass

import numpy as np

def cross_entropy_loss(y_true, y_pred):
    
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    N = y_true.shape[0]
    
    # Pick probability of correct class for each sample
    correct_probs = y_pred[np.arange(N), y_true]
    
    # Compute loss
    loss = -np.sum(np.log(correct_probs)) / N
    
    return float(loss)