from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    num_train = X.shape[0]
    num_class = W.shape[1]
    scores = X.dot(W)
    softmax_scores = np.zeros_like(scores)
    # scores = np.exp(scores) # might be too large! numerical stability is low!
    for i in range(num_train):
        softmax_scores[i] = scores[i] - np.max(scores[i])
        softmax_scores[i] = np.exp(softmax_scores[i])
        loss_tmp = - np.log( softmax_scores[i, y[i]] / np.sum(softmax_scores[i]) )
        loss += loss_tmp
        # Calculus!!
        for j in range(num_class):
            dW[:, j] += X[i].T * ( softmax_scores[i, j] / np.sum(softmax_scores[i]) )
        dW[:, y[i]] -= X[i].T
        
    loss /= num_train
    dW /= num_train
    loss += reg * np.sum(W * W)      
    dW += 2 * reg * W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    num_train = X.shape[0]
    num_class = W.shape[1]
    scores = X.dot(W)
    
    # Create softmax scores for numerical stability issue.
    # Stupid way:
#     scores_row_max = np.max(scores, axis=1)[np.newaxis] # Change 1d array to 2d array.
#     scores_row_max = np.transpose(scores_row_max)
#     scores -= scores_row_max
    # Smart way:
    scores -= np.max(scores, axis=1, keepdims=True)
    
    # Calculate softmax loss.
    scores = np.exp(scores)
    softmax = scores / np.sum(scores, axis=1, keepdims=True)
    loss = np.sum( - np.log( softmax[np.arange(num_train), y] ) )
    
    # Calculate softmax gradient.
    softmax[np.arange(num_train), y] -= 1
    dW = X.T.dot(softmax)
#     dW[:, y] -= X[np.arange(num_train), :].T   # This is wrong! why?
    
    loss /= num_train
    dW /= num_train
    # Add regularization item.
    loss += reg * np.sum(W * W)
    dW += 2 * reg * W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
