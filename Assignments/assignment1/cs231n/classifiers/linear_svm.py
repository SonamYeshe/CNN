from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange

def svm_loss_naive(W, X, y, reg):
    """
    Structured SVM loss function, naive implementation (with loops).

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
    dW = np.zeros(W.shape) # initialize the gradient as zero

    # compute the loss and the gradient
    num_classes = W.shape[1]
    num_train = X.shape[0]
    loss = 0.0
    for i in range(num_train):
        scores = X[i].dot(W)
        correct_class_score = scores[y[i]]
        for j in range(num_classes):
            if j == y[i]:
                continue
            margin = scores[j] - correct_class_score + 1 # note delta = 1
            if margin > 0:
                loss += margin
                dW[:, j] += X[i] # Added by Jiawei, ds/dW = x
                dW[:, y[i]] -= X[i] # Added by Jiawei, ds/dW = x

    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.
    loss /= num_train
    dW /= num_train # Added by Jiawei, L = sum(L_i) / num_train

    # Add regularization to the loss.
    loss += reg * np.sum(W * W) # W*W is element-wise multiplication.
    dW += reg * 2 * W # Added by Jiawei, d(W^2) = 2*W

    #############################################################################
    # TODO:                                                                     #
    # Compute the gradient of the loss function and store it dW.                #
    # Rather that first computing the loss and then computing the derivative,   #
    # it may be simpler to compute the derivative at the same time that the     #
    # loss is being computed. As a result you may need to modify some of the    #
    # code above to compute the gradient.                                       #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    return loss, dW



def svm_loss_vectorized(W, X, y, reg):
    """
    Structured SVM loss function, vectorized implementation.

    Inputs and outputs are the same as svm_loss_naive.
    """
    loss = 0.0
    dW = np.zeros(W.shape) # initialize the gradient as zero

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the structured SVM loss, storing the    #
    # result in loss.                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    num_train = X.shape[0]
    ## build the Multiclass SVM loss fuction
    scores = X.dot(W)
    
    # DUMMY to calculate true scores!!!
#     yy = np.reshape(y, (num_train, 1))
#     xx = list(range(num_train)) # 'list' object has no attribute 'shape'
#     xx = np.reshape(xx, (num_train, 1))
#     scores -= scores[xx, yy] # this is a matrix broadcast step & the shape of 'scores[xx, yy]' follows the shape of 'xx' and 'yy'
    
    # Treasure to calculate true scores!!!
    scores_true = scores[np.arange(num_train), y].reshape(num_train,1)
    margin = scores - scores_true + 1
    
    # DUMMY to element-wise compare int 0 and a matrix!!!
#     scores += 1
#     scores = np.maximum(np.zeros(scores.shape), scores) # np.maximum provides element-wise comparing function
    
    # Treasure to element-wise compare int 0 and a matrix!!!
    margin = np.maximum(margin, 0)
    
    # DUMMY to element-wise delete true margins!!!
#     loss = np.sum(scores) - num_train # extra added 500 should be deducted, because if j=y_i, max(0, s_j - s_y_i + 1) = 1
    
    # Treasure to delete true margins one by one!!!
    margin[np.arange(num_train), y] = 0
    
    #loss /= num_train
    loss = np.sum(margin) / num_train
    
    # add the regularization term
    loss += reg * np.sum(W * W)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the gradient for the structured SVM     #
    # loss, storing the result in dW.                                           #
    #                                                                           #
    # Hint: Instead of computing the gradient from scratch, it may be easier    #
    # to reuse some of the intermediate values that you used to compute the     #
    # loss.                                                                     #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # DUMMY to calculate gradient from scratch!!!
    # num_classes = W.shape[1]
    # dW[:, np.arange(num_classes)] += np.reshape(np.sum(X, axis = 0), (-1, 1))
    #dW[:, y] -= 

    X_mask = np.zeros(margin.shape)
    X_mask[margin > 0] = 1 # margin>0 means inccorect class
    X_mask[np.arange(num_train), y] = -np.sum(X_mask, axis=1) # Subtract incorrect class (-s_y)
    dW = X.T.dot(X_mask)
    dW /= num_train
    dW += 2 * reg * W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
