import numpy as np
from random import shuffle

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
  num_train_samples = X.shape[0]

  for i in range(num_train_samples):
    scores = X[i] @ W 
    adjusted_scores = scores - scores.max() 
    softmax_activation = (np.exp(adjusted_scores))/(np.sum(np.exp(adjusted_scores)))
    loss -= np.log(softmax_activation[y[i]])
    for j in range(W.shape[1]):
        dW[:, j] += X[i] * softmax_activation[j]
    dW[:, y[i]] -= X[i]

  loss = (loss/num_train_samples) + reg * np.sum(W*W)
  dW = (dW/num_train_samples) + 2 * reg * W 

  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

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
  num_train_samples = X.shape[0]
  scores = X @ W
  adjusted_scores = scores - np.max(scores, axis=1, keepdims=True)
  softmax_activation = np.exp(adjusted_scores) / np.sum(np.exp(adjusted_scores), axis=1, keepdims=True)
  
  loss = -np.log(softmax_activation[range(num_train_samples), y]).sum()
  loss = (loss / num_train_samples) + reg * np.sum(W * W)
  
  softmax_activation[np.arange(num_train_samples), y] -= 1
  
  dW = X.T @ softmax_activation
  dW = (dW / num_train_samples) + 2 * reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

