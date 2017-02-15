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
  [D,C] = W.shape
  N = X.shape[0]
    
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  loss_i = np.zeros((N,1)) 
  for iT in range(N):    
    # Scores f of dimension 1xC
    scores = X[iT,:].dot(W)
        
    # Normalization trick to avoid numerical instability
    VAL = max(scores)    
    scores -= VAL
    
    sum_scores = sum(np.exp(scores))
    
    # L_i = -log {\frac{e^{f(x_i)_{y_i}}} {\sum_j e^{f(x_i)_j}}} = 
    # - f(x_i)_{y_i} + log \sum_j e^{f(x_i)_j}            
    loss_i[iT] = -scores[y[iT]] + np.log(sum_scores) 

    # Computation of the gradient for the current example
    # dw_j = 1/num_train * \sum_i[x_i * (p(y_i = j)-Ind{y_i = j} )]    
    dLi = np.zeros(W.shape)    
    for iC in range(C):
        prob = np.exp( scores[iC]) / sum_scores
        dLi[:,iC] = (prob - (iC == y[iT])) * X[iT,:]

    dW += dLi
  
  # Regularization
  loss = loss_i.mean() + 0.5 * reg * np.sum(W * W)  
  dW = (1.0/N) * dW + reg * W 
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
  C = W.shape[1]
  N = X.shape[0]
  scores = X.dot(W)
  
  # Normalization trick to avoid numerical instability  
  VAL = np.max(scores, axis=1)
  scores = scores - np.tile(VAL,(C,1)).T
            
  # Compute the Normalized Scores     
  prob = np.exp( scores ) 
  summ = np.sum(prob, axis=1)
  summ = np.tile(summ, (C,1))
    
  prob = prob / summ.T
          
  # We have to extract the correct class for each example  
  correct_class_score = prob[range(N),y]

  loss = -np.log ( correct_class_score )   
  loss = loss.mean() + 0.5 * reg * np.sum(W * W) 

  ind = np.zeros(prob.shape)
  ind[range(N),y] = 1

  dW = X.T.dot((prob - ind)) 
  dW = (1.0/N) * dW + reg * W

  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

