import numpy as np
import random

def softmax(x):
    """
    Compute the softmax function for each row of the input x.
    """
    axis = len(x.shape) - 1
    x -= np.max(x, axis=axis, keepdims=True)
    x = np.exp(x) / np.sum(np.exp(x), axis=axis, keepdims=True)
    
    return x

def test_softmax_basic():
    """
    Some simple tests to get you started. 
    Warning: these are not exhaustive.
    """
    print("Running basic tests...")
    test1 = softmax(np.array([1,2]))
    print(test1)
    assert np.amax(np.fabs(test1 - np.array(
        [0.26894142,  0.73105858]))) <= 1e-6

    test2 = softmax(np.array([[1001,1002],[3,4]]))
    print(test2)
    assert np.amax(np.fabs(test2 - np.array(
        [[0.26894142, 0.73105858], [0.26894142, 0.73105858]]))) <= 1e-6

    test3 = softmax(np.array([[-1001,-1002]]))
    print(test3)
    assert np.amax(np.fabs(test3 - np.array(
        [0.73105858, 0.26894142]))) <= 1e-6

    print("You should verify these results!\n")

def test_softmax():
    """ 
    Use this space to test your softmax implementation by running:
        python q1_softmax.py 
    This function will not be called by the autograder, nor will
    your tests be graded.
    """
    print("Running your tests...")
    ### YOUR CODE HERE
    raise NotImplementedError
    ### END YOUR CODE  

if __name__ == "__main__":
    test_softmax_basic()
    #test_softmax()