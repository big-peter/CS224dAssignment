import numpy as np


N = 3
d = 2
inMat = np.random.randn(N, d)

def softmax(inMat):
	assert len(inMat) > 1
	inMat -= np.max(inMat, axis=1, keepdims=True)
	inMat = np.exp(inMat) / np.sum(np.exp(inMat), axis=1, keepdims=True)
	return inMat

def sigmoid(xVector):
	return 1 / (1 + np.exp(xVector))

def sigmoidGradient(xVector):
	gradient = sigmoid(xVector) * (1 - sigmoid(xVector))
	return gradient

#print(softmax(inMat))
xVector = np.random.randn(5)
print(sigmoidGradient(xVector))


