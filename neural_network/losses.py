import numpy as np

class Loss:

	def __init__(self):
		pass

	def forward(self, output, target):
		pass

	def backward(self, output, target):
		pass


class NegativeLogLikelihood(Loss):

	def forward(self, output, target):
		return -np.log(output[target])

	def backward(self, output, target):
		gradient = output.copy()
		gradient[target] -= 1
		return gradient


class MeanSquaredError(Loss):

	def forward(self, output, target):
		return np.mean((output - target) ** 2)

	def backward(self, output, target):
		return output - target


class MeanAbsoluteError(Loss):

	def forward(self, output, target):
		return np.mean(np.abs(output - target))

	def backward(self, output, target):
		return output - target
