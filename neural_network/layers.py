import numpy as np
from . import utils

class Layer:

	def create(self, input_size):
		pass

	def forward(self, input):
		pass

	def backward(self, gradient):
		pass


class ReLU(Layer):

	def create(self, input_size):
		self.input_size = input_size
		self.output_size = input_size

	def forward(self, input):
		self.input = input
		return np.maximum(0, input)

	def backward(self, gradient):
		return gradient * (self.input > 0)


class Tanh(Layer):

	def create(self, input_size):
		self.input_size = input_size
		self.output_size = input_size

	def forward(self, input):
		self.input = input
		return np.tanh(input)

	def backward(self, gradient):
		return gradient * (1 - np.tanh(self.input) ** 2)


class Softmax(Layer):

	def create(self, input_size):
		self.input_size = input_size
		self.output_size = input_size

	def forward(self, input):
		self.input = input
		b = input.max()
		e = np.exp(input - b)
		return e / e.sum()

	def backward(self, gradient):
		return gradient.copy()


class Input(Layer):

	def __init__(self, input_size):
		self.input_size = input_size
		self.output_size = input_size

	def forward(self, input):
		return input.copy()

	def backward(self, gradient):
		return gradient.copy()


class Parameter(Layer):

	def init(self, next_layer):
		pass


class Linear(Parameter):

	def __init__(self, nb_neurons):
		self.output_size = nb_neurons

	def create(self, input_size):
		self.input_size = input_size
		self.weights = [np.zeros((self.output_size, input_size)), np.zeros((self.output_size))]
		self.velocities = [np.zeros((self.output_size, input_size)), np.zeros((self.output_size))]
		self.m = [np.zeros((self.output_size, input_size)), np.zeros((self.output_size))]
		self.v = [np.zeros((self.output_size, input_size)), np.zeros((self.output_size))]
		self.gradients = [np.zeros((self.output_size, input_size)), np.zeros((self.output_size))]

	def init(self, next_layer):
		if type(next_layer) == ReLU:
			utils.kaiming_init(self.weights)
		else:
			utils.glorot_init(self.weights)

	def forward(self, input):
		self.input = input
		return self.weights[0] @ input + self.weights[1]

	def backward(self, gradient):
		self.gradients[0] += np.outer(gradient, self.input)
		self.gradients[1] += gradient
		return self.weights[0].T @ gradient
