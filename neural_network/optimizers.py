import numpy as np
from .layers import Parameter

class Optimizer:

	def init(self):
		pass

	def update(self, layers):
		pass

	def iteration(self):
		pass


class SGD(Optimizer):

	def __init__(self, learning_rate = 0.01, momentum = 0.9):
		self.learning_rate = learning_rate
		self.momentum = momentum

	def init(self):
		pass

	def update(self, layers):
		for layer in layers:
			if isinstance(layer, Parameter):
				for i in range(len(layer.weights)):
					layer.velocities[i] = (self.momentum * layer.velocities[i]) + ((1. - self.momentum) * layer.gradients[i])
					layer.weights[i] -= layer.velocities[i] * self.learning_rate

	def iteration(self):
		pass


class Adam(Optimizer):

	def __init__(self, learning_rate = 0.001, beta_1 = 0.9, beta_2 = 0.999, epsilon = 1e-8):
		self.learning_rate = learning_rate
		self.beta_1 = beta_1
		self.beta_2 = beta_2
		self.epsilon = epsilon
		self.t = 1

	def init(self):
		self.t = 1

	def update(self, layers):
		for layer in layers:
			if isinstance(layer, Parameter):
				for i in range(len(layer.weights)):
					layer.m[i] = self.beta_1 * layer.m[i] + (1. - self.beta_1) * layer.gradients[i]
					layer.v[i] = self.beta_2 * layer.v[i] + (1. - self.beta_2) * layer.gradients[i] ** 2
					m_hat = layer.m[i] / (1. - (self.beta_1 ** self.t))
					v_hat = layer.v[i] / (1. - (self.beta_2 ** self.t))
					layer.weights[i] -= self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)

	def iteration(self):
		self.t += 1
