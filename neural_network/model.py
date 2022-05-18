from copy import deepcopy
from .layers import Layer, Input, Softmax, Parameter
from .optimizers import Optimizer
from .losses import Loss, NegativeLogLikelihood
from . import utils
import math

class Model:

	def __init__(self):
		self.layers = []

	def add(self, layer):

		if not isinstance(layer, Layer):
			raise TypeError("Layer must be an instance of Layer")

		if len(self.layers) > 0:
			if type(layer) == Input:
				raise Exception("Input layer cannot be added after other layers")
			layer.create(self.layers[-1].output_size)
		elif type(layer) != Input:
			raise Exception("First layer must be an input layer")
		else:
			self.input_size = layer.input_size

		self.layers.append(layer)

	def compile(self, loss, optimizer):

		if not isinstance(optimizer, Optimizer):
			raise TypeError("Optimizer must be an instance of Optimizer")
		if not isinstance(loss, Loss):
			raise TypeError("Loss must be an instance of Loss")

		self.optimizer = optimizer
		self.loss = loss

		for i in range(len(self.layers) - 1):
			if type(self.layers[i]) == Softmax:
				raise Exception("Softmax layer must be the last layer")
			elif isinstance(self.layers[i], Parameter):
				self.layers[i].init(self.layers[i + 1])

		if isinstance(self.layers[-1], Parameter):
			self.layers[-1].init(None)

		self.output_size = self.layers[-1].output_size

	def forward(self, input):

		for layer in self.layers:
			input = layer.forward(input)

		return input

	def backward(self, gradient):

		for layer in reversed(self.layers):
			gradient = layer.backward(gradient)

	def clear_gradients(self):

		for layer in self.layers:
			if isinstance(layer, Parameter):
				for gradient in layer.gradients:
					gradient[:] = 0

	def average_gradients(self, batch_size):

		for layer in self.layers:
			if isinstance(layer, Parameter):
				for gradient in layer.gradients:
					gradient /= batch_size

	def check_input(self, x, y):

		nb_data = x.shape[0]

		if y.shape[0] != nb_data:
			raise Exception("Number of labels must be equal to the number of data")

		x_copy = utils.flatten(x.copy())
		y_copy = utils.flatten(y.copy())

		if x_copy.shape[1] != self.input_size:
			raise Exception("Features size must be equal to the input size of the model")

		if type(self.loss) == NegativeLogLikelihood:
			if y_copy.shape[1] != self.output_size and y_copy.shape[1] != 1:
				raise Exception("Labels size must equal to 1 or the output size of the model")
			if y_copy.shape[1] == self.output_size:
				y_copy = y_copy.argmax(axis = 1)
		else:
			if y_copy.shape[1] != self.output_size:
				raise Exception("Labels size must be equal to the output size of the model")

		return x_copy, y_copy

	def train(self, x_train, y_train, epochs, batch_size, x_val = None, y_val = None, print_frequency = 1):

		x_train_copy, y_train_copy = self.check_input(x_train, y_train)

		if x_val is not None:
			x_val_copy, y_val_copy = self.check_input(x_val, y_val)

		nb_data = x_train_copy.shape[0]
		best_dev = 0
		best_model = deepcopy(self)
		self.optimizer.init()

		for epoch in range(epochs):

			utils.shuffle(x_train_copy, y_train_copy)

			for batch in range(math.floor(nb_data / batch_size)):

				loss = 0
				accuracy = 0
				self.clear_gradients()

				for i in range(batch * batch_size, (batch + 1) * batch_size):

					# Forward
					output = self.forward(x_train_copy[i])
					loss += self.loss.forward(output, y_train_copy[i])

					if output.argmax() == y_train_copy[i]:
						accuracy += 1

					# Backward
					gradient = self.loss.backward(output, y_train_copy[i])
					self.backward(gradient)

				# Update
				self.average_gradients(batch_size)
				self.optimizer.update(self.layers)

				# Tests
				if (batch + 1) % print_frequency == 0:

					loss /= batch_size
					accuracy /= batch_size

					if x_val_copy is not None:

						val_accuracy, val_loss = self.test(x_val_copy, y_val_copy, False, False)

						# Save the best model
						if val_accuracy > best_dev :
							best_model = deepcopy(self)
							best_dev = val_accuracy

						msg = "Epoch %i | batch %i | train loss: %.2f | train accuracy: %.1f%% | val loss: %.2f | val accuracy: %.1f%%   " % (epoch + 1, batch + 1, loss, accuracy * 100., val_loss, val_accuracy * 100.)

					else:
						msg = "Epoch %i | batch %i | train loss: %.2f | train accuracy: %.1f%%   " % (epoch + 1, batch + 1, loss, accuracy * 100.)

					if batch == int(nb_data / batch_size) - 1: print(msg)
					else: print(msg, end = "\r")

			self.optimizer.iteration()

		if x_val_copy is not None:
			self.layers = best_model.layers

	def predict(self, x):
		output = self.forward(x)
		return output.argmax()

	def test(self, x, y, check_data = True, print_results = True):

		if check_data:
			x_copy, y_copy = self.check_input(x, y)
		else:
			x_copy, y_copy = x, y

		nb_data = x.shape[0]
		loss = 0
		accuracy = 0

		for i in range(nb_data):

			output = self.forward(x_copy[i])
			loss += self.loss.forward(output, y_copy[i])

			if output.argmax() == y_copy[i]:
				accuracy += 1

		if print_results:
			print("Test loss: %.2f | test accuracy: %.1f%%" % (loss / nb_data, (accuracy / nb_data) * 100.))

		return accuracy / nb_data, loss / nb_data
