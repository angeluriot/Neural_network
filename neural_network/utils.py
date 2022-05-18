import numpy as np

def glorot_init(weights):
	range = np.sqrt(6. / (weights[0].shape[0] + weights[0].shape[1]))
	weights[0] = np.random.uniform(-range, range, size = weights[0].shape)
	weights[1] = np.zeros(weights[1].shape)

def kaiming_init(weights):
	range = np.sqrt(6. / weights[0].shape[1])
	weights[0] = np.random.uniform(-range, range, size = weights[0].shape)
	weights[1] = np.zeros(weights[1].shape) + 0.01

def shuffle(x, y):
	index_list = np.array([i for i in range(x.shape[0])])
	np.random.shuffle(index_list)
	x = x[index_list]
	y = y[index_list]

def flatten(x):
	return x.reshape(x.shape[0], -1)
