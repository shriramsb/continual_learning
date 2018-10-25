import numpy as np
import tensorflow as tf

class Network(object):
	def __init__(self):
		self.layers = None			# list of layers
		self.createLayers()			# populate self.layers with network architecture

	# specify network architecture
	def createLayers(self):
		self.layers = []
		self.layers.append(tf.layers.Dense(units=500, activation=tf.nn.relu))
		self.layers.append(tf.layers.Dense(units=500, activation=tf.nn.relu))
		self.layers.append(tf.layers.Dense(units=10))

	# returns tensor at each point in forward propagation of input tensor x
	def forward(self, x, apply_dropout, keep_prob_input=1.0, keep_prob_hidden=1.0):
		layer_output = []
		input_shape = np.prod(x.shape.as_list()[1:])
		x = tf.reshape(x, [-1, input_shape])
		if (apply_dropout):
			x = tf.nn.dropout(x, keep_prob_input)
		y = x
		for i in range(len(self.layers) - 1):
			y = self.layers[i](y)
			if (apply_dropout):
				y = tf.nn.dropout(y, keep_prob_hidden)
			layer_output.append(y)
		y = self.layers[-1](y)
		layer_output.append(y)
		return y, layer_output

	# returns all variables in a list
	def getLayerVariables(self):
		l = []
		for i in range(len(self.layers)):
			l.extend(self.layers[i].variables)
		return l

	# name of the network : better to represent its architecture
	def name(self):
		return 'fc500_fc500_fc10'