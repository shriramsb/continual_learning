import numpy as np
import tensorflow as tf

class Network(object):
	def __init__(self):
		self.layers = []
		self.batch_norm_layers = []
		self.num_residual_blocks = 5
		self.createLayers()
		
		self.bn_momentum = 0.9
		self.bn_eps = 1e-5

	def residualBlock(self, l, is_training, increase_dim=False, projection=False, last=False):
		input_num_filters = l.shape[-1]

		if increase_dim:
			first_stride = (2, 2)
			out_num_filters = input_num_filters * 2
		else:
			first_stride = (1, 1)
			out_num_filters = input_num_filters

		stack_1 = tf.nn.relu(tf.layers.batch_normalization(tf.layers.conv2d(l, filters=out_num_filters, kernel_size=(3, 3), strides=first_stride, padding='same', use_bias=False), training=is_training, momentum=self.bn_momentum, epsilon=self.bn_eps))
		stack_2 = tf.layers.batch_normalization(tf.layers.conv2d(stack_1, filters=out_num_filters, kernel_size=(3, 3), padding='same', use_bias=False), training=is_training, momentum=self.bn_momentum, epsilon=self.bn_eps)

		if increase_dim:
			if projection:
				projection = tf.layers.batch_normalization(tf.layers.conv2d(l, filters=out_num_filters, kernel_size=(1, 1), strides=(2, 2), padding='same', use_bias=False), training=is_training, momentum=self.bn_momentum, epsilon=self.bn_eps)
				if last:
					block = stack_2	+ projection
				else:
					block = tf.nn.relu(stack_2 + projection)
			else:
				#################### have to see here ########################
				if last:
					block = stack_2
				else:
					block = tf.nn.relu(stack_2)
				#################### have to see here ########################
		else:
			#################### need to change last ################# pytorch resnet doesn't consider 'last'
			if last:
				block = stack_2 + l
			else:
				block = tf.nn.relu(stack_2 + l)

		return block



	# See what data_format='channels_last' mean in MaxPooling2D ?
	def createLayers(self):
		pass

	def forward(self, x, apply_dropout, keep_prob_input=1.0, keep_prob_hidden=1.0, is_training=False):
		with tf.variable_scope('resnet_layers'):
			layer_output = []
			
			y = x
			
			y = tf.layers.batch_normalization(tf.layers.conv2d(y, filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same', use_bias=False), training=is_training, momentum=self.bn_momentum, epsilon=self.bn_eps)
			y = tf.nn.relu(y)
			layer_output.append(y)

			# first stack 32 x 32 x 16
			for _ in range(self.num_residual_blocks):
				y = self.residualBlock(y, is_training=is_training)
				layer_output.append(y)

			# second stack 16 x 16 x 32
			y = self.residualBlock(y, is_training=is_training, increase_dim=True, projection=True)
			layer_output.append(y)
			for _ in range(1, self.num_residual_blocks):
				y = self.residualBlock(y, is_training=is_training)
				layer_output.append(y)

			# third stack 8 x 8 x 64
			y = self.residualBlock(y, is_training=is_training, increase_dim=True, projection=True)
			layer_output.append(y)
			for _ in range(1, self.num_residual_blocks - 1):
				y = self.residualBlock(y, is_training=is_training)
				layer_output.append(y)

			y = self.residualBlock(y, is_training=is_training, last=False) 			# edit me! set last=True
			layer_output.append(y)

			# global average pooling
			y = tf.reduce_mean(y, axis=[1, 2])
			layer_output.append(y)

			y = tf.layers.dense(y, units=100)
			layer_output.append(y)
			
			return y, layer_output

	def getLayerVariables(self):
		# l = []
		# for i in range(len(self.layers)):
		# 	l.extend(self.layers[i].variables)
		# return l
		return [v for v in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='resnet_layers')]

	def name(self):
		return 'n1'

	def appendResidualBlock(self, l, increase_dim=False, projection=False, last=False):
		pass
		# input_num_filters = l.shape[-1]

		# if increase_dim:
		# 	first_stride = (2, 2)
		# 	out_num_filters = input_num_filters * 2
		# else:
		# 	first_stride = (1, 1)
		# 	out_num_filters = input_num_filters

		# self.layers.append(tf.layers.conv2D(filters=out_num_filters, kernel_size=(3, 3), strides=first_stride, padding='same', activation=tf.nn.relu))
		# self.layers.append(tf.layers.conv2D(filters=out_num_filters, kernel_size=(3, 3), padding='same'))

		# self.batch_norm_layers.append(tf.layers.BatchNormalization())
		# self.batch_norm_layers.append(tf.layers.BatchNormalization())

		# stack_1 = self.batch_norm_layers[-2](self.layers[-2](l))
		# stack_2 = self.batch_norm_layers[-1](self.layers[-1](stack_1))

		# if increase_dim:
		# 	if projection:
		# 		self.layers.append(tf.layers.conv2D(filters=out_num_filters, kernel_size=(1, 1), strides=(2, 2), padding='same'))
		# 		self.batch_norm_layers.append(tf.layers.BatchNormalization())
		# 		projection = self.batch_norm_layers[-1](self.layers[-1](l))
		# 		if last:
		# 			block = stack_2 + projection
		# 		else:
		# 			block = tf.nn.relu(stack_2 + projection)
		# 	else:
		# 		#################### have to see here ########################
		# 		if last:
		# 			block = stack_2
		# 		else:
		# 			block = tf.nn.relu(stack_2)
		# 		#################### have to see here ########################
		# else:
		# 	if last:
		# 		block = stack_2 + l
		# 	else:
		# 		block = tf.nn.relu(stack_2 + l)

		# return block