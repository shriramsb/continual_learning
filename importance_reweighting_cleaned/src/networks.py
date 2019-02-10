import numpy as np
import tensorflow as tf

class ResNet32(object):
	def __init__(self):
		self.layers = []
		self.batch_norm_layers = []
		self.num_residual_blocks = 5
		
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


	def forward(self, x, hparams=None):
		is_training = hparams['is_training']
		output_shape = hparams['output_shape']
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

			if (hparams is None or hparams['use_relu_last']):
				y = self.residualBlock(y, is_training=is_training, last=False)
			else:
				y = self.residualBlock(y, is_training=is_training, last=True)
			layer_output.append(y)

			# global average pooling
			y = tf.reduce_mean(y, axis=[1, 2])
			layer_output.append(y)

			if (hparams is None or not hparams['cosine_classifier']):
				y = tf.layers.dense(y, units=output_shape[0])
			else:
				in_channels = 64
				limit = np.sqrt(6 / (in_channels + output_shape[0]))
				W = tf.Variable(np.random.uniform(-limit, limit, (in_channels, out_channels)), dtype=tf.float32)
				init_scale_val = 10.0 if ('scale_scores_init' not in hparams.keys()) else hparams['scale_scores_init']
				scale_cls = tf.Variable(init_scale_val, dtype=tf.float32)
				y = scale_cls * tf.linalg.matmul(y, W)
			layer_output.append(y)
			
			return y, layer_output

	def getLayerVariables(self):
		return None

	def name(self):
		return 'ResNet32'

	# Holds only for resnet
	def getPenultimateLayerVariables(self):
		return [v for v in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='resnet_layers') if 'dense' in v.name]