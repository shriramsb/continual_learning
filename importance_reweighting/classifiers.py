import os
import numpy as np
import tensorflow as tf

from network import Network

# Helps in doing single step of training on the Network object which it has
class Classifier(object):
	def __init__(self, network, input_shape, output_shape, checkpoint_path):
		self.x = None                   		# tf.placeholder for training inputs
		self.y = None 					
		self.keep_prob_input = None     		#  tf.placeholder for dropout values
		self.keep_prob_hidden = None

		self.loss_weights = None 				# weight for each example in a batch in loss function - used in importance reweighting

		self.input_shape = input_shape 			# input dimensions to the network
		self.output_shape = output_shape 		# output dimensions to the network
		self.checkpoint_path = checkpoint_path 	

		self.createPlaceholders()

		# hyperparameters
		self.learning_rate = 5e-6
		self.apply_dropout = True
		self.dropout_input_prob = 1.0
		self.dropout_hidden_prob = 1.0

		self.network = network 					# Network object
		# feed-forward for training inputs
		self.scores, self.layer_output = self.network.forward(self.x, self.apply_dropout, self.keep_prob_input, self.keep_prob_hidden)

		self.theta = None 						# list of tf trainable variables used to hold values of current task

		self.loss = None 						# tf Tensor - loss
		self.accuracy = None 					# tf Tensor - accuracy
		self.createLossAccuracy() 				# computation graph for calculating loss and accuracy

		self.saver = tf.train.Saver(max_to_keep=100, keep_checkpoint_every_n_hours=1, var_list=self.theta)
	
	# creates tensorflow placeholders for training
	def createPlaceholders(self):
		with tf.name_scope("prediction-inputs"):
			self.x = tf.placeholder(tf.float32, [None] + list(self.input_shape), name='x-input')
			self.y = tf.placeholder(tf.float32, [None] + list(self.output_shape), name='y-input')
		with tf.name_scope("dropout-probabilities"):
			self.keep_prob_input = tf.placeholder(tf.float32)
			self.keep_prob_hidden = tf.placeholder(tf.float32)
		with tf.name_scope("loss-weights"):
			self.loss_weights = tf.placeholder(tf.float32, [None])

	# create computation graph for loss and accuracy
	def createLossAccuracy(self):
		with tf.name_scope("loss"):
			# improve : try just softmax_cross_entropy instead of sotmax_cross_entropy_with_logits?
			average_nll = tf.reduce_sum(self.loss_weights * tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.y)) / tf.reduce_sum(self.loss_weights)
			self.loss = average_nll
		with tf.name_scope('accuracy'):
			accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.scores, 1), tf.argmax(self.y, 1)), tf.float32))
			self.accuracy = accuracy

	# setup train step ; to be called just before start of training
	def prepareForTraining(self, sess, model_name, model_init_name):
		self.train_step = self.createTrainStep()
		init = tf.global_variables_initializer()
		sess.run(init)
		if model_init_name:
			print("Restoring paramters from %s" % (model_init_name, ))
			self.restoreModel(sess, model_init_name)
	
	# create optimizer, loss function for training
	def createTrainStep(self):
		with tf.name_scope("optimizer"):
			self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
			return self.optimizer.minimize(self.loss, var_list=self.theta)
	
	# restore model with name : model_name
	def restoreModel(self, sess, model_name):
		ckpt = tf.train.get_checkpoint_state(checkpoint_dir=self.checkpoint_path, latest_filename=model_name)   # doubt : meaning of this?
		self.saver.restore(sess=sess, save_path=ckpt.model_checkpoint_path)

	# save weights with name : time_step, model_name
	def saveWeights(self, time_step, sess, model_name):
		if not os.path.exists(self.checkpoint_path):
			os.makedirs(self.checkpoint_path)
		self.saver.save(sess=sess, save_path=self.checkpoint_path + model_name + '.ckpt', global_step=time_step,
						latest_filename=model_name)     # meaning of this?
		print('saving model ' + model_name + ' at time step ' + str(time_step))

	# get accuracy, given batch in feed_dict
	def evaluate(self, sess, feed_dict):
		if self.apply_dropout:
			feed_dict.update({self.keep_prob_input: 1.0, self.keep_prob_hidden: 1.0})
		accuracy = sess.run([self.loss, self.accuracy], feed_dict=feed_dict)
		return accuracy

	# get predictions, given batch in feed_dict
	def getPredictions(self, sess, feed_dict):
		if self.apply_dropout:
			feed_dict.update({self.keep_prob_input: 1.0, self.keep_prob_hidden: 1.0})
		scores, y = sess.run([self.scores, self.y], feed_dict=feed_dict)
		return scores, y

	# Make single iteration of train, given input batch in feed_dict
	def singleTrainStep(self, sess, feed_dict):
		_, loss = sess.run([self.train_step, self.loss], feed_dict=feed_dict)
		return loss

	# creat feed_dict using batch_xs, batch_ys
	def createFeedDict(self, batch_xs, batch_ys, weights=None):
		if weights is None:
			weights = np.array([1 for _ in range(batch_xs.shape[0])])
		feed_dict = {self.x: batch_xs, self.y: batch_ys, self.loss_weights: weights}
		if self.apply_dropout:
			feed_dict.update({self.keep_prob_hidden: self.dropout_hidden_prob, self.keep_prob_input: self.dropout_input_prob})
		return feed_dict

	# set hyperparamters
	def updateHparams(self, hparams):
		for k, v in hparams.items():
			if (not isinstance(k, str)):
				raise Exception('Panic! hyperparameter key not string')
			setattr(self, k, v)

	# get output of layer just before logits
	def getPenultimateOutput(self, sess, feed_dict):
		penultimate_output = sess.run(self.layer_output[-2], feed_dict=feed_dict)
		return penultimate_output