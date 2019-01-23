import os
import numpy as np
import tensorflow as tf

from network import Network

# Helps in doing single step of training on the Network object which it has
class Classifier(object):
	def __init__(self, network, input_shape, output_shape, checkpoint_path, reweigh_points_loss):
		self.x = None                   		# tf.placeholder for training inputs
		self.y = None 					
		self.keep_prob_input = None     		#  tf.placeholder for dropout values
		self.keep_prob_hidden = None

		self.loss_weights = None 				# weight for each example in a batch in loss function - used in importance reweighting

		self.input_shape = input_shape 			# input dimensions to the network
		self.output_shape = output_shape 		# output dimensions to the network
		self.checkpoint_path = checkpoint_path

		self.learning_rate = None
		self.is_training = None
		self.createPlaceholders()

		# mask unwanted scores ; simulates increasing softmax outputs when new class appears
		self.scores_mask = None 				# tf variable storing mask
		self.scores_mask_placeholder = None 	# tf placeholder to assign self.scores_mask
		self.scores_mask_assign_op = None 		# tf operation to assign scores_mask
		self.createScoresMask() 				# creates above objects for masking

		self.distill_mask = None 				# tf variable storing mask
		self.distill_mask_placeholder = None 	# tf placeholder to assign self.distill_mask
		self.distill_mask_assign_op = None 		# tf operation to assign scores_mask
		self.scores_distill_size = 0
		self.createDistillMask() 				# creates above objects for masking

		# hyperparameters
		# self.learning_rate = None
		self.momentum = 0.9
		self.reg = 0.0
		self.apply_dropout = True
		self.dropout_input_prob = 1.0
		self.dropout_hidden_prob = 1.0

		self.network = network 					# Network object
		# feed-forward for training inputs
		self.scores, self.layer_output = self.network.forward(self.x, self.apply_dropout, self.keep_prob_input, self.keep_prob_hidden, is_training=self.is_training)
		self.scores_distill = tf.boolean_mask(self.layer_output[-1], self.distill_mask, axis=1) 	# mask scores to get previous task's classes' scores
		self.scores = tf.boolean_mask(self.scores, self.scores_mask, axis=1) 	# mask scores to remove unassigned class's scores

		self.theta = self.network.getLayerVariables() 		# list of tf trainable variables used to hold values of current task

		self.loss = None 								# tf Tensor - loss
		self.l2_loss = None 							# tf Tensor - regularization term
		self.accuracy = None 							# tf Tensor - accuracy
		self.createLossAccuracy(reweigh_points_loss) 	# computation graph for calculating loss and accuracy

		self.saver = tf.train.Saver(max_to_keep=100, keep_checkpoint_every_n_hours=1, var_list=self.theta)
	
	# creates tensorflow placeholders for training
	def createPlaceholders(self):
		with tf.name_scope("prediction-inputs"):
			self.x = tf.placeholder(tf.float32, [None] + list(self.input_shape), name='x-input')
			self.y = tf.placeholder(tf.float32, [None] + [None for _ in range(len(self.output_shape))], name='y-input')
		with tf.name_scope("dropout-probabilities"):
			self.keep_prob_input = tf.placeholder(tf.float32)
			self.keep_prob_hidden = tf.placeholder(tf.float32)
		with tf.name_scope("loss-weights"):
			self.loss_weights = tf.placeholder(tf.float32, [None])
		with tf.name_scope("hprarms"):
			self.learning_rate = tf.placeholder(tf.float32, [])
		
		self.is_training = tf.placeholder(tf.bool, [])
		self.teacher_outputs = tf.placeholder(tf.float32, [None] + [None for _ in range(len(self.output_shape))])

	# creates self.scores_mask, op to assign it ; default mask set to using all outputs
	def createScoresMask(self):
		self.scores_mask = tf.Variable(np.ones(self.output_shape, dtype=np.bool), trainable=False)
		self.scores_mask_placeholder = tf.placeholder(tf.bool, shape=list(self.output_shape))
		self.scores_mask_assign = tf.assign(self.scores_mask, self.scores_mask_placeholder)

	# creates self.scores_mask, op to assign it ; default mask set to using all outputs
	def createDistillMask(self):
		self.distill_mask = tf.Variable(np.ones(self.output_shape, dtype=np.bool), trainable=False)
		self.distill_mask_placeholder = tf.placeholder(tf.bool, shape=list(self.output_shape))
		self.distill_mask_assign = tf.assign(self.distill_mask, self.distill_mask_placeholder)
	
	def setScoresMask(self, sess, mask):
		sess.run(self.scores_mask_assign, feed_dict={self.scores_mask_placeholder: mask})

	def setDistillMask(self, sess, mask):
		sess.run(self.distill_mask_assign, feed_dict={self.distill_mask_placeholder: mask})
		self.scores_distill_size = np.sum(mask)

	# create computation graph for loss and accuracy
	def createLossAccuracy(self, reweigh_points_loss, use_distill=False, T=None, alpha=None):
		with tf.name_scope("loss"):
			# improve : try just softmax_cross_entropy instead of sotmax_cross_entropy_with_logits?
			if (reweigh_points_loss):
				average_nll = tf.reduce_sum(self.loss_weights * tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.y)) / tf.reduce_sum(self.loss_weights)
			else:
				average_nll = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.y))

			if use_distill:
				student_log_prob_old = tf.nn.log_softmax(self.scores_distill / T)
				teacher_prob_old = tf.nn.softmax(self.teacher_outputs / T)
				distill_loss = -1 * tf.reduce_mean(tf.reduce_sum(student_log_prob_old * teacher_prob_old, axis=-1))

			if not use_distill:
				self.loss = average_nll
			else:
				self.loss = (1 - alpha) * average_nll + alpha * (T ** 2) * distill_loss
			self.l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables() if ('bias' not in v.name and 'batch' not in v.name)])
		with tf.name_scope('accuracy'):
			accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.scores, 1), tf.argmax(self.y, 1)), tf.float32))
			self.accuracy = accuracy

	# setup train step ; to be called just before start of training
	def prepareForTraining(self, sess, model_init_name, only_penultimate_train=False):
		self.train_step = self.createTrainStep(sess, only_penultimate_train)
		init = tf.global_variables_initializer()
		sess.run(init)
		if model_init_name:
			print("Restoring paramters from %s" % (model_init_name, ))
			self.restoreModel(sess, model_init_name)
	
	# create optimizer, loss function for training
	def createTrainStep(self, sess, only_penultimate_train=False):
		with tf.name_scope("optimizer"):
			self.optimizer = tf.train.MomentumOptimizer(learning_rate=self.learning_rate, momentum=self.momentum)
			update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
			with tf.control_dependencies(update_ops):
				if (not only_penultimate_train):
					ret_val = self.optimizer.minimize(self.loss + self.reg / 2 * self.l2_loss, var_list=self.theta)
				else:
					ret_val = self.optimizer.minimize(self.loss + self.reg / 2 * self.l2_loss, var_list=self.network.getPenultimateLayerVariables())
		init_optimizer_op = tf.initializers.variables(self.optimizer.variables())
		sess.run(init_optimizer_op)
		return ret_val
		
	
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
		_, loss, accuracy = sess.run([self.train_step, self.loss, self.accuracy], feed_dict=feed_dict)
		return loss, accuracy

	# creat feed_dict using batch_xs, batch_ys
	def createFeedDict(self, batch_xs, batch_ys, learning_rate=None, weights=None, is_training=False, teacher_outputs=None):
		if weights is None:
			weights = np.array([1 for _ in range(batch_xs.shape[0])])
		feed_dict = {self.x: batch_xs, self.y: batch_ys, self.loss_weights: weights}
		if learning_rate is not None:
			feed_dict.update({self.learning_rate: learning_rate})
		if self.apply_dropout:
			feed_dict.update({self.keep_prob_hidden: self.dropout_hidden_prob, self.keep_prob_input: self.dropout_input_prob})
		feed_dict.update({self.is_training: is_training})
		if (teacher_outputs is not None):
			feed_dict.update({self.teacher_outputs: teacher_outputs})
		else:
			if self.scores_distill_size > 0:
				feed_dict.update({self.teacher_outputs: np.zeros((batch_xs.shape[0], self.scores_distill_size))})
			else:
				feed_dict.update({self.teacher_outputs: np.zeros((batch_xs.shape[0], 1))})
		return feed_dict

	# set hyperparamters
	def updateHparams(self, hparams):
		for k, v in hparams.items():
			if (not isinstance(k, str)):
				raise Exception('Panic! hyperparameter key not string')
			if (k == 'learning_rate'):
				continue
			setattr(self, k, v)

	# get output of layer just before logits
	def getLayerOutput(self, sess, feed_dict, index):
		penultimate_output = sess.run(self.layer_output[index], feed_dict=feed_dict)
		return penultimate_output