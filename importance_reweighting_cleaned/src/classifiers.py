import os
import numpy as np
import tensorflow as tf
from copy import deepcopy
import sys

# Helps in doing single step of training on the Network object which it has
class Classifier(object):
	def __init__(self, network, input_shape, output_shape, classifier_params, network_params):
		self.x = None                   		# tf.placeholder for training inputs
		self.y = None 					
		self.keep_prob_input = None     		#  tf.placeholder for dropout values
		self.keep_prob_hidden = None

		self.loss_weights = None 				# weight for each example in a batch in loss function - used in importance reweighting

		self.input_shape = input_shape 			# input dimensions to the network
		self.output_shape = output_shape 		# output dimensions to the network

		self.learning_rate = None
		self.is_training = None
		self.createPlaceholders()

		# mask unwanted scores ; simulates increasing softmax outputs when new class appears
		# scores_mask - variable, scores_mask_placeholder used to assign value to scores_mask
		self.scores_mask = None 				
		self.scores_mask_placeholder = None 	
		self.scores_mask_assign_op = None 		
		self.createScoresMask() 				

		# extracting previous task's output from network for distillation
		self.distill_mask = None 				
		self.distill_mask_placeholder = None 	
		self.distill_mask_assign_op = None 		
		self.scores_distill_size = 0
		self.createDistillMask() 				

		# hyperparameters
		self.hparams = {}
		self.hparams['momentum'] = 0.9
		self.hparams['reg'] = 0.0
		self.hparams['dropout_prob'] = (1.0, 1.0)
		self.hparams['T'] = self.hparams['alpha'] = None
		self.hparams['reg_type'] = network_params['reg_type']
		self.updateHparams(classifier_params)


		self.network = network 					# Network object
		# feed-forward for training inputs
		network_params = deepcopy(network_params)
		network_params.update({'dropout' : (self.keep_prob_input, self.keep_prob_hidden), 'output_shape' : output_shape, 
								'is_training' : self.is_training})
		self.scores, self.layer_output = self.network.forward(self.x, hparams=network_params)
		self.scores_distill = tf.boolean_mask(self.layer_output[-1], self.distill_mask, axis=1)
		self.scores = tf.boolean_mask(self.scores, self.scores_mask, axis=1)

		# trainable variables in network 
		# CHECK : currently returns None; using None uses all variables in the whole program in optimizer
		self.theta = self.network.getLayerVariables()

		# CHECK EXECUTE if expression of reg_loss is correctly uses variables of only type 'kernel'
		self.loss = None 								
		self.reg_loss = None 							
		self.accuracy = None 							
		self.createLossAccuracy(classifier_params['reweigh_points_loss'], classifier_params['T'], classifier_params['alpha'])

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
	def createLossAccuracy(self, reweigh_points_loss, T=None, alpha=None):
		with tf.name_scope("loss"):
			# improve : try just softmax_cross_entropy instead of sotmax_cross_entropy_with_logits?
			if (self.hparams['loss_type'] == 'cross-entropy'):
				if (reweigh_points_loss):
					true_target_loss = tf.reduce_sum(self.loss_weights * tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.y)) / tf.reduce_sum(self.loss_weights)
				else:
					true_target_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.y))
			
			# no support for not reweigh_points_loss in 'svm'
			elif (self.hparams['loss_type'] == 'svm'):
				if (not reweigh_points_loss):
					true_logit_indices = tf.where(tf.equal(self.y, 1))
					true_logits = tf.gather_nd(self.scores, true_logit_indices)
					true_target_loss = tf.reduce_mean(tf.reduce_sum(tf.nn.relu(self.scores - tf.expand_dims(true_logits, dim=1) + 1), axis=1)) - 1
				else:
					print("createLossAccuracy : feature not supported. Exiting...")
					sys.exit(0)

			if T is not None:
				student_log_prob_old = tf.nn.log_softmax(self.scores_distill / T)
				teacher_prob_old = tf.nn.softmax(self.teacher_outputs / T)
				distill_loss = -1 * tf.reduce_mean(tf.reduce_sum(student_log_prob_old * teacher_prob_old, axis=-1))

			if T is None:
				self.loss = true_target_loss
			else:
				self.loss = (1 - alpha) * true_target_loss + alpha * (T ** 2) * distill_loss

			if (self.hparams['reg_type'] == 'l2'):
				self.reg_loss = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables() if ('bias' not in v.name and 'batch' not in v.name)])
			elif (self.hparams['reg_type'] == 'l1'):
				self.reg_loss = tf.add_n([tf.math.reduce_sum(tf.math.abs(v)) for v in tf.trainable_variables() if ('bias' not in v.name and 'batch' not in v.name)])
			else:
				print("bad network params reg_type. Exiting...")
				sys.exit(0)
		with tf.name_scope('accuracy'):
			accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.scores, 1), tf.argmax(self.y, 1)), tf.float32))
			self.accuracy = accuracy

	# setup train step ; to be called just before start of training
	def prepareForTraining(self, sess, model_init_path, only_penultimate_train=False):
		self.train_step = self.createTrainStep(sess, only_penultimate_train)
		init = tf.global_variables_initializer()
		sess.run(init)
		if model_init_path:
			print("Restoring paramters from %s" % (model_init_path, ))
			self.restoreModel(sess, model_init_path)
	
	# create optimizer, loss function for training
	def createTrainStep(self, sess, only_penultimate_train=False):
		with tf.name_scope("optimizer"):
			self.optimizer = tf.train.MomentumOptimizer(learning_rate=self.learning_rate, momentum=self.hparams['momentum'])
			update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
			with tf.control_dependencies(update_ops):
				if (not only_penultimate_train):
					ret_val = self.optimizer.minimize(self.loss + self.hparams['reg'] * self.reg_loss, var_list=self.theta)
				else:
					ret_val = self.optimizer.minimize(self.loss + self.hparams['reg'] * self.reg_loss, var_list=self.network.getPenultimateLayerVariables())
		init_optimizer_op = tf.initializers.variables(self.optimizer.variables())
		sess.run(init_optimizer_op)
		return ret_val
		
	
	# restore model with name : model_name
	def restoreModel(self, sess, model_path):
		ckpt = tf.train.get_checkpoint_state(checkpoint_dir=model_path, latest_filename='model')   # doubt : meaning of this?
		self.saver.restore(sess=sess, save_path=ckpt.model_checkpoint_path)

	# save weights with name : time_step, model_name
	def saveWeights(self, time_step, sess, model_path):
		self.saver.save(sess=sess, save_path=os.path.join(model_path, 'model.ckpt'), global_step=time_step,
						latest_filename='model')     # meaning of this?
		print('saving model at ' + model_path + ' at time step ' + str(time_step))

	# get accuracy, given batch in feed_dict
	def evaluateLossAccuracy(self, sess, feed_dict):
		feed_dict.update({self.keep_prob_input: 1.0, self.keep_prob_hidden: 1.0})
		loss, accuracy = sess.run([self.loss, self.accuracy], feed_dict=feed_dict)
		return loss, accuracy

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
	def createFeedDict(self, batch_xs, batch_ys, is_training=False, learning_rate=None, teacher_outputs=None, weights=None):
		if weights is None:
			weights = np.array([1 for _ in range(batch_xs.shape[0])])
		feed_dict = {self.x: batch_xs, self.y: batch_ys, self.loss_weights: weights}
		if learning_rate is not None:
			feed_dict.update({self.learning_rate: learning_rate})
		
		feed_dict.update({self.keep_prob_hidden: self.hparams['dropout_prob'][1], self.keep_prob_input: self.hparams['dropout_prob'][0]})
		
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
			self.hparams[k] = hparams[k]

	# get output of layer just before logits
	def getLayerOutput(self, sess, feed_dict, index):
		penultimate_output = sess.run(self.layer_output[index], feed_dict=feed_dict)
		return penultimate_output