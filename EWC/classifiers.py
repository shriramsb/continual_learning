import os
import tensorflow as tf

from network import Network

# Helps in doing single step of training on the Network object which it has. Maintains the value of EWC fisher information
class Classifier(object):
	def __init__(self, network, input_shape, output_shape, checkpoint_path):
		self.x = None                           # tf.plcaeholder for training inputs
		self.y = None

		self.x_fisher = None                    # tf.plcaeholder for calculation of fisher information
		self.y_fisher = None
		self.fisher_average_scale = None        # tf.plcaeholder of scaling importance of parameter after accumulating using training data

		self.keep_prob_input = None             #  tf.plcaeholder for dropout values
		self.keep_prob_hidden = None
		self.input_shape = input_shape          # input dimensions to the network
		self.output_shape = output_shape        # ouput dimension from the network
		self.checkpoint_path = checkpoint_path  
		self.ewc_batch_size = 100               # batch size for calculation of EWC

		self.createPlaceholders()

		# hyperparameters
		self.learning_rate = 5e-6
		self.fisher_multiplier = 0.0
		self.apply_dropout = True
		self.dropout_input_prob = 1.0
		self.dropout_hidden_prob = 1.0

		self.network = network                  # Network object
		# feed-forward for training inputs
		self.scores, self.layer_output = self.network.forward(self.x, self.apply_dropout, self.keep_prob_input, self.keep_prob_hidden)

		self.theta = None                       # list of tf trainable variables used to hold values of current task
		self.theta_lagged = None                # list of tf trainable variables used to hold values of previous task
		self.fisher_diagonal = None             # list of tf variable having importance of each variable
		self.createFisherDiagonal()             # create tf variables - fisher

		self.loss = None                        # tf Tensor - loss
		self.accuracy = None                    # tf Tensor - accuracy
		self.createLossAccuracy()               # computation graph for calculating loss and accuracy

		self.new_fisher_diagonal = None             # temporary tf variable for calculating fisher information at end of training
		self.zero_new_fisher_diagonal = None        # tf operation to zero out self.new_fisher_diagonal
		self.accumulate_squared_gradients = None    # tf operation to calcuate and accumulate fisher diagonal in self.new_fisher_diagonal
		self.fisher_full_batch_average_op = None    # tf operation to scale self.new_fisher_diagonal by self.fisher_average_scale
		self.update_theta_op = None                 # tf operation to update theta_lagged with theta for constraining next task
		self.update_with_new_fisher_diagonal_op = None  # tf operation to take average of parameter importances 
		self.createFisherOps()                      # computation graph for calculating fisher

		self.saver = tf.train.Saver(max_to_keep=100, keep_checkpoint_every_n_hours=1, var_list=self.theta + self.theta_lagged + self.fisher_diagonal)
	
	# creates tensorflow placeholders for training, fisher calculation input, averaging accumulated importance
	def createPlaceholders(self):
		with tf.name_scope("prediction-inputs"):
			self.x = tf.placeholder(tf.float32, [None] + list(self.input_shape), name='x-input')
			self.y = tf.placeholder(tf.float32, [None] + list(self.output_shape), name='y-input')
		with tf.name_scope("dropout-probabilities"):
			self.keep_prob_input = tf.placeholder(tf.float32)
			self.keep_prob_hidden = tf.placeholder(tf.float32)
		with tf.name_scope("fisher-inputs"):
			self.x_fisher = tf.placeholder(tf.float32, [self.ewc_batch_size] + list(self.input_shape))
			self.y_fisher = tf.placeholder(tf.float32, [self.ewc_batch_size] + list(self.output_shape))

		self.fisher_average_scale = tf.placeholder(tf.float32)

	# create tf variables for previous tasks' parameters and fisher diagonal
	def createFisherDiagonal(self):
		self.theta = self.network.getLayerVariables()
		self.theta_lagged = []
		self.fisher_diagonal = []
		# for each tf variable in network, create two variables of same size, for backed up parameter values and their importances
		for i in range(len(self.theta)):
			self.theta_lagged.append(tf.Variable(tf.constant(0.0, shape=self.theta[i].shape), trainable=False))
			self.fisher_diagonal.append(tf.Variable(tf.constant(0.0, shape=self.theta[i].shape), trainable=False))

	# create computation graph for loss and accuracy
	def createLossAccuracy(self):
		with tf.name_scope("loss"):
			average_nll = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.y))  # wat abt just softmax_cross_entropy
			self.loss = average_nll
		with tf.name_scope('accuracy'):
			accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.scores, 1), tf.argmax(self.y, 1)), tf.float32))
			self.accuracy = accuracy

	# create computation graph for calculating fisher - approximates sum(double derivatives) by sum(derivative^2)
	def createFisherOps(self):
		self.new_fisher_diagonal = []
		self.zero_new_fisher_diagonal = []
		# for each variable in network, create temporary variable for storing new fisher diagonal at the end of training a task
		# create tf operation for zeroing out each self.new_fisher_diagonal
		for i in range(len(self.theta)):
			self.new_fisher_diagonal.append(tf.Variable(tf.constant(0.0, shape=self.theta[i].shape), trainable=False))
			self.zero_new_fisher_diagonal.append(tf.assign(self.new_fisher_diagonal[i], tf.constant(0.0, shape=self.new_fisher_diagonal[i].shape)))     # just use .op here to avoid getting return value

		scores, _ = self.network.forward(self.x_fisher, apply_dropout=False)                        # prediction for current batch
		unaggregated_nll = tf.reduce_sum(-1 * self.y_fisher * tf.nn.log_softmax(scores), axis=1)    # nll for each input in batch
		self.accumulate_squared_gradients = []                      # list (sum_i of gradient^2 of nll[i] w.r.t parameter) for each parameter
		for i in range(len(self.theta)):
			sum_gradient_squared = tf.add_n([tf.square(tf.gradients(unaggregated_nll[j], self.theta[i])[0]) for j in range(self.ewc_batch_size)])
			self.accumulate_squared_gradients.append(tf.assign_add(self.new_fisher_diagonal[i], sum_gradient_squared))   #just use .op ?
		
		# scaling down accumulated fisher in self.new_fisher_diagonal
		self.fisher_full_batch_average_op = [tf.assign(var, self.fisher_average_scale * var) for var in self.new_fisher_diagonal]
		# update backed up variables with current variable for storing in file, at the end of training
		self.update_theta_op = [v1.assign(v2) for v1, v2 in zip(self.theta_lagged, self.theta)]

		# update fisher diagonal with sum of old one and the one calculated for current task
		# improve : add weighted sum?
		self.update_with_new_fisher_diagonal_op = [v1.assign_add(v2) for v1, v2 in zip(self.fisher_diagonal, self.new_fisher_diagonal)]

	# setup train step ; to be called just before start of training
	def prepareForTraining(self, sess, model_init_name):
		self.train_step = self.createTrainStep()		# create train step with desired optimizer, learning rate and fisher multiplier
		init = tf.global_variables_initializer()		# initialize all variables
		sess.run(init)
		if model_init_name:								# restore parameters if ini
			print("Restoring paramters from %s" % (model_init_name, ))
			self.restoreModel(sess, model_init_name)
	
	# create optimizer, loss function for training
	def createTrainStep(self):
		with tf.name_scope("optimizer"):
			self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
			# fisher penalty
			penalty = tf.add_n([tf.reduce_sum(tf.square(w1-w2)*f) for w1, w2, f
								in zip(self.theta, self.theta_lagged, self.fisher_diagonal)])
			self.loss_with_penalty = self.loss + (self.fisher_multiplier / 2) * penalty
			return self.optimizer.minimize(self.loss + (self.fisher_multiplier / 2) * penalty, var_list=self.theta)
	
	# restore model with name : model_name
	def restoreModel(self, sess, model_name):
		ckpt = tf.train.get_checkpoint_state(checkpoint_dir=self.checkpoint_path, latest_filename=model_name)   # meaning of this?
		self.saver.restore(sess=sess, save_path=ckpt.model_checkpoint_path)

	# save weights with name : time_step, model_name
	def saveWeights(self, time_step, sess, model_name):
		if not os.path.exists(self.checkpoint_path):
			os.makedirs(self.checkpoint_path)
		self.saver.save(sess=sess, save_path=self.checkpoint_path + model_name + '.ckpt', global_step=time_step,
						latest_filename=model_name)     # doubt : meaning of this?
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
		_, loss, loss_with_penalty = sess.run([self.train_step, self.loss, self.loss_with_penalty], feed_dict=feed_dict)
		return loss, loss_with_penalty

	# creat feed_dict using batch_xs, batch_ys
	def createFeedDict(self, batch_xs, batch_ys):
		feed_dict = {self.x: batch_xs, self.y: batch_ys}
		if self.apply_dropout:
			feed_dict.update({self.keep_prob_hidden: self.dropout_hidden_prob, self.keep_prob_input: self.dropout_input_prob})
		return feed_dict

	# Calculate and update parameter importances (fisher information), given current dataset
	def updateFisherFullBatch(self, sess, dataset):
		dataset.initializeIterator(self.ewc_batch_size)
		num_iters = dataset.images.shape[0] // self.ewc_batch_size 			# will leave out some examples if total elements not divisible by batch_size
		sess.run(self.zero_new_fisher_diagonal)
		# calculate sum of gradient^2 and accumulate in self.new_fisher_diagonal
		for _ in range(0, num_iters):
			batch_xs, batch_ys = dataset.nextBatch(sess)
			sess.run(self.accumulate_squared_gradients, feed_dict={self.x_fisher: batch_xs, self.y_fisher: batch_ys})

		# average self.new_fisher_diagonal
		sess.run(self.fisher_full_batch_average_op, feed_dict={self.fisher_average_scale: 1.0 / (num_iters * self.ewc_batch_size)})
		sess.run(self.update_theta_op)						# update backed up parameters to current values
		sess.run(self.update_with_new_fisher_diagonal_op)	# update self.fisher_diagonal with sum of previous value and current calculate one

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