import os
import tensorflow as tf

from network import Network


class Classifier(object):
    def __init__(self, network, input_shape, output_shape, checkpoint_path):
        self.x = None
        self.y = None
        self.x_fisher = None
        self.y_fisher = None
        self.keep_prob_input = None     # get input dropout values
        self.keep_prob_hidden = None
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.checkpoint_path = checkpoint_path
        self.ewc_batch_size = 100

        self.createPlaceholders()

        self.learning_rate = 5e-6
        self.fisher_multiplier = 0.0
        self.apply_dropout = True
        self.dropout_input_prob = 1.0
        self.dropout_hidden_prob = 1.0

        self.network = network
        self.scores, self.layer_output = self.network.forward(self.x, self.apply_dropout, self.keep_prob_input, self.keep_prob_hidden)

        self.theta = None
        self.theta_lagged = None
        self.fisher_diagonal = None
        self.new_fisher_diagonal = None
        self.createFisherDiagonal()

        self.loss = None
        self.accuracy = None
        self.createLossAccuracy()

        self.new_fisher_diagonal = None
        self.zero_new_fisher_diagonal = None
        self.accumulate_squared_gradients = None        
        self.fisher_full_batch_average_op = None
        self.update_theta_op = None
        self.update_with_new_fisher_diagonal_op = None
        self.createFisherOps()

        self.saver = tf.train.Saver(max_to_keep=100, keep_checkpoint_every_n_hours=1, var_list=self.theta + self.theta_lagged + self.fisher_diagonal)
    
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

    def createFisherDiagonal(self):
        self.theta = self.network.getLayerVariables()
        self.theta_lagged = []
        self.fisher_diagonal = []
        for i in range(len(self.theta)):
            self.theta_lagged.append(tf.Variable(tf.constant(0.0, shape=self.theta[i].shape), trainable=False))
            self.fisher_diagonal.append(tf.Variable(tf.constant(0.0, shape=self.theta[i].shape), trainable=False))

    def createLossAccuracy(self):
        with tf.name_scope("loss"):
            average_nll = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.y))  # wat abt just softmax_cross_entropy
            # tf.summary.scalar("loss", average_nll)
            self.loss = average_nll
        with tf.name_scope('accuracy'):
            accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.scores, 1), tf.argmax(self.y, 1)), tf.float32))
            # tf.summary.scalar('accuracy', accuracy)
            self.accuracy = accuracy

    def createFisherOps(self):
        self.new_fisher_diagonal = []
        self.zero_new_fisher_diagonal = []
        for i in range(len(self.theta)):
            self.new_fisher_diagonal.append(tf.Variable(tf.constant(0.0, shape=self.theta[i].shape), trainable=False))
            self.zero_new_fisher_diagonal.append(tf.assign(self.new_fisher_diagonal[i], tf.constant(0.0, shape=self.new_fisher_diagonal[i].shape)))     # just use .op here to avoid getting return value

        scores, _ = self.network.forward(self.x_fisher, apply_dropout=False)
        unaggregated_nll = tf.reduce_sum(-1 * self.y_fisher * tf.nn.log_softmax(scores), axis=1)
        self.accumulate_squared_gradients = []
        for i in range(len(self.theta)):
            sum_gradient_squared = tf.add_n([tf.square(tf.gradients(unaggregated_nll[j], self.theta[i])[0]) for j in range(self.ewc_batch_size)])
            self.accumulate_squared_gradients.append(tf.assign_add(self.new_fisher_diagonal[i], sum_gradient_squared))   #just use .op ?
        
        self.fisher_full_batch_average_op = [tf.assign(var, self.fisher_average_scale * var) for var in self.new_fisher_diagonal]
        self.update_theta_op = [v1.assign(v2) for v1, v2 in zip(self.theta_lagged, self.theta)]

        self.update_with_new_fisher_diagonal_op = [v1.assign_add(v2) for v1, v2 in zip(self.fisher_diagonal, self.new_fisher_diagonal)]

    def prepareForTraining(self, sess, model_name, model_init_name):
        # self.writer = tf.summary.FileWriter(self.summaries_path + model_name, sess.graph)
        # self.merged = tf.summary.merge_all()
        self.train_step = self.createTrainStep(self.fisher_multiplier)
        init = tf.global_variables_initializer()
        sess.run(init)
        if model_init_name:
            print("Restoring paramters from %s" % (model_init_name, ))
            self.restoreModel(sess, model_init_name)
    
    def createTrainStep(self, fisher_multiplier):
        with tf.name_scope("optimizer"):
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
            penalty = tf.add_n([tf.reduce_sum(tf.square(w1-w2)*f) for w1, w2, f
                                in zip(self.theta, self.theta_lagged, self.fisher_diagonal)])
            self.loss_with_penalty = self.loss + (fisher_multiplier / 2) * penalty
            return self.optimizer.minimize(self.loss + (fisher_multiplier / 2) * penalty, var_list=self.theta)
    
    def restoreModel(self, sess, model_name):
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir=self.checkpoint_path, latest_filename=model_name)   # meaning of this?
        self.saver.restore(sess=sess, save_path=ckpt.model_checkpoint_path)

    def saveWeights(self, time_step, sess, model_name):
        if not os.path.exists(self.checkpoint_path):
            os.makedirs(self.checkpoint_path)
        self.saver.save(sess=sess, save_path=self.checkpoint_path + model_name + '.ckpt', global_step=time_step,
                        latest_filename=model_name)     # meaning of this?
        print('saving model ' + model_name + ' at time step ' + str(time_step))

    def evaluate(self, sess, feed_dict):
        if self.apply_dropout:
            feed_dict.update({self.keep_prob_input: 1.0, self.keep_prob_hidden: 1.0})
        accuracy = sess.run([self.loss, self.accuracy], feed_dict=feed_dict)
        return accuracy

    def getPredictions(self, sess, feed_dict):
        if self.apply_dropout:
            feed_dict.update({self.keep_prob_input: 1.0, self.keep_prob_hidden: 1.0})
        scores, y = sess.run([self.scores, self.y], feed_dict=feed_dict)
        return scores, y

    def singleTrainStep(self, sess, feed_dict):
        _, loss, loss_with_penalty = sess.run([self.train_step, self.loss, self.loss_with_penalty], feed_dict=feed_dict)
        return loss, loss_with_penalty

    def createFeedDict(self, batch_xs, batch_ys):
        feed_dict = {self.x: batch_xs, self.y: batch_ys}
        if self.apply_dropout:
            feed_dict.update({self.keep_prob_hidden: self.dropout_hidden_prob, self.keep_prob_input: self.dropout_input_prob})
        return feed_dict

    def updateFisherFullBatch(self, sess, dataset):
        dataset.initializeIterator(self.ewc_batch_size)
        num_iters = dataset.images.shape[0] // self.ewc_batch_size
        sess.run(self.zero_new_fisher_diagonal)
        for _ in range(0, num_iters):
            self.accumulateFisher(sess, dataset)
        sess.run(self.fisher_full_batch_average_op, feed_dict={self.fisher_average_scale: 1.0 / (num_iters * self.ewc_batch_size)})
        sess.run(self.update_theta_op)
        sess.run(self.update_with_new_fisher_diagonal_op)   # better way of adding, like scaled addition?

    def accumulateFisher(self, sess, dataset):
        batch_xs, batch_ys = dataset.nextBatch(sess)
        sess.run(self.accumulate_squared_gradients, feed_dict={self.x_fisher: batch_xs, self.y_fisher: batch_ys})

    def updateHparams(self, hparams):
        for k, v in hparams.items():
            if (not isinstance(k, str)):
                raise Exception('Panic! hyperparameter key not string')
            setattr(self, k, v)

    def getPenultimateOutput(self, sess, feed_dict):
        penultimate_output = sess.run(self.layer_output[-2], feed_dict=feed_dict)
        return penultimate_output