import numpy as np

from copy import deepcopy
from classifiers import Classifier
from numpy.random import RandomState
from queue import PriorityQueue
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets

import sys

import tensorflow as tf

PRNG = RandomState(12345)
MINI_BATCH_SIZE = 250
LOG_FREQUENCY = 100

class MyDataset(object):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels
        self.dataset = tf.data.Dataset.from_tensor_slices((images, labels))
        self.batch_size = 0
        self.initialized = True

    def initialize_iterator(self, batch_size):
        self.initialized = False
        self.batch_size = batch_size
        self.dataset_temp = self.dataset.batch(batch_size)
        self.iterator = self.dataset_temp.make_initializable_iterator()
        self.next_element = self.iterator.get_next()

    def reinitialize_iterator(self):
        self.initialized = False
        self.dataset_temp = self.dataset.batch(self.batch_size)
        self.iterator = self.dataset_temp.make_initializable_iterator()
        self.next_element = self.iterator.get_next()

    def next_batch(self, sess, batch_size):
        if (not self.initialized):
            sess.run(self.iterator.initializer)
            self.initialized = True

        try:
            ret = sess.run(self.next_element)
        except tf.errors.OutOfRangeError:
            sess.run(self.iterator.initializer)
            ret = sess.run(self.next_element)
        
        return ret


class MyTask(object):
    def __init__(self, task):
        self.train = MyDataset(task.train._images, task.train._labels)
        self.validation = MyDataset(task.validation._images, task.validation._labels)
        self.test = MyDataset(task.test._images, task.test._labels)

class HyperparameterTuner(object):
    def __init__(self, sess, hidden_layers, hidden_units, num_perms, trials, epochs, checkpoint_path, summaries_path, data_path):
        self.hidden_layers = hidden_layers
        self.hidden_units = hidden_units
        self.num_perms = num_perms
        self.epochs = epochs
        self.data_path = data_path
        self.task_list = self.create_split_mnist_task(num_perms)
        self.trial_learning_rates = [PRNG.uniform(1e-4, 1e-3) for _ in range(0, trials)]
        self.best_parameters = []
        self.sess = sess
        self.classifier = Classifier(num_class=10,
                                     num_features=784,
                                     fc_hidden_units=[hidden_units for _ in range(hidden_layers)],
                                     apply_dropout=True,
                                     checkpoint_path=checkpoint_path,
                                     summaries_path=summaries_path)

    def search(self):
        for t in range(0, self.num_perms):
            queue = PriorityQueue()
            for learning_rate in self.trial_learning_rates:
                self.train_on_task(t, learning_rate, queue)
            self.best_parameters.append(queue.get())
            self.evaluate()

    def evaluate(self):
        accuracies = []
        for parameters in self.best_parameters:
            accuracy = self.classifier.test(sess=self.sess,
                                            model_name=parameters[1],
                                            batch_xs=self.task_list[0].test.images,
                                            batch_ys=self.task_list[0].test.labels)
            accuracies.append(accuracy)
        print(accuracies)

    def train_on_task(self, t, lr, queue):
        model_name = self.file_name(lr, t)
        dataset_train = self.task_list[t].train
        dataset_lagged = self.task_list[t - 1].train if t > 0 else None
        model_init_name = self.best_parameters[t - 1][1] if t > 0 else None
        dataset_train.initialize_iterator(MINI_BATCH_SIZE)
        if (dataset_lagged is not None):
            dataset_lagged.initialize_iterator(MINI_BATCH_SIZE)
        self.classifier.train(sess=self.sess,
                              model_name=model_name,
                              model_init_name=model_init_name,
                              dataset=dataset_train,
                              dataset_lagged=dataset_lagged,
                              num_updates=(self.task_list[t].train.images.shape[0]//MINI_BATCH_SIZE)*self.epochs,
                              mini_batch_size=MINI_BATCH_SIZE,
                              log_frequency=LOG_FREQUENCY,
                              fisher_multiplier=1.0/lr,
                              learning_rate=lr)
        accuracy = self.classifier.test(sess=self.sess,
                                        model_name=model_name,
                                        batch_xs=self.task_list[0].validation.images,
                                        batch_ys=self.task_list[0].validation.labels)
        queue.put((-accuracy, model_name))

    def create_split_mnist_task(self, num_datasets):
        mnist = read_data_sets(self.data_path, one_hot=True)
        seed = 1
        task_list = self.split_mnist(mnist, num_datasets, seed)
        return task_list

    @staticmethod
    def split_mnist(mnist, num_datasets, seed):
        np.random.seed(seed)
        num_class = 10
        task_size = num_class // num_datasets
        task_list = []
        train_labels = np.argmax(mnist.train.labels, axis=1)
        validation_labels = np.argmax(mnist.validation.labels, axis=1)
        test_labels = np.argmax(mnist.test.labels, axis=1)
        for i in range(num_datasets):
            cur_train_indices = [False] * mnist.train.images.shape[0]
            cur_validation_indices = [False] * mnist.validation.images.shape[0]
            cur_test_indices = [False] * mnist.test.images.shape[0]
            for j in range(task_size):
                cur_train_indices = np.logical_or(cur_train_indices, (train_labels == (i * task_size + j)))
                cur_validation_indices = np.logical_or(cur_validation_indices, (validation_labels == (i * task_size + j)))
                cur_test_indices = np.logical_or(cur_test_indices, (test_labels == (i * task_size + j)))

            task = deepcopy(mnist)
            task.train._images = task.train._images[cur_train_indices]
            task.train._labels = task.train._labels[cur_train_indices]
            task.validation._images = task.validation._images[cur_validation_indices]
            task.validation._labels = task.validation._labels[cur_validation_indices]
            task.test._images = task.test._images[cur_test_indices]
            task.test._labels = task.test._labels[cur_test_indices]
            task = MyTask(task)
            task_list.append(task)

        return task_list

    def create_permuted_mnist_task(self, num_datasets):
        mnist = read_data_sets(self.data_path, one_hot=True)
        task_list = [mnist]
        for seed in range(1, num_datasets):
            task_list.append(self.permute(mnist, seed))
        return task_list

    @staticmethod
    def permute(task, seed):
        np.random.seed(seed)
        perm = np.random.permutation(task.train._images.shape[1])
        permuted = deepcopy(task)
        permuted.train._images = permuted.train._images[:, perm]
        permuted.test._images = permuted.test._images[:, perm]
        permuted.validation._images = permuted.validation._images[:, perm]
        return permuted

    def file_name(self, lr, t):
        return 'layers=%d,hidden=%d,lr=%.5f,multiplier=%.2f,mbsize=%d,epochs=%d,perm=%d' \
               % (self.hidden_layers, self.hidden_units, lr, 1 / lr, MINI_BATCH_SIZE, self.epochs, t)
