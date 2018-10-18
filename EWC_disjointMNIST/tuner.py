import numpy as np

from copy import deepcopy
from classifiers import Classifier
from numpy.random import RandomState
from queue import PriorityQueue
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets

import sys
import os
import time
import pickle

import tensorflow as tf

PRNG = RandomState(12345)
MINI_BATCH_SIZE = 250
LOG_FREQUENCY = 100

class MyDataset(object):
    # todo: shuffle dataset
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

    def next_batch(self, sess):
        if (not self.initialized):
            sess.run(self.iterator.initializer)
            self.initialized = True

        try:
            ret = sess.run(self.next_element)
        except tf.errors.OutOfRangeError:
            sess.run(self.iterator.initializer)
            ret = sess.run(self.next_element)
        
        return ret

    def get_data(self, start, end):
        return self.images[start: end, :], self.labels[start: end, :]


class MyTask(object):
    def __init__(self, task, train_images=None, train_labels=None):
        if train_images is None:
            self.train = MyDataset(task.train._images, task.train._labels)
            self.validation = MyDataset(task.validation._images, task.validation._labels)
            self.test = MyDataset(task.test._images, task.test._labels)
        else:
            self.train = MyDataset(train_images, train_labels)
            self.validation = MyDataset(task.validation.images, task.validation.labels)
            self.test = MyDataset(task.test.images, task.test.labels)

# class HyperparameterTuner(object):
#     def __init__(self, sess, hidden_layers, hidden_units, trials, epochs, 
#                 checkpoint_path, summaries_path, data_path, split_path,
#                 dropout_keep_input=1.0, dropout_keep_hidden=1.0):
#         self.hidden_layers = hidden_layers
#         self.hidden_units = hidden_units
#         self.split = self.read_split(split_path)
#         self.num_split = len(self.split)
#         self.epochs = epochs
#         self.data_path = data_path
#         self.task_list = self.create_split_mnist_task()
#         self.trial_learning_rates = [PRNG.uniform(1e-4, 1e-3) for _ in range(0, trials)]
#         self.best_parameters = []
#         self.sess = sess
#         self.classifier = Classifier(num_class=10,
#                                      num_features=784,
#                                      fc_hidden_units=[hidden_units for _ in range(hidden_layers)],
#                                      apply_dropout=True,
#                                      checkpoint_path=checkpoint_path,
#                                      summaries_path=summaries_path,
#                                      dropout_keep_input=dropout_keep_input,
#                                      dropout_keep_hidden=dropout_keep_hidden)

#     def search(self):
#         for t in range(0, self.num_split):
#             queue = PriorityQueue()
#             for learning_rate in self.trial_learning_rates:
#                 self.train_on_task(t, learning_rate, queue)
#             self.best_parameters.append(queue.get())
#             self.evaluate()

#     def evaluate(self):
#         accuracies = []
#         for parameters in self.best_parameters:
#             accuracy = self.classifier.test(sess=self.sess,
#                                             model_name=parameters[1],
#                                             batch_xs=self.task_list[0].test.images,
#                                             batch_ys=self.task_list[0].test.labels)
#             accuracies.append(accuracy)
#         print(accuracies)

#     def train_on_task(self, t, lr, queue):
#         model_name = self.file_name(lr, 1.0 / lr, t)
#         dataset_train = self.task_list[t].train
#         dataset_lagged = self.task_list[t - 1].train if t > 0 else None
#         model_init_name = self.best_parameters[t - 1][1] if t > 0 else None
#         dataset_train.initialize_iterator(MINI_BATCH_SIZE)
#         if (dataset_lagged is not None):
#             dataset_lagged.initialize_iterator(MINI_BATCH_SIZE)
#         self.classifier.train(sess=self.sess,
#                               model_name=model_name,
#                               model_init_name=model_init_name,
#                               dataset=dataset_train,
#                               dataset_lagged=dataset_lagged,
#                               num_updates=(self.task_list[t].train.images.shape[0]//MINI_BATCH_SIZE)*self.epochs,
#                               mini_batch_size=MINI_BATCH_SIZE,
#                               log_frequency=LOG_FREQUENCY,
#                               fisher_multiplier=1.0/lr,
#                               learning_rate=lr)
#         accuracy = self.classifier.test(sess=self.sess,
#                                         model_name=model_name,
#                                         batch_xs=self.task_list[0].validation.images,
#                                         batch_ys=self.task_list[0].validation.labels)
#         queue.put((-accuracy, model_name))

#     def create_split_mnist_task(self):
#         mnist = read_data_sets(self.data_path, one_hot=True)
#         seed = 1
#         task_list = self.split_mnist(mnist, self.split, seed)
#         return task_list

#     @staticmethod
#     def read_split(split_path):
#         split = []
#         try:
#             f = open(split_path)
#             while (True):
#                 line = f.readline()
#                 if (line == ""):
#                     break
#                 split.append([float(i) for i in line.split()])
#         except IOError:
#             print("split path file not found")
#             exit(-1)
#         return split
    
#     @staticmethod
#     def split_mnist(mnist, dataset_split, seed):
#         np.random.seed(seed)
#         task_list = []
#         train_labels = np.argmax(mnist.train.labels, axis=1)
#         validation_labels = np.argmax(mnist.validation.labels, axis=1)
#         test_labels = np.argmax(mnist.test.labels, axis=1)
#         for i in range(len(dataset_split)):
#             cur_train_indices = [False] * mnist.train.images.shape[0]
#             cur_validation_indices = [False] * mnist.validation.images.shape[0]
#             cur_test_indices = [False] * mnist.test.images.shape[0]
#             for j in range(len(dataset_split[i])):
#                 cur_train_indices = np.logical_or(cur_train_indices, (train_labels == dataset_split[i][j]))
#                 cur_validation_indices = np.logical_or(cur_validation_indices, (validation_labels == dataset_split[i][j]))
#                 cur_test_indices = np.logical_or(cur_test_indices, (test_labels == dataset_split[i][j]))

#             task = deepcopy(mnist)
#             task.train._images = task.train._images[cur_train_indices]
#             task.train._labels = task.train._labels[cur_train_indices]
#             task.validation._images = task.validation._images[cur_validation_indices]
#             task.validation._labels = task.validation._labels[cur_validation_indices]
#             task.test._images = task.test._images[cur_test_indices]
#             task.test._labels = task.test._labels[cur_test_indices]
#             task = MyTask(task)
#             task_list.append(task)

#         return task_list

#     def create_permuted_mnist_task(self, num_datasets):
#         mnist = read_data_sets(self.data_path, one_hot=True)
#         task_list = [mnist]
#         for seed in range(1, num_datasets):
#             task_list.append(self.permute(mnist, seed))
#         return task_list

#     @staticmethod
#     def permute(task, seed):
#         np.random.seed(seed)
#         perm = np.random.permutation(task.train._images.shape[1])
#         permuted = deepcopy(task)
#         permuted.train._images = permuted.train._images[:, perm]
#         permuted.test._images = permuted.test._images[:, perm]
#         permuted.validation._images = permuted.validation._images[:, perm]
#         return permuted

#     def file_name(self, lr, fm, t):
#         return 'layers=%d,hidden=%d,lr=%.5f,multiplier=%.2f,mbsize=%d,epochs=%d,perm=%d' \
#                % (self.hidden_layers, self.hidden_units, lr, fm, MINI_BATCH_SIZE, self.epochs, t)


class HyperparameterTuner(object):
    def __init__(self, sess, network, input_shape, output_shape, 
                checkpoint_path, summaries_path,
                read_datasets, 
                load_best_hparams=False):
        
        self.sess = sess
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.checkpoint_path = checkpoint_path
        self.summaries_path = summaries_path

        # self.split = self.read_split(split_path)
        # self.num_split = len(self.split)
        # self.task_list = self.create_split_mnist_task()
        self.split = None
        self.task_list = None
        self.split, self.task_list = read_datasets()
        self.num_tasks = len(self.split)
        
        self.best_hparams = [None for _ in range(self.num_tasks)]
        self.results_list = [{} for _ in range(self.num_tasks)]
        self.hparams_list = [[] for _ in range(self.num_tasks)]
        self.default_hparams = {'learning_rate': 5e-6, 'fisher_multiplier': 0.0, 
                                'dropout_input_prob': 1.0, 'dropout_hidden_prob': 1.0}
        self.count_not_improving_threshold = 10
        self.eval_frequency = 100
        self.print_every = 1000
        
        if not os.path.exists(self.checkpoint_path):
            os.makedirs(self.checkpoint_path)
        if not os.path.exists(self.summaries_path):
            os.makedirs(self.summaries_path)

        best_hparams_filename = 'best_hparams.dat'
        self.best_hparams_filepath = os.path.join(summaries_path, best_hparams_filename)
        if load_best_hparams:
            with open(self.best_hparams_filepath, 'rb') as fp:
                self.best_hparams = pickle.load(fp)

        self.classifier = Classifier(network, input_shape, output_shape, checkpoint_path)

        self.save_penultimate_output = False
        self.per_example_append = 0
        self.appended_task_list = [None for _ in range(self.num_tasks)]

    def train(self, t, hparams, batch_size, model_init_name, num_updates=0, verbose=False):
        # make sure all previous tasks have been trained
        default_hparams = deepcopy(self.default_hparams)
        default_hparams.update(hparams)
        hparams = default_hparams
        self.classifier.update_hparams(hparams)
        
        model_name = self.file_name(t, hparams)
        print("Training with %s" % (model_name, ))
        self.classifier.prepare_for_training(sess=self.sess, 
                                            model_name=model_name, 
                                            model_init_name=model_init_name)
        
        val_acc = [[] for _ in range(t + 1)]
        val_loss = [[] for _ in range(t + 1)]
        loss = []
        loss_with_penalty = []
        cur_best_avg = 0.0
        cur_best_avg_num_updates = -1
        i = 0
        count_not_improving = 0
        dataset_train = self.appended_task_list[t].train
        dataset_val = self.task_list[t].validation
        dataset_train.initialize_iterator(batch_size)
        dataset_val.initialize_iterator(batch_size)

        updates_per_epoch = dataset_train.images.shape[0] // batch_size
        num_tolerate_epochs = 2
        while (True):
            batch_xs, batch_ys = dataset_train.next_batch(self.sess)
            feed_dict = self.classifier.create_feed_dict(batch_xs, batch_ys)
            cur_loss, cur_loss_with_penalty = self.classifier.single_train_step(self.sess, feed_dict)
            loss.append(cur_loss)
            loss_with_penalty.append(cur_loss_with_penalty)
            if (i % self.eval_frequency == 0):
                cur_iter_avg = 0.0
                cur_iter_num_classes = 0 # actually doesn't depend on iterations
                for j in range(t + 1):
                    val_data = self.task_list[j].validation
                    feed_dict = self.classifier.create_feed_dict(val_data.images, val_data.labels)
                    accuracy = self.classifier.evaluate(self.sess, feed_dict)
                    val_loss[j].append(accuracy[0])
                    val_acc[j].append(accuracy[1])
                    cur_iter_avg += accuracy[1] * len(self.split[j]) # assuming all classes are equally likely
                    cur_iter_num_classes += len(self.split[j])
                cur_iter_avg /= cur_iter_num_classes

                if (val_acc[-1][-1] == np.max(np.array(val_acc)[:, -1]) and cur_best_avg >= cur_iter_avg):
                    count_not_improving += 1
                else:
                    count_not_improving = 0

                if (cur_iter_avg > cur_best_avg):
                    cur_best_avg = cur_iter_avg
                    cur_best_avg_num_updates = i

                if (count_not_improving * self.eval_frequency >= updates_per_epoch * num_tolerate_epochs):
                    if (num_updates == 0):
                        break
            
            if (verbose and (i % self.print_every == 0)):
                print("validation accuracies: %s, loss: %f, loss with penalty: %f" % (str(np.array(val_acc)[:, -1]), loss[-1], loss_with_penalty[-1]))

            i += 1
            if (num_updates > 0 and (i >= num_updates)):
                break
                
            total_updates = i

        print("epochs: %f, final train loss: %f, validation accuracies: %s" % (i / updates_per_epoch, loss[-1], str(np.array(val_acc)[:, -1])))
        print("best epochs: %f, best_avg: %f, validation accuracies: %s" % 
                (cur_best_avg_num_updates / updates_per_epoch, cur_best_avg, np.array(val_acc)[:, cur_best_avg_num_updates // self.eval_frequency]))
        return val_acc, val_loss, loss, loss_with_penalty, cur_best_avg, cur_best_avg_num_updates, total_updates

    def get_appended_task(self, t, model_init_name, batch_size):
        appended_task = None
        with open(self.checkpoint_path + model_init_name + '_penultimate_output.txt', 'rb') as f:
            old_penultimate_output, old_taskid_offset = pickle.load(f)
        
        cur_penultimate_output, _ = self.get_penultimate_output(t, batch_size)

        similarity = np.matmul(cur_penultimate_output, old_penultimate_output.T)
        cur_penultimate_output_norm = np.sqrt(np.sum((cur_penultimate_output ** 2), axis=1))
        old_penultimate_output_norm = np.sqrt(np.sum((old_penultimate_output ** 2), axis=1))
        similarity = similarity / old_penultimate_output_norm / np.expand_dims(cur_penultimate_output_norm, axis=1)
        topk_similar = np.argsort(similarity, axis=-1)[:, -self.per_example_append: ]
        train_task = self.task_list[t].train
        appended_images_shape = tuple([train_task.images.shape[0] * (self.per_example_append + 1)] + list(train_task.images.shape)[1: ])
        appended_labels_shape = tuple([train_task.labels.shape[0] * (self.per_example_append + 1)] + list(train_task.labels.shape)[1: ])
        appended_images = np.empty(appended_images_shape)
        appended_labels = np.empty(appended_labels_shape)

        offset = 0
        for i in range(train_task.images.shape[0]):
            appended_images[offset] = train_task.images[i]
            appended_labels[offset] = train_task.labels[i]
            for j in range(self.per_example_append):
                index = old_taskid_offset[topk_similar[i, j]]
                appended_images[offset + j + 1] = self.task_list[index[0]].train.images[index[1]]
                appended_labels[offset + j + 1] = self.task_list[index[0]].train.labels[index[1]]
            offset += 1 + self.per_example_append
        appended_task = MyTask(self.task_list[t], appended_images, appended_labels)    
        
        return appended_task


    def tune_on_task(self, t, batch_size, model_init_name=None, num_updates=0, verbose=False, save_weights=True):
        best_avg = 0.0
        best_hparams = None
        if model_init_name is None:
            model_init_name = self.best_hparams[t - 1][-1] if t > 0 else None
        
        # todo: retrieve model_init_name and append points from old dataset to current task and add it to self.appended_task_list
        if (self.per_example_append > 0):
            if (t == 0):
                self.appended_task_list[0] = self.task_list[0]
            else:
                self.classifier.restore_model(self.sess, model_init_name)
                self.appended_task_list[t] = self.get_appended_task(t, model_init_name, batch_size)
        else:
            self.appended_task_list[t] = self.task_list[t]
            
        for hparams in self.hparams_list[t]:
            cur_result = self.train(t, hparams, batch_size, model_init_name, num_updates=num_updates, verbose=verbose)
            self.classifier.update_fisher_full_batch(self.sess, self.task_list[t].train)
            val_acc, val_loss, loss, loss_with_penalty, cur_best_avg, cur_best_avg_num_updates, total_updates = cur_result
            if (save_weights):
                self.classifier.save_weights(total_updates, self.sess, self.file_name(t, hparams))
            self.save_results(cur_result, self.file_name(t, hparams))
            hparams_tuple = tuple([v for k, v in sorted(hparams.items())])
            self.results_list[t][hparams_tuple] = {}
            self.results_list[t][hparams_tuple]['val_acc'] = val_acc
            self.results_list[t][hparams_tuple]['val_loss'] = val_loss
            self.results_list[t][hparams_tuple]['loss'] = loss
            self.results_list[t][hparams_tuple]['loss_with_penalty'] = loss_with_penalty
            self.results_list[t][hparams_tuple]['best_avg'] = cur_best_avg
            self.results_list[t][hparams_tuple]['best_avg_updates'] = cur_best_avg_num_updates
            if (cur_best_avg > best_avg):
                best_avg = cur_best_avg
                best_hparams = hparams
        
        
        if (self.best_hparams[t] is None):
            self.best_hparams[t] = (best_hparams, self.file_name(t, best_hparams))
        else:
            prev_best_hparams_tuple = tuple([v for k, v in sorted(self.best_hparams[t][0].items())])
            prev_best_avg = self.results_list[t][prev_best_hparams_tuple]['best_avg']
            if (best_avg > prev_best_avg):
                self.best_hparams[t] = (best_hparams, self.file_name(t, best_hparams))

        best_hparams_tuple = tuple([v for k, v in sorted(best_hparams.items())])
        cur_result = self.train(t, best_hparams, batch_size, model_init_name,
                                num_updates=self.results_list[t][best_hparams_tuple]['best_avg_updates'])
        self.classifier.update_fisher_full_batch(self.sess, self.task_list[t].train)
        val_acc, val_loss, loss, loss_with_penalty, cur_best_avg, cur_best_avg_num_updates, total_updates = cur_result
        self.classifier.save_weights(self.results_list[t][best_hparams_tuple]['best_avg_updates'], 
                                        self.sess, self.file_name(t, best_hparams))

        # check: save penultimate output to a file associated to current model: doing this only for best model as it is the one mostly gonna be used for next task
        if (self.save_penultimate_output):
            print("calculating penultimate output...")
            start_time = time.time()
            penultimate_output, taskid_offset = self.get_all_penultimate_output(t, batch_size)
            print("time taken: %f", time.time() - start_time)
            print("saving penultimate output...")
            with open(self.checkpoint_path + self.file_name(t, best_hparams) + '_penultimate_output.txt', 'wb') as f:
                pickle.dump((penultimate_output, taskid_offset), f)

        return best_avg, best_hparams

    def get_all_penultimate_output(self, t, batch_size):
        # check: for each dataset in self.task_list[0:t + 1], append to a single numpy array
        total_elements = sum([task.train.images.shape[0] for task in self.task_list[0: t + 1]])
        # assuming penultimate layer is output of fc layer
        penultimate_output_size = int(self.classifier.layer_output[-2].shape[-1])
        penultimate_output = np.empty(shape=(total_elements, penultimate_output_size))
        taskid_offset = np.full((total_elements, 2), -1)
        offset = 0
        for i in range(t + 1):
            cur_num_elements = self.task_list[i].train.images.shape[0]
            cur_penultimate_output, cur_taskid_offset = self.get_penultimate_output(i, batch_size)
            penultimate_output[offset: offset + cur_num_elements, :] = cur_penultimate_output
            taskid_offset[offset: offset + cur_num_elements] = cur_taskid_offset
            offset += cur_num_elements            
        
        return penultimate_output, taskid_offset
    
    def get_penultimate_output(self, t, batch_size):
        num_elements = self.task_list[t].train.images.shape[0]
        penultimate_output_size = int(self.classifier.layer_output[-2].shape[-1])
        penultimate_output = np.empty(shape=(num_elements, penultimate_output_size))
        taskid_offset = np.full((num_elements, 2), -1)
        offset = 0

        num_batches = self.task_list[t].train.images.shape[0] // batch_size
        for j in range(num_batches):
            batch_xs, batch_ys = self.task_list[t].train.get_data(j * batch_size, (j + 1) * batch_size)
            feed_dict = self.classifier.create_feed_dict(batch_xs, batch_ys)
            cur_penultimate_output = self.classifier.get_penultimate_output(self.sess, feed_dict)
            penultimate_output[offset: offset + batch_size, :] = cur_penultimate_output
            taskid_offset[offset: offset + batch_size, 0] = np.full((batch_size, ), t)
            taskid_offset[offset: offset + batch_size, 1] = np.arange(j * batch_size, (j + 1) * batch_size)
            offset += batch_size
        if (self.task_list[t].train.images.shape[0] % batch_size != 0):
            j += 1
            num_remaining = self.task_list[t].train.images.shape[0] % batch_size
            batch_xs, batch_ys = self.task_list[t].train.get_data(j * batch_size, j * batch_size + num_remaining)
            feed_dict = self.classifier.create_feed_dict(batch_xs, batch_ys)
            cur_penultimate_output = self.classifier.get_penultimate_output(self.sess, feed_dict)
            penultimate_output[offset: offset + num_remaining, :] = cur_penultimate_output
            taskid_offset[offset: offset + num_remaining, 0] = np.full((num_remaining, ), t)
            taskid_offset[offset: offset + num_remaining, 1] = np.arange(j * batch_size, j * batch_size + num_remaining)
            offset += num_remaining

        return penultimate_output, taskid_offset



    def validation_accuracy(self, t, batch_size):
        self.classifier.restore_model(self.sess, self.best_hparams[t][1])
        accuracy = [None for _ in range(t + 1)]
        for i in range(t + 1):
            num_batches = self.task_list[i].validation.images.shape[0] // batch_size
            dataset = self.task_list[i].validation
            dataset.initialize_iterator(batch_size)
            cur_accuracy = 0.0
            for j in range(num_batches):
                batch_xs, batch_ys = dataset.next_batch(self.sess)
                feed_dict = self.classifier.create_feed_dict(batch_xs, batch_ys)
                cur_accuracy += self.classifier.evaluate(self.sess, feed_dict)[1]
            cur_accuracy /= num_batches
            accuracy[i] = cur_accuracy
        return accuracy

    def test(self, t, batch_size):
        self.classifier.restore_model(self.sess, self.best_hparams[t][1])
        accuracy = [None for _ in range(t + 1)]
        for i in range(t + 1):
            num_batches = self.task_list[i].test.images.shape[0] // batch_size
            dataset = self.task_list[i].test
            dataset.initialize_iterator(batch_size)
            cur_accuracy = 0.0
            for j in range(num_batches):
                batch_xs, batch_ys = dataset.next_batch(self.sess)
                feed_dict = self.classifier.create_feed_dict(batch_xs, batch_ys)
                cur_accuracy += self.classifier.evaluate(self.sess, feed_dict)[1]
            cur_accuracy /= num_batches
            accuracy[i] = cur_accuracy
        return accuracy

    def test_split(self, t, batch_size, split):
        self.classifier.restore_model(self.sess, self.best_hparams[t][1])
        accuracy = np.array([0 for _ in range(len(split))])
        elements_per_split = np.array([0 for _ in range(len(split))])
        for i in range(t + 1):
            num_batches = self.task_list[i].test.images.shape[0] // batch_size
            dataset = self.task_list[i].test
            dataset.initialize_iterator(batch_size)
            for j in range(num_batches):
                batch_xs, batch_ys = dataset.next_batch(self.sess)
                feed_dict = self.classifier.create_feed_dict(batch_xs, batch_ys)
                scores, y = self.classifier.get_predictions(self.sess, feed_dict)
                y_pred = np.argmax(scores, axis=1)
                y_true = np.argmax(y, axis=1)
                for k in range(len(split)):
                    y_split_index = np.isin(y_true, split[k])
                    y_pred_temp = y_pred[y_split_index]
                    y_true_temp = y_true[y_split_index]
                    accuracy[k] += np.sum(y_true_temp == y_pred_temp)
                    elements_per_split[k] += y_temp.shape[0]

        accuracy = accuracy / elements_per_split
        return list(accuracy)



    def save_best_hparams(self):
        with open(self.best_hparams_filepath, 'wb') as fp:
            pickle.dump(self.best_hparams, fp)

    def load_best_hparams(self):
        with open(self.best_hparams_filepath, 'rb') as fp:
            self.best_hparams = pickle.load(fp)

    def save_results(self, result, file_name):
        with open(self.summaries_path + file_name, 'wb') as fp:
            pickle.dump(result, fp)

    def save_results_list(self):
        with open(self.summaries_path + 'all_results', 'wb') as fp:
            pickle.dump(self.results_list, fp)

    def load_results_list(self):
        with open(self.summaries_path + 'all_results', 'rb') as fp:
            self.results_list = pickle.load(fp)

    def file_name(self, t, hparams):
        model_name = ''
        for k, v in sorted(hparams.items()):
            model_name += str(k) + '=' + str(v) + ','
        model_name += 'task=' + str(t)
        return model_name