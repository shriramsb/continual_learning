import numpy as np
import math

from copy import deepcopy
from classifiers import Classifier
from numpy.random import RandomState
from queue import PriorityQueue
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets

import sys
import os
import time
import pickle

import torch

import tensorflow as tf

PRNG = RandomState(12345)
MINI_BATCH_SIZE = 250
LOG_FREQUENCY = 100
VALIDATION_BATCH_SIZE = 1024

class MyDataset(object):
    # todo: shuffle dataset
    def __init__(self, images, labels, weights=None):
        self.images = images
        self.labels = labels
        # self.dataset = tf.data.Dataset.from_tensor_slices((self.images, self.labels))
        self.batch_size = 0
        # self.initialized = True
        self.weights = weights
        self.pos = 0

    def initializeIterator(self, batch_size):
        # self.initialized = False
        self.batch_size = batch_size
        # self.dataset_temp = self.dataset.batch(batch_size)
        # self.iterator = self.dataset_temp.make_initializable_iterator()
        # self.next_element = self.iterator.get_next()
        self.pos = 0

    def reinitializeIterator(self):
        # self.initialized = False
        # self.dataset_temp = self.dataset.batch(self.batch_size)
        # self.iterator = self.dataset_temp.make_initializable_iterator()
        # self.next_element = self.iterator.get_next()
        self.pos = 0

    def nextBatch(self, sess):
        if (self.pos + self.batch_size >= self.images.shape[0]):
            ret = self.images[self.pos : ], self.labels[self.pos : ]
            self.pos = 0
        else:
            ret = self.images[self.pos : self.pos + self.batch_size], self.labels[self.pos : self.pos + self.batch_size]
            self.pos = self.pos + self.batch_size
        
        return ret

    def nextBatchSample(self, sess):
        total_examples = self.images.shape[0]
        sampled_indices = np.random.choice(range(total_examples), p = self.weights, size=self.batch_size)
        batch_xs = self.images[sampled_indices]
        batch_ys = self.labels[sampled_indices]
        batch_weights = 1.0 / self.weights[sampled_indices]
        return batch_xs, batch_ys, batch_weights

    def getData(self, start, end):
        return self.images[start: end, :], self.labels[start: end, :]


class MyTask(object):
    def __init__(self, task, train_images=None, train_labels=None, weights=None):        
        if train_images is None:
            if weights is None:
                weights = np.array([1 for _ in range(task.train.images.shape[0])]) / task.train.images.shape[0]
            self.train = MyDataset(task.train._images, task.train._labels, weights)
            self.validation = MyDataset(task.validation._images, task.validation._labels)
            self.test = MyDataset(task.test._images, task.test._labels)
        else:
            if weights is None:
                weights = np.array([1 for _ in range(train_images.shape[0])]) / train_images.shape[0]
            self.train = MyDataset(train_images, train_labels, weights)
            self.validation = MyDataset(task.validation.images, task.validation.labels)
            self.test = MyDataset(task.test.images, task.test.labels)



class HyperparameterTuner(object):
    def __init__(self, sess, network, input_shape, output_shape, 
                checkpoint_path, summaries_path,
                readDatasets, 
                load_best_hparams=False):
        
        self.sess = sess
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.checkpoint_path = checkpoint_path
        self.summaries_path = summaries_path

        self.split = None
        self.num_tasks = None
        self.task_weights = None
        self.task_list = None
        self.split, self.num_tasks, self.task_weights, self.task_list = readDatasets()
        
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

        self.use_gpu = True

        self.save_penultimate_output = True
        self.per_example_append = 2
        self.appended_task_list = [None for _ in range(self.num_tasks)]
        self.tuner_hparams = {'old:new': self.per_example_append}

    def train(self, t, hparams, batch_size, model_init_name, num_updates=0, verbose=False):
        # make sure all previous tasks have been trained
        default_hparams = deepcopy(self.default_hparams)
        default_hparams.update(hparams)
        hparams = default_hparams
        self.classifier.updateHparams(hparams)
        
        model_name = self.fileName(t, hparams)
        print("Training with %s" % (model_name, ))
        self.classifier.prepareForTraining(sess=self.sess, 
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
        dataset_train.initializeIterator(batch_size)
        dataset_val.initializeIterator(batch_size)

        updates_per_epoch = dataset_train.images.shape[0] // batch_size
        num_tolerate_epochs = 2
        while (True):
            batch_xs, batch_ys, batch_weights = dataset_train.nextBatchSample(self.sess)
            feed_dict = self.classifier.createFeedDict(batch_xs, batch_ys, batch_weights)
            cur_loss, cur_loss_with_penalty = self.classifier.singleTrainStep(self.sess, feed_dict)
            loss.append(cur_loss)
            loss_with_penalty.append(cur_loss_with_penalty)
            if (i % self.eval_frequency == 0):
                cur_iter_weighted_avg = 0.0
                cur_iter_weights_sum = 0 # actually doesn't depend on iterations
                accuracy = self.validationAccuracy(t, VALIDATION_BATCH_SIZE, restore_model=False, get_loss=True)
                for j in range(t + 1):
                    val_loss[j].append(accuracy[0][j])
                    val_acc[j].append(accuracy[1][j])
                    cur_iter_weighted_avg += accuracy[1][j] * self.task_weights[j]
                    cur_iter_weights_sum += self.task_weights[j]
                cur_iter_weighted_avg /= cur_iter_weights_sum

                if (val_acc[-1][-1] > np.max(np.array(val_acc)[:, -1]) / 2 and cur_best_avg >= cur_iter_weighted_avg):
                    count_not_improving += 1
                else:
                    count_not_improving = 0

                if (cur_iter_weighted_avg > cur_best_avg):
                    cur_best_avg = cur_iter_weighted_avg
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


    def getAppendedTask(self, t, model_init_name, batch_size, optimize_space=False, equal_weights=False):
        appended_task = None

        if (not equal_weights):
            with open(self.checkpoint_path + model_init_name + '_penultimate_output.txt', 'rb') as f:
                old_penultimate_output, old_taskid_offset = pickle.load(f)
        
            cur_penultimate_output, _ = self.getPenultimateOutput(t, batch_size)

            cur_penultimate_output_norm = np.sqrt(np.sum((cur_penultimate_output ** 2), axis=1))
            old_penultimate_output_norm = np.sqrt(np.sum((old_penultimate_output ** 2), axis=1))
            
            if (optimize_space):
                similarity = np.empty((cur_penultimate_output.shape[0], (old_penultimate_output.T).shape[1]), np.float32)
                if (self.use_gpu):
                    b = torch.Tensor(old_penultimate_output.T).cuda()
                for i in range(cur_penultimate_output.shape[0]):
                    if (self.use_gpu):
                        a = torch.Tensor(np.expand_dims(cur_penultimate_output[i], axis=0)).cuda()
                        similarity[i] = torch.mm(a, b).cpu()
                        del a
                    else:
                        similarity[i] = np.matmul(cur_penultimate_output[i], old_penultimate_output.T)
                    similarity[i] = similarity[i] / old_penultimate_output_norm / cur_penultimate_output_norm[i]
                if (self.use_gpu):
                    del b
            else:
                similarity = np.matmul(cur_penultimate_output, old_penultimate_output.T)
                similarity = similarity / old_penultimate_output_norm / np.expand_dims(cur_penultimate_output_norm, axis=1)

            train_task = self.task_list[t].train

            old_task_weights = np.sum(similarity, axis=0)
            old_task_weights = old_task_weights / np.sum(old_task_weights)
            old_task_weights = old_task_weights * self.per_example_append / (self.per_example_append + 1)
            cur_task_weights = np.array([1.0 / (self.per_example_append + 1) for _ in range(train_task.images.shape[0])]) / train_task.images.shape[0]

            appended_images_shape = tuple([train_task.images.shape[0] + old_penultimate_output.shape[0]] + list(train_task.images.shape)[1: ])
            appended_labels_shape = tuple([train_task.labels.shape[0] + old_penultimate_output.shape[0]] + list(train_task.labels.shape)[1: ])
            appended_weights_shape = tuple([train_task.labels.shape[0] + old_penultimate_output.shape[0]])

            appended_images = np.empty(appended_images_shape)
            appended_labels = np.empty(appended_labels_shape)
            appended_weights = np.empty(appended_weights_shape)

            offset = 0
            for i in range(t + 1):
                appended_images[offset : offset + self.task_list[i].train.images.shape[0]] = self.task_list[i].train.images
                appended_labels[offset : offset + self.task_list[i].train.labels.shape[0]] = self.task_list[i].train.labels
                offset += self.task_list[i].train.images.shape[0]
            
            appended_weights[0 : old_task_weights.shape[0]] = old_task_weights
            appended_weights[ old_task_weights.shape[0] : ] = cur_task_weights
            
            appended_task = MyTask(self.task_list[t], train_images=appended_images, train_labels=appended_labels, weights=appended_weights)
        
        else:
            train_task = self.task_list[t].train
            num_elements = 0
            for i in range(t + 1):
                num_elements += self.task_list[i].train.images.shape[0]

            appended_images_shape = tuple([num_elements] + list(train_task.images.shape)[1: ])
            appended_labels_shape = tuple([num_elements] + list(train_task.labels.shape)[1: ])
            appended_weights_shape = tuple([num_elements])            
            
            appended_images = np.empty(appended_images_shape)
            appended_labels = np.empty(appended_labels_shape)
            appended_weights = np.empty(appended_weights_shape)

            offset = 0
            for i in range(t + 1):
                appended_images[offset : offset + self.task_list[i].train.images.shape[0]] = self.task_list[i].train.images
                appended_labels[offset : offset + self.task_list[i].train.labels.shape[0]] = self.task_list[i].train.labels
                offset += self.task_list[i].train.images.shape[0]
            
            appended_weights[ : ] = 1.0 / num_elements
            
            appended_task = MyTask(self.task_list[t], train_images=appended_images, train_labels=appended_labels, weights=appended_weights)

        return appended_task


    def tuneOnTask(self, t, batch_size, model_init_name=None, num_updates=0, verbose=False, save_weights=True):
        best_avg = 0.0
        best_hparams = None
        if model_init_name is None:
            model_init_name = self.best_hparams[t - 1][-1] if t > 0 else None
        
        # todo: retrieve model_init_name and append points from old dataset to current task and add it to self.appended_task_list
        if (self.per_example_append > 0):
            if (t == 0):
                self.appended_task_list[0] = self.task_list[0]
            else:
                self.classifier.restoreModel(self.sess, model_init_name)
                self.appended_task_list[t] = self.getAppendedTask(t, model_init_name, batch_size)
        else:
            self.appended_task_list[t] = self.task_list[t]
            
        for hparams in self.hparams_list[t]:
            cur_result = self.train(t, hparams, batch_size, model_init_name, num_updates=num_updates, verbose=verbose)
            if self.classifier.use_ewc:
                self.classifier.updateFisherFullBatch(self.sess, self.task_list[t].train)
            val_acc, val_loss, loss, loss_with_penalty, cur_best_avg, cur_best_avg_num_updates, total_updates = cur_result
            if (save_weights):
                self.classifier.saveWeights(total_updates, self.sess, self.fileName(t, hparams))
            self.saveResults(cur_result, self.fileName(t, hparams))
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
            self.best_hparams[t] = (best_hparams, self.fileName(t, best_hparams))
        else:
            prev_best_hparams_tuple = tuple([v for k, v in sorted(self.best_hparams[t][0].items())])
            prev_best_avg = self.results_list[t][prev_best_hparams_tuple]['best_avg']
            if (best_avg > prev_best_avg):
                self.best_hparams[t] = (best_hparams, self.fileName(t, best_hparams))

        best_hparams_tuple = tuple([v for k, v in sorted(best_hparams.items())])
        cur_result = self.train(t, best_hparams, batch_size, model_init_name,
                                num_updates=self.results_list[t][best_hparams_tuple]['best_avg_updates'])
        if self.classifier.use_ewc:
            self.classifier.updateFisherFullBatch(self.sess, self.task_list[t].train)
        val_acc, val_loss, loss, loss_with_penalty, cur_best_avg, cur_best_avg_num_updates, total_updates = cur_result
        self.classifier.saveWeights(self.results_list[t][best_hparams_tuple]['best_avg_updates'], 
                                        self.sess, self.fileName(t, best_hparams))

        # check: save penultimate output to a file associated to current model: doing this only for best model as it is the one mostly gonna be used for next task
        if (self.save_penultimate_output):
            print("calculating penultimate output...")
            start_time = time.time()
            penultimate_output, taskid_offset = self.getAllPenultimateOutput(t, batch_size)
            print("time taken: %f", time.time() - start_time)
            print("saving penultimate output...")
            with open(self.checkpoint_path + self.fileName(t, best_hparams) + '_penultimate_output.txt', 'wb') as f:
                pickle.dump((penultimate_output, taskid_offset), f)

        return best_avg, best_hparams

    def setPerExampleAppend(self, val):
        self.per_example_append = val
        self.tuner_hparams['old:new'] = self.per_example_append

    def tuneTasksInRange(self, start, end, batch_size, num_hparams, num_updates=0, verbose=False, equal_weights=False): # currently only positive num_updates allowed
        if (num_updates <= 0):
            print("bad num_updates argument.. stopping")
            return 0, tuner.hparams_list[t][0]

        best_avg = 0.0
        best_hparams_index = -1

        for k in range(num_hparams):
            for i in range(start, end + 1):
                model_init_name = None
                if (i > 0):
                    model_init_name = self.fileName(i - 1, self.hparams_list[i - 1][k], self.tuner_hparams)
                
                if (self.per_example_append > 0):
                    if i == 0:
                        self.appended_task_list[i] = self.task_list[i]
                    else:
                        self.classifier.restoreModel(self.sess, model_init_name)
                        self.appended_task_list[i] = self.getAppendedTask(i, model_init_name, batch_size, optimize_space=True, equal_weights=equal_weights)
                else:
                    self.appended_task_list[i] = self.task_list[i]

                hparams = self.hparams_list[i][k]
                cur_result = self.train(i, hparams, batch_size, model_init_name, num_updates=num_updates, verbose=verbose)
                val_acc, val_loss, loss, loss_with_penalty, cur_best_avg, cur_best_avg_num_updates, total_updates = cur_result
                self.classifier.saveWeights(total_updates, self.sess, self.fileName(i, hparams, self.tuner_hparams))
                self.saveResults(cur_result, self.fileName(i, hparams, self.tuner_hparams))
                hparams_tuple = [v for k, v in sorted(hparams.items())]
                hparams_tuple = tuple(hparams_tuple + [v for k, v in sorted(self.tuner_hparams.items())])
                self.results_list[i][hparams_tuple] = {}
                self.results_list[i][hparams_tuple]['val_acc'] = val_acc
                self.results_list[i][hparams_tuple]['val_loss'] = val_loss
                self.results_list[i][hparams_tuple]['loss'] = loss
                self.results_list[i][hparams_tuple]['loss_with_penalty'] = loss_with_penalty
                self.results_list[i][hparams_tuple]['best_avg'] = cur_best_avg
                self.results_list[i][hparams_tuple]['best_avg_updates'] = cur_best_avg_num_updates

                if (self.save_penultimate_output):
                    print("calculating penultimate output...")
                    start_time = time.time()
                    penultimate_output, taskid_offset = self.getAllPenultimateOutput(i, batch_size)
                    print("time taken: %f", time.time() - start_time)
                    print("saving penultimate output...")
                    with open(self.checkpoint_path + self.fileName(i, hparams, self.tuner_hparams) + '_penultimate_output.txt', 'wb') as f:
                        pickle.dump((penultimate_output, taskid_offset), f)

            if (cur_best_avg > best_avg):
                best_avg = cur_best_avg
                best_hparams_index = k

        for i in range(end + 1):
            self.best_hparams[i] = self.hparams_list[i][best_hparams_index]

        return best_avg, best_hparams_index

    def getAllPenultimateOutput(self, t, batch_size):
        # check: for each dataset in self.task_list[0:t + 1], append to a single numpy array
        total_elements = sum([task.train.images.shape[0] for task in self.task_list[0: t + 1]])
        # assuming penultimate layer is output of fc layer
        penultimate_output_size = int(self.classifier.layer_output[-2].shape[-1])
        penultimate_output = np.empty(shape=(total_elements, penultimate_output_size))
        taskid_offset = np.full((total_elements, 2), -1)
        offset = 0
        for i in range(t + 1):
            cur_num_elements = self.task_list[i].train.images.shape[0]
            cur_penultimate_output, cur_taskid_offset = self.getPenultimateOutput(i, batch_size)
            penultimate_output[offset: offset + cur_num_elements, :] = cur_penultimate_output
            taskid_offset[offset: offset + cur_num_elements] = cur_taskid_offset
            offset += cur_num_elements            
        
        return penultimate_output, taskid_offset
    
    def getPenultimateOutput(self, t, batch_size):
        num_elements = self.task_list[t].train.images.shape[0]
        penultimate_output_size = int(self.classifier.layer_output[-2].shape[-1])
        penultimate_output = np.empty(shape=(num_elements, penultimate_output_size))
        taskid_offset = np.full((num_elements, 2), -1)
        offset = 0

        num_batches = self.task_list[t].train.images.shape[0] // batch_size
        for j in range(num_batches):
            batch_xs, batch_ys = self.task_list[t].train.getData(j * batch_size, (j + 1) * batch_size)
            feed_dict = self.classifier.createFeedDict(batch_xs, batch_ys)
            cur_penultimate_output = self.classifier.getPenultimateOutput(self.sess, feed_dict)
            penultimate_output[offset: offset + batch_size, :] = cur_penultimate_output
            taskid_offset[offset: offset + batch_size, 0] = np.full((batch_size, ), t)
            taskid_offset[offset: offset + batch_size, 1] = np.arange(j * batch_size, (j + 1) * batch_size)
            offset += batch_size
        if (self.task_list[t].train.images.shape[0] % batch_size != 0):
            j += 1
            num_remaining = self.task_list[t].train.images.shape[0] % batch_size
            batch_xs, batch_ys = self.task_list[t].train.getData(j * batch_size, j * batch_size + num_remaining)
            feed_dict = self.classifier.createFeedDict(batch_xs, batch_ys)
            cur_penultimate_output = self.classifier.getPenultimateOutput(self.sess, feed_dict)
            penultimate_output[offset: offset + num_remaining, :] = cur_penultimate_output
            taskid_offset[offset: offset + num_remaining, 0] = np.full((num_remaining, ), t)
            taskid_offset[offset: offset + num_remaining, 1] = np.arange(j * batch_size, j * batch_size + num_remaining)
            offset += num_remaining

        return penultimate_output, taskid_offset



    def validationAccuracy(self, t, batch_size, restore_model=True, get_loss=False):
        if restore_model:
            self.classifier.restoreModel(self.sess, self.best_hparams[t][1])
        accuracy = [None for _ in range(t + 1)]
        loss = [None for _ in range(t + 1)]
        for i in range(t + 1):
            num_batches = math.ceil(self.task_list[i].validation.images.shape[0] / batch_size)
            dataset = self.task_list[i].validation
            dataset.initializeIterator(batch_size)
            cur_accuracy = 0.0
            cur_loss = 0.0
            for j in range(num_batches):
                batch_xs, batch_ys = dataset.nextBatch(self.sess)
                feed_dict = self.classifier.createFeedDict(batch_xs, batch_ys)
                eval_result = self.classifier.evaluate(self.sess, feed_dict)
                cur_loss += eval_result[0] * batch_xs.shape[0]
                cur_accuracy += eval_result[1] * batch_xs.shape[0]
            cur_loss /= self.task_list[i].validation.images.shape[0]
            cur_accuracy /= self.task_list[i].validation.images.shape[0]
            accuracy[i] = cur_accuracy
            loss[i] = cur_loss
        if (get_loss):
            return loss, accuracy
        else:
            return accuracy

    def test(self, t, batch_size, restore_model=True):
        if restore_model:
            self.classifier.restoreModel(self.sess, self.best_hparams[t][1])
        accuracy = [None for _ in range(t + 1)]
        for i in range(t + 1):
            num_batches = math.ceil(self.task_list[i].test.images.shape[0] / batch_size)
            dataset = self.task_list[i].test
            dataset.initializeIterator(batch_size)
            cur_accuracy = 0.0
            for j in range(num_batches):
                batch_xs, batch_ys = dataset.nextBatch(self.sess)
                feed_dict = self.classifier.createFeedDict(batch_xs, batch_ys)
                cur_accuracy += self.classifier.evaluate(self.sess, feed_dict)[1] * batch_xs.shape[0]
            cur_accuracy /= self.task_list[i].test.images.shape[0]
            accuracy[i] = cur_accuracy
        return accuracy

    def testSplit(self, t, batch_size, split):
        self.classifier.restoreModel(self.sess, self.best_hparams[t][1])
        accuracy = np.array([0 for _ in range(len(split))])
        elements_per_split = np.array([0 for _ in range(len(split))])
        for i in range(t + 1):
            num_batches = self.task_list[i].test.images.shape[0] // batch_size
            dataset = self.task_list[i].test
            dataset.initializeIterator(batch_size)
            for j in range(num_batches):
                batch_xs, batch_ys = dataset.nextBatch(self.sess)
                feed_dict = self.classifier.createFeedDict(batch_xs, batch_ys)
                scores, y = self.classifier.getPredictions(self.sess, feed_dict)
                y_pred = np.argmax(scores, axis=1)
                y_true = np.argmax(y, axis=1)
                for k in range(len(split)):
                    y_split_index = np.isin(y_true, split[k])
                    y_pred_temp = y_pred[y_split_index]
                    y_true_temp = y_true[y_split_index]
                    accuracy[k] += np.sum(y_true_temp == y_pred_temp)
                    elements_per_split[k] += y_true_temp.shape[0]

        accuracy = accuracy / elements_per_split
        return list(accuracy)



    def saveBestHparams(self):
        with open(self.best_hparams_filepath, 'wb') as fp:
            pickle.dump(self.best_hparams, fp)

    def loadBestHparams(self):
        with open(self.best_hparams_filepath, 'rb') as fp:
            self.best_hparams = pickle.load(fp)

    def saveResults(self, result, file_name):
        with open(self.summaries_path + file_name, 'wb') as fp:
            pickle.dump(result, fp)

    def saveResultsList(self):
        with open(self.summaries_path + 'all_results', 'wb') as fp:
            pickle.dump(self.results_list, fp)

    def loadResultsList(self):
        with open(self.summaries_path + 'all_results', 'rb') as fp:
            self.results_list = pickle.load(fp)

    def fileName(self, t, hparams, tuner_hparams=None):
        model_name = ''
        for k, v in sorted(hparams.items()):
            model_name += str(k) + '=' + str(v) + ','

        if tuner_hparams is not None:
            for k, v in sorted(tuner_hparams.items()):
                model_name += str(k) + '=' + str(v) + ','

        model_name += 'task=' + str(t)
        return model_name