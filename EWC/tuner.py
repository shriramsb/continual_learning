import numpy as np
import math

from copy import deepcopy
from classifiers import Classifier
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets

import sys
import os
import time
import pickle

import torch

import tensorflow as tf

MINI_BATCH_SIZE = 250
LOG_FREQUENCY = 100
VALIDATION_BATCH_SIZE = 1024

# Dataset - helps access dataset batch-wise
class MyDataset(object):
	# improve : shuffle dataset?
	def __init__(self, images, labels):
		self.images = images
		self.labels = labels
		self.batch_size = 0
		self.pos = 0                            # pointer storing position starting from which to return data for nextBatch()

	# modify batch size and position pointer to the start of dataset 
	def initializeIterator(self, batch_size):
		self.batch_size = batch_size
		self.pos = 0

	# position pointer to the start of dataset
	def reinitializeIterator(self):
		self.pos = 0

	# get next batch, in round-robin fashion
	def nextBatch(self, sess):
		# last batch might be smaller than self.batch_size
		if (self.pos + self.batch_size >= self.images.shape[0]):
			ret = self.images[self.pos : ], self.labels[self.pos : ]
			self.pos = 0
		else:
			ret = self.images[self.pos : self.pos + self.batch_size], self.labels[self.pos : self.pos + self.batch_size]
			self.pos = self.pos + self.batch_size
		
		return ret

	# get data in [start, end)
	def getData(self, start, end):
		return self.images[start: end, :], self.labels[start: end, :]

# Task having dataset for train, dev, test
class MyTask(object):
	def __init__(self, task, train_images=None, train_labels=None):
		# MNIST dataset from tf loaded dataset
		if (type(task) == tf.contrib.learn.datasets.base.Datasets):
			self.train = MyDataset(task.train._images, task.train._labels)
			self.validation = MyDataset(task.validation._images, task.validation._labels)
			self.test = MyDataset(task.test._images, task.test._labels)
		else:
			if train_images is None:
				self.train = MyDataset(task.train.images, task.train.labels)
				self.validation = MyDataset(task.validation.images, task.validation.labels)
				self.test = MyDataset(task.test.images, task.test.labels)
			else:
				self.train = MyDataset(train_images, train_labels)
				self.validation = MyDataset(task.validation.images, task.validation.labels)
				self.test = MyDataset(task.test.images, task.test.labels)


# Helps in tuning hyperparameters of classifer (network) on different tasks
class HyperparameterTuner(object):
	def __init__(self, sess, network, input_shape, output_shape, 
				checkpoint_path, summaries_path,
				readDatasets, 
				load_best_hparams=False):
		
		self.sess = sess 									# tf session
		self.input_shape = input_shape						# input shape to network
		self.output_shape = output_shape					# output shape from network
		self.checkpoint_path = checkpoint_path
		self.summaries_path = summaries_path				# path to store summary - train loss, val accuracy, etc

		# dataset to train on
		self.split = None 									# list (list (labels of each task))
		self.num_tasks = None 								# number of sequential tasks
		self.task_weights = None 							# weights to be given to validation accuracy for each task for weighted average of perforamance
		self.task_list = None 								# list of MyTask objects, specifying tasks
		self.split, self.num_tasks, self.task_weights, self.task_list = readDatasets() 	# readDataset() passed as argument to __init__, returns dataset
		
		self.cumulative_split = []
		for t in range(self.num_tasks):
			self.cumulative_split.append([])
			for i in range(t + 1):
				self.cumulative_split[t].extend(self.split[i])

		self.best_hparams = [None for _ in range(self.num_tasks)] 						# best hparams after training, for each task
		self.results_list = [{} for _ in range(self.num_tasks)] 						# results (list of loss, accuracy) for each task, hparam
		self.hparams_list = [[] for _ in range(self.num_tasks)] 						# list of hparams for each task to tune on
		self.default_hparams = {'learning_rate': 5e-6, 'fisher_multiplier': 0.0,  		# default values of hparams
								'dropout_input_prob': 1.0, 'dropout_hidden_prob': 1.0}
		self.num_tolerate_epochs = 2 							# number of epochs to wait if validation accuracy isn't improving
		self.eval_frequency = 1 								# frequency of calculation of validation accuracy
		self.print_every = 1000 								# frequency of printing, if verbose during training
		
		# create checkpoint and summaries directories if they don't exist
		if not os.path.exists(self.checkpoint_path):
			os.makedirs(self.checkpoint_path)
		if not os.path.exists(self.summaries_path):
			os.makedirs(self.summaries_path)

		# name of best hparams file, which stores best hparams of all tasks
		best_hparams_filename = 'best_hparams.dat'
		self.best_hparams_filepath = os.path.join(summaries_path, best_hparams_filename)
		if load_best_hparams:
			with open(self.best_hparams_filepath, 'rb') as fp:
				self.best_hparams = pickle.load(fp)

		# classifier object
		self.classifier = Classifier(network, input_shape, output_shape, checkpoint_path)

		self.use_gpu = True 											# if use gpu for matrix multiplication - uses PyTorch for mm
		
		self.save_penultimate_output = False 							# specifies whether to save penultimate output after training on a task
		# number of examples from all old tasks to append to current task (per example), just before starting to train on it, to avoid forgetting of old tasks
		# task after appending with examples to train set from old tasks
		self.appended_task_list = [None for _ in range(self.num_tasks)]
		self.tuner_hparams = {'old:new' : 4}
		self.tuner_hparams['bf_num_images'] = 2000

	def setPerExampleAppend(self, val):
		self.tuner_hparams['old:new'] = val
		if (val > 0):
			self.save_penultimate_output = True
		else:
			self.save_penultimate_output = False

	# get learning rate for current epoch
	def getLearningRate(self, base_lr_list, epoch):
		for i in range(len(base_lr_list) - 1):
			if (epoch < base_lr_list[i][0]):
				return base_lr_list[i][1]
		
		return base_lr_list[-1]

	# train on a given task with given hparams - hparams
	# save every epoch not supported yet
	def train(self, t, hparams, batch_size, model_init_name, num_updates=-1, verbose=False, save_weights_every_epoch=False):
		# make sure hparams has all required hparams for classifier
		default_hparams = deepcopy(self.default_hparams)
		default_hparams.update(hparams)
		hparams = default_hparams
		self.classifier.updateHparams(hparams)
		
		# model_name depending on current hparams
		model_name = self.fileName(t, hparams, self.tuner_hparams)
		print("Training with %s" % (model_name, ))
		# restore model from model_init_name and create train step
		self.classifier.prepareForTraining(sess=self.sess, 
											model_init_name=model_init_name)
		
		# variables to monitor training
		val_acc = [[] for _ in range(t + 1)] 				# validation loss, accuracy for all tasks till current task
		val_loss = [[] for _ in range(t + 1)]
		loss = []											# training loss, loss with fisher penalty for current task
		loss_with_penalty = []
		train_acc = []

		cur_best_avg = 0.0 									# best weighted average validation accuracy and update number at which it occurs
		cur_best_avg_num_epoch = -1
		i = 0												# keeps track of iteration number
		count_not_improving = 0 							# number of updates for which average validation accuracy isn't improving (starts after some threshold iterations)
		dataset_train = self.appended_task_list[t].train 	# current task's train, validation datasets
		dataset_val = self.task_list[t].validation 			
		dataset_train.initializeIterator(batch_size) 		# set batch_size and pointer to start of dataset
		dataset_val.initializeIterator(batch_size)

		updates_per_epoch = math.ceil(self.task_list[t].train.images.shape[0] / batch_size) 	# number of train steps in an epoch
		self.num_tolerate_epochs = 2 												# number of epochs to wait if average validation accuracy isn't improving
		epoch = 0
		# training loop
		while (True):
			# single step of training
			batch_xs, batch_ys = dataset_train.nextBatch(self.sess)
			cur_lr = self.getLearningRate(hparams['learning_rate'][0], epoch) 		# balancing phase not currently used
			feed_dict = self.classifier.createFeedDict(batch_xs, batch_ys, learning_rate=cur_lr, is_training=True)
			cur_loss, cur_loss_with_penalty, cur_accuracy = self.classifier.singleTrainStep(self.sess, feed_dict)

			loss.append(cur_loss)
			loss_with_penalty.append(cur_loss_with_penalty)
			train_acc.append(cur_accuracy)
			
			i += 1
			if (i % updates_per_epoch == 0):
				epoch += 1

			if (epoch % self.eval_frequency == 0 and i % updates_per_epoch == 0):	
				cur_iter_weighted_avg = 0.0
				cur_iter_weights_sum = 0
				# get validation accuracy for all tasks till 't'
				accuracy = self.validationAccuracy(t, VALIDATION_BATCH_SIZE, restore_model=False, get_loss=True)
				# calculating weighted average of accuracy
				for j in range(t + 1):
					val_loss[j].append(accuracy[0][j])
					val_acc[j].append(accuracy[1][j])
					cur_iter_weighted_avg += accuracy[1][j] * self.task_weights[j]
					cur_iter_weights_sum += self.task_weights[j]
				cur_iter_weighted_avg /= cur_iter_weights_sum

				# if accuracy of current task > (max. accuracy) / 2, then start counting non-improving updates
				if (val_acc[-1][-1] >= np.max(np.array(val_acc)[:, -1]) / 2 and cur_best_avg >= cur_iter_weighted_avg):
					count_not_improving += 1
				else:
					count_not_improving = 0

				# store update number at which validation accuracy is maximum
				if (cur_iter_weighted_avg > cur_best_avg):
					cur_best_avg = cur_iter_weighted_avg
					cur_best_avg_num_epoch = epoch

				# stop training if validation accuracy not improving for self.num_tolerate_epochs epochs
				if (count_not_improving * self.eval_frequency >= self.num_tolerate_epochs):
					if (num_updates == -1):
						break

				print("epoch: %d, iter: %d/%d, validation accuracies: %s, average train loss: %f, average train accuracy: %f" % 
						(epoch, i - epoch * updates_per_epoch, updates_per_epoch,
						str(np.array(val_acc)[:, -1]), 
						np.mean(loss[i - updates_per_epoch : ]), np.mean(train_acc[i - updates_per_epoch : ])))
			
			# print stats if verbose
			if (verbose and (i % self.print_every == 0)):
				print("epoch: %d, iter: %d/%d, loss: %f, accuracy: %f" % 
						(epoch, i - epoch * updates_per_epoch, updates_per_epoch,
						loss[-1], train_acc[-1]))

			# break if num_updates is specified ; break at nearest epoch which requires updates >= num_updates
			if (num_updates >= 0 and (epoch >= math.ceil(num_updates / updates_per_epoch))):
				break
				
			total_updates = i

		print("epochs: %f, final train loss: %f, validation accuracies: %s" % (epoch, loss[-1], str(np.array(val_acc)[:, -1])))
		print("best epochs: %f, best_avg: %f, validation accuracies: %s" % 
				(cur_best_avg_num_epoch, cur_best_avg, np.array(val_acc)[:, (cur_best_avg_num_epoch - 1) // self.eval_frequency]))
		
		ret = {}
		ret['val_acc'] = val_acc
		ret['val_loss'] = val_loss
		ret['loss'] = loss
		ret['loss_with_penalty'] = loss_with_penalty
		ret['acc'] = train_acc
		ret['best_avg'] = cur_best_avg
		ret['best_epoch'] = cur_best_avg_num_epoch
		ret['updates_per_epoch'] = updates_per_epoch
		ret['total_updates'] = total_updates
		ret['total_epochs'] = epoch
		return ret

	def getAppendedRandomTask(self, t):
		num_elements = self.tuner_hparams['bf_num_images']
		train_task = self.task_list[t].train

		classes = self.cumulative_split[t]
		elements_per_class = math.floor(num_elements / len(classes))
		num_elements = elements_per_class * len(classes)

		appended_images_shape = tuple([num_elements] + list(train_task.images.shape)[1: ])
		appended_labels_shape = tuple([num_elements] + list(train_task.labels.shape)[1: ])       
		
		appended_images = np.empty(appended_images_shape)
		appended_labels = np.empty(appended_labels_shape)
		offset = 0

		# TODO : need to take care of case where cur_indices.shape[0] < elements_per_class. Currently sampling with replace, which is actually wrong
		for i in range(t + 1):
			for j in range(len(self.split[i])):
				cur_indices = (np.argmax(self.task_list[i].train.labels, axis=1) == self.split[i][j])
				cur_images = self.task_list[i].train.images[cur_indices]
				cur_labels = self.task_list[i].train.labels[cur_indices]
				if (cur_images.shape[0] >= elements_per_class):
					sample_indices = np.random.choice(range(cur_images.shape[0]), replace=False, size=elements_per_class)
				else:
					sample_indices = np.random.choice(range(cur_images.shape[0]), replace=True, size=elements_per_class)
				appended_images[offset : offset + elements_per_class] = cur_images[sample_indices]
				appended_labels[offset : offset + elements_per_class] = cur_labels[sample_indices]
				offset += elements_per_class
		
		appended_random_task = MyTask(self.task_list[t], train_images=appended_images, train_labels=appended_labels)

		return appended_random_task


	# append task with examples from previous tasks
	def getAppendedTask(self, t, model_init_name, batch_size, optimize_space=False, equal_weights=False, apply_log=False):
		appended_task = None

		# if equal_weights is True, then assign equal weights to all points till current task. 
		# Otherwise, weigh each point of previous tasks by cosine similarity to current task's points
		if (not equal_weights):
			if (model_init_name is not None):
				self.classifier.restoreModel(self.sess, model_init_name)
			# load penultimate output of previous tasks' examples for model_init_name 
			with open(self.checkpoint_path + model_init_name + '_penultimate_output.txt', 'rb') as f:
				old_penultimate_output, old_taskid_offset = pickle.load(f)
		
			# penultimate output of current task for model_init_name
			cur_penultimate_output, _ = self.getPenultimateOutput(t, batch_size)

			# norm of vectors for cosine similarity
			cur_penultimate_output_norm = np.sqrt(np.sum((cur_penultimate_output ** 2), axis=1))
			old_penultimate_output_norm = np.sqrt(np.sum((old_penultimate_output ** 2), axis=1))

			# similarity - each row gives cosine similarity between an example from current task with all examples from previous tasks
			# if required to optimize space, do multiplication one row at a time of current task's example's output with matrix of previous tasks' examples' outputs
			if (optimize_space):
				similarity = np.empty((cur_penultimate_output.shape[0], (old_penultimate_output.T).shape[1]), np.float32)
				# use gpu for multiplication if available ; using PyTorch here
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
					if (apply_log):
						similarity[i] = -np.log(similarity[i] + 1e-24)
				if (self.use_gpu):
					del b
			else:
				similarity = np.matmul(cur_penultimate_output, old_penultimate_output.T)
				similarity = similarity / old_penultimate_output_norm / np.expand_dims(cur_penultimate_output_norm, axis=1)
				if (apply_log):
					similarity = -np.log(similarity + 1e-24)
		
			train_task = self.task_list[t].train

			old_new_ratio = self.tuner_hparams['old:new']
			topk_similar = np.argsort(similarity, axis=-1)[:, -old_new_ratio: ]
			appended_images_shape = tuple([train_task.images.shape[0] * (old_new_ratio + 1)] + list(train_task.images.shape)[1: ])
			appended_labels_shape = tuple([train_task.labels.shape[0] * (old_new_ratio + 1)] + list(train_task.labels.shape)[1: ])
			appended_images = np.empty(appended_images_shape)
			appended_labels = np.empty(appended_labels_shape)

			offset = 0
			for i in range(train_task.images.shape[0]):
				appended_images[offset] = train_task.images[i]
				appended_labels[offset] = train_task.labels[i]
				for j in range(old_new_ratio):
					index = old_taskid_offset[topk_similar[i, j]]
					appended_images[offset + j + 1] = self.task_list[index[0]].train.images[index[1]]
					appended_labels[offset + j + 1] = self.task_list[index[0]].train.labels[index[1]]
				offset += 1 + old_new_ratio

			appended_task = MyTask(self.task_list[t], appended_images, appended_labels)    		
			
		else:
			train_task = self.task_list[t].train
			old_new_ratio = self.tuner_hparams['old:new']
			temp_bf_num_images = self.tuner_hparams['bf_num_images']
			self.tuner_hparams['bf_num_images'] = self.task_list[t].train.images.shape[0] * old_new_ratio
			old_task_subset = self.getAppendedRandomTask(t - 1)
			self.tuner_hparams['bf_num_images'] = temp_bf_num_images
			appended_images_shape = tuple([train_task.images.shape[0] + old_task_subset.train.images.shape[0]] + list(train_task.images.shape)[1: ])
			appended_labels_shape = tuple([train_task.images.shape[0] + old_task_subset.train.images.shape[0]] + list(train_task.labels.shape)[1: ])
			appended_images = np.empty(appended_images_shape)
			appended_labels = np.empty(appended_labels_shape)
			offset = 0
			appended_images[offset : offset + old_task_subset.train.images.shape[0]] = old_task_subset.train.images
			appended_labels[offset : offset + old_task_subset.train.images.shape[0]] = old_task_subset.train.labels
			offset += old_task_subset.train.images.shape[0]
			appended_images[offset : ] = self.task_list[t].train.images
			appended_labels[offset : ] = self.task_list[t].train.labels

			shuffler = np.arange(appended_images.shape[0])
			np.random.shuffle(shuffler)
			appended_images = appended_images[shuffler]
			appended_labels = appended_labels[shuffler]
			
			appended_task = MyTask(self.task_list[t], train_images=appended_images, train_labels=appended_labels)

		return appended_task


	def hparamsDictToTuple(self, hparams, tuner_hparams):
		hparams_tuple = [v for k, v in sorted(hparams.items())]
		hparams_tuple = hparams_tuple + [v for k, v in sorted(tuner_hparams.items())]
		return tuple(hparams_tuple)

	# Train on task 't' with hparams in self.hparams_list[t] and find the best one ; calls train() internally ; ; make sure all previous tasks have been trained
	def tuneOnTask(self, t, batch_size, model_init_name=None, num_updates=-1, verbose=False, restore_params=True, 
					equal_weights=False, apply_log=False,
					save_weights_every_epoch=False, save_weights_end=True,
					final_train=False):
		best_avg = 0.0							# best average validation accuracy and hparams corresponding to it from self.hparams_list[t]
		best_hparams = None

		if (not restore_params):
			model_init_name = None

		if model_init_name is None:				# model with which to initialize weights
			if (restore_params):
				model_init_name = self.best_hparams[t - 1][-1] if t > 0 else None
		
		# appende examples to current task from old task if required
		old_new_ratio = self.tuner_hparams['old:new']
		if (old_new_ratio > 0):
			if (t == 0):
				self.appended_task_list[0] = self.task_list[0]
			else:
				self.appended_task_list[t] = self.getAppendedTask(t, model_init_name, batch_size, optimize_space=True, equal_weights=equal_weights, 
																	apply_log=apply_log)
		else:
			self.appended_task_list[t] = self.task_list[t]
			

		# loop through self.hparams_list[t], train with it and find the best one
		for hparams in self.hparams_list[t]:
			cur_result = self.train(t, hparams, batch_size, model_init_name, num_updates=num_updates, verbose=verbose, 
									save_weights_every_epoch=save_weights_every_epoch)
			cur_best_avg = cur_result['best_avg']
			if (save_weights_end):
				self.classifier.updateFisherFullBatch(self.sess, self.task_list[t].train)
				self.classifier.saveWeights(cur_result['total_updates'], self.sess, self.fileName(t, hparams, self.tuner_hparams))
			self.saveResults(cur_result, self.fileName(t, hparams, self.tuner_hparams))
			hparams_tuple = self.hparamsDictToTuple(hparams, self.tuner_hparams)
			self.results_list[t][hparams_tuple] = cur_result
			if (cur_best_avg > best_avg):
				best_avg = cur_best_avg
				best_hparams = hparams
		
		
		# update best hparams
		if (self.best_hparams[t] is None):
			self.best_hparams[t] = (best_hparams, self.tuner_hparams, self.fileName(t, best_hparams, self.tuner_hparams))
		else:
			prev_best_hparams_tuple = self.hparamsDictToTuple(self.best_hparams[t][0], self.best_hparams[t][1])
			prev_best_avg = self.results_list[t][prev_best_hparams_tuple]['best_avg']
			if (best_avg > prev_best_avg):
				self.best_hparams[t] = (best_hparams, self.tuner_hparams, self.fileName(t, best_hparams, self.tuner_hparams))

		# retrain model for best_avg_updates for best hparam to get optimal validation accuracy
		if final_train:
			best_hparams_tuple = self.hparamsDictToTuple(best_hparams, self.tuner_hparams)
			cur_result = self.train(t, best_hparams, batch_size, model_init_name,
									num_updates=self.results_list[t][best_hparams_tuple]['best_epoch'] * self.results_list[t][best_hparams_tuple]['updates_per_epoch'])
			self.classifier.updateFisherFullBatch(self.sess, self.task_list[t].train)
			self.classifier.saveWeights(self.results_list[t][best_hparams_tuple]['best_epoch'] * self.results_list[t][best_hparams_tuple]['updates_per_epoch'], 
											self.sess, self.fileName(t, best_hparams, self.tuner_hparams))

		# calculate penultimate output of all tasks till 't' and save to file
		if (self.save_penultimate_output):
			print("calculating penultimate output...")
			start_time = time.time()
			penultimate_output, taskid_offset = self.getAllPenultimateOutput(t, batch_size)
			print("time taken: %f", time.time() - start_time)
			print("saving penultimate output...")
			with open(self.checkpoint_path + self.fileName(t, best_hparams, self.tuner_hparams) + '_penultimate_output.txt', 'wb') as f:
				pickle.dump((penultimate_output, taskid_offset), f)

		return best_avg, best_hparams

	# train on a range of tasks sequentially [start, end] with different hparams ; currently only positive num_updates allowed
	def tuneTasksInRange(self, start, end, batch_size, num_hparams, num_updates=0, verbose=False, equal_weights=False, restore_params=True, 
							apply_log=False, early_stop=False, 
							old_new_ratio_list=None):
		if (num_updates < 0):
			print("bad num_updates argument.. stopping")
			return 0, self.hparams_list[start][0]

		best_avg = 0.0
		best_hparams_index = -1

		# for each hparam, train on tasks in [start, end]
		for k in range(num_hparams):
			# requires model to have been trained on task (start - 1) with same hparams self.hparams_list[start - 1][k]
			for i in range(start, end + 1):
				model_init_name = None
				if (i > 0):
					if (restore_params):
						model_init_name = self.fileName(i - 1, self.hparams_list[i - 1][k], self.tuner_hparams)
				
				if (old_new_ratio_list is not None):
					self.setPerExampleAppend(old_new_ratio_list[i])
				old_new_ratio = self.tuner_hparams['old:new']
				if (old_new_ratio > 0):
					if i == 0:
						self.appended_task_list[i] = self.task_list[i]
					else:
						self.appended_task_list[i] = self.getAppendedTask(i, model_init_name, batch_size, optimize_space=True, equal_weights=equal_weights, 
																			apply_log=apply_log)
				else:
					self.appended_task_list[i] = self.task_list[i]

				hparams = self.hparams_list[i][k]
				cur_result = self.train(i, hparams, batch_size, model_init_name, num_updates=num_updates, verbose=verbose)
				if (not early_stop):
					self.classifier.updateFisherFullBatch(self.sess, self.task_list[i].train)
					self.classifier.saveWeights(cur_result['total_updates'], self.sess, self.fileName(i, hparams, self.tuner_hparams))
				
				self.saveResults(cur_result, self.fileName(i, hparams, self.tuner_hparams))
				hparams_tuple = self.hparamsDictToTuple(hparams, self.tuner_hparams)
				self.results_list[i][hparams_tuple] = cur_result

				if early_stop:
					cur_result = self.train(i, hparams, batch_size, model_init_name,
									num_updates=self.results_list[i][hparams_tuple]['best_epoch'] * self.results_list[i][hparams_tuple]['updates_per_epoch'])
					self.classifier.updateFisherFullBatch(self.sess, self.task_list[i].train)
					self.classifier.saveWeights(self.results_list[i][hparams_tuple]['best_epoch'] * self.results_list[i][hparams_tuple]['updates_per_epoch'], 
												self.sess, self.fileName(i, hparams, self.tuner_hparams))
				
				cur_best_avg = self.results_list[i][hparams_tuple]['best_avg']

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
			if (old_new_ratio_list is not None):
				self.setPerExampleAppend(old_new_ratio_list[i])
			self.best_hparams[i] = (self.hparams_list[i][best_hparams_index], self.tuner_hparams, self.fileName(i, self.hparams_list[i][best_hparams_index], self.tuner_hparams))

		return best_avg, best_hparams_index

	# get penultimate output of all layers till 't' using current parameters of network
	def getAllPenultimateOutput(self, t, batch_size):
		total_elements = sum([task.train.images.shape[0] for task in self.task_list[0: t + 1]])
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
	
	# get penultimate output for task 't' using current parameters of network
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


	# validation loss, accuracy till task 't'
	def validationAccuracy(self, t, batch_size, restore_model=True, get_loss=False):
		if restore_model:
			self.classifier.restoreModel(self.sess, self.best_hparams[t][-1])
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
				cur_accuracy += eval_result[1] * batch_xs.shape[0]
				cur_loss += eval_result[0] * batch_xs.shape[0]
			cur_accuracy /= self.task_list[i].validation.images.shape[0]
			cur_loss /= self.task_list[i].validation.images.shape[0]
			accuracy[i] = cur_accuracy
			loss[i] = cur_loss
		if (get_loss):
			return loss, accuracy
		else:
			return accuracy

	# test accuracy till task 't'
	def test(self, t, batch_size, restore_model=True, model_init_name=None):
		if restore_model:
			if model_init_name is not None:
				self.classifier.restoreModel(self.sess, model_init_name)
			else:
				self.classifier.restoreModel(self.sess, self.best_hparams[t][-1])
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

	# need to check again
	def testSplit(self, t, batch_size, split):
		self.classifier.restoreModel(self.sess, self.best_hparams[t][-1])
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


	# utility functions to save/load self.best_hparams, self.results_list
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

	def fileName(self, t, hparams, tuner_hparams):
		model_name = ''
		for k, v in sorted(hparams.items()):
			model_name += str(k) + '=' + str(v) + ','
		for k, v in sorted(tuner_hparams.items()):
			model_name += str(k) + '=' + str(v) + ','
		model_name += 'task=' + str(t)
		return model_name