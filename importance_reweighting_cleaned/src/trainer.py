import numpy as np
import math

from copy import deepcopy
from src.classifiers import Classifier

import sys
import os
import shutil
import time
import pickle

import torch
import tensorflow as tf

import tensorboardX

import src.dataset as dataset
import src.dataset_loader as dataset_loader

VALIDATION_BATCH_SIZE = 1024


# Helps in tuning hyperparameters of classifer (network) on different tasks
class MultiTaskTrainer(object):
	def __init__(self, sess, network, input_shape, output_shape, hparams):
		
		self.sess = sess                                            # tf.Session object
		self.input_shape = input_shape                              
		self.output_shape = output_shape 

		dataset_params = hparams['dataset']
		ret = dataset_loader.readDatasets(dataset_params['data_path'], output_shape[0], dataset_params['label_shuffle_seed'], 
											dataset_params['class_per_split'], dataset_params['task_weights'], 
											dataset_params['percent_validation'], 
											dataset_params['dataset_name'])
		self.split, self.num_tasks, self.task_weights, self.task_list, self.label_names = ret

		# used for masking outputs for each task
		for t in range(self.num_tasks):
			self.split[t].sort()
		self.cumulative_split = []
		for t in range(self.num_tasks):
			self.cumulative_split.append([])
			for i in range(t + 1):
				self.cumulative_split[t].extend(self.split[i])
			self.cumulative_split[t].sort()
		
		self.results_list = [None for _ in range(self.num_tasks)] 	# results (list of loss, accuracy) for each task

		self.hparams = hparams

		self.eval_frequency = 1 										# frequency of calculation of validation accuracy
		
		# create checkpoint and summaries directories if they don't exist
		if not os.path.exists(self.hparams['logging']['checkpoints_path']):
			os.makedirs(self.hparams['logging']['checkpoints_path'])
		if not os.path.exists(self.hparams['logging']['summaries_path']):
			os.makedirs(self.hparams['logging']['summaries_path'])

		# name of best hparams file, which stores best hparams of all tasks

		# classifier object
		self.classifier = Classifier(network, input_shape, output_shape, 
										hparams['classifier'][0], hparams['network'])

		# task after appending with examples to train set from old tasks
		self.appended_task_list = [None for _ in range(self.num_tasks)]

	def getLearningRate(self, base_lr_list, epoch):
		for i in range(len(base_lr_list) - 1):
			if (epoch < base_lr_list[i][0]):
				return base_lr_list[i][1]
		
		return base_lr_list[-1]

	# train on a given task with given hparams - hparams
	# only_penultimate_train available only in bf phase
	def train(self, t, hparams, logging_params, training_params, dataset_params, network_params, model_init_path, is_bf_phase=False):

		batch_size = training_params['batch_size']
		self.classifier.updateHparams(hparams)

		use_distill = t > 0 and (hparams['T'] is not None) and (not is_bf_phase)
		if (use_distill):
			self.classifier.createLossAccuracy(hparams['reweigh_points_loss'], T=hparams['T'], alpha=hparams['alpha'])
		else:
			self.classifier.createLossAccuracy(hparams['reweigh_points_loss'], T=None)
		
		print("Training with", hparams)
		# restore model from model_init_name and create train step

		if (not is_bf_phase):
			self.classifier.prepareForTraining(sess=self.sess,
												model_init_path=model_init_path)
		else:
			self.classifier.train_step = self.classifier.createTrainStep(self.sess, only_penultimate_train=hparams['only_penultimate_train'])

		if (network_params['mask_softmax']):
			self.setScoresMask(t)
		if (use_distill):
			self.setDistillMask(t - 1)

		summaries_path = os.path.join(logging_params['summaries_path'], str(t))
		if (not os.path.exists(summaries_path)):
			os.makedirs(summaries_path)
		else:
			shutil.rmtree(summaries_path)
			os.makedirs(summaries_path)

		if (logging_params['is_write_tensorboard']):
			writer = tensorboardX.SummaryWriter(log_dir=os.path.join(summaries_path, 'tensorboardX'))

		# variables to monitor training
		val_acc = [[] for _ in range(t + 1)]
		val_loss = [[] for _ in range(t + 1)]
		loss = [] 													
		train_acc = []
		
		i = 0 														# iteration number

		dataset_train = self.appended_task_list[t].train
		dataset_val = self.task_list[t].validation
		dataset_train.initializeIterator(batch_size)
		dataset_val.initializeIterator(batch_size)

		sample_random = is_bf_phase

		updates_per_epoch = math.ceil(self.task_list[t].train.images.shape[0] / batch_size) 	# number of train steps in an epoch
		if (not is_bf_phase):
			num_updates = training_params['epochs'] * updates_per_epoch
		else:
			num_updates = training_params['epochs_bf'] * updates_per_epoch
		epoch = 0
		# training loop
		while (True):
			if (i % updates_per_epoch == 0):
				cur_iter_weighted_avg = 0.0
				cur_iter_weights_sum = 0

				cur_val_loss, cur_val_acc = self.valTestAccuracy(t, VALIDATION_BATCH_SIZE, get_loss=True, type='validation')
				# calculating weighted average of accuracy
				for j in range(t + 1):
					val_loss[j].append(cur_val_loss[j])
					val_acc[j].append(cur_val_acc[j])
					cur_iter_weighted_avg += cur_val_acc[j] * self.task_weights[j]
					cur_iter_weights_sum += self.task_weights[j]
				cur_iter_weighted_avg /= cur_iter_weights_sum

				if (logging_params['is_write_tensorboard']):
					val_acc_dict = {str(j) : cur_val_acc[j] for j in range(t + 1)}
					val_acc_dict.update({'all_task_mean' : cur_iter_weighted_avg})
					writer.add_scalars('data/validation_accuracy', val_acc_dict, epoch)

				print("epoch: %d, iter: %d/%d, validation accuracies: %s, average train loss: %f, average train accuracy: %f" % 
						(epoch, i - epoch * updates_per_epoch, updates_per_epoch,
						str(np.array(val_acc)[:, -1]), 
						np.mean(loss[i - updates_per_epoch : ]), np.mean(train_acc[i - updates_per_epoch : ])))

				if (epoch >= math.ceil(num_updates / updates_per_epoch)):
					break

			next_batch = dataset_train.nextBatchSample(random_crop=dataset_params['random_crop_flip'], random_flip=dataset_params['random_crop_flip'], 
														epsilon=dataset_params['epsilon'], sample_random=sample_random)
			batch_xs, batch_ys, batch_weights, batch_final_outputs = next_batch

			if (network_params['mask_softmax']):
				batch_ys = batch_ys[:, self.active_outputs]
			if (not is_bf_phase):
				cur_lr = self.getLearningRate(hparams['learning_rate'][0], epoch)
			else:
				cur_lr = self.getLearningRate(hparams['learning_rate'][1], epoch)

			if (not use_distill):
				feed_dict = self.classifier.createFeedDict(batch_xs, batch_ys, learning_rate=cur_lr, weights=batch_weights, is_training=True)
			else:
				feed_dict = self.classifier.createFeedDict(batch_xs, batch_ys, learning_rate=cur_lr, weights=batch_weights, is_training=True, 
															teacher_outputs=batch_final_outputs)
			cur_loss, cur_accuracy = self.classifier.singleTrainStep(self.sess, feed_dict)
			
			loss.append(cur_loss)
			train_acc.append(cur_accuracy)
			
			# print stats if verbose
			if (self.hparams['logging']['verbose'] and (i % self.hparams['logging']['print_every'] == 0)):
				print("epoch: %d, iter: %d/%d, loss: %f, accuracy: %f" % 
						(epoch, i - epoch * updates_per_epoch, updates_per_epoch,
						loss[-1], train_acc[-1]))

			if (logging_params['is_write_tensorboard'] and (i % self.hparams['logging']['tensorboard_train_log_every'] == 0)):
				writer.add_scalar('data/train_loss', loss[-1], i)
				writer.add_scalar('data/train_acc', train_acc[-1], i)

			i += 1
			if (i % updates_per_epoch == 0):
				epoch += 1

			
			# USEFUL : code to add embedding
			# if (self.is_write_tensorboard and i % updates_per_epoch == 0):
			# 	feed_dict = self.classifier.createFeedDict(self.task_list[0].validation.images, self.task_list[0].validation.labels)
			# 	layer_output = self.classifier.getLayerOutput(self.sess, feed_dict, -2)
			# 	writer.add_embedding(layer_output, 
			# 							metadata=np.argmax(self.task_list[0].validation.labels, axis=1), 
			# 							global_step=epoch)

				
		total_updates = i

		if (logging_params['is_write_tensorboard']):
			writer.close()

		print("epochs: %f, final train loss: %f, validation accuracies: %s" % (epoch, loss[-1], str(np.array(val_acc)[:, -1])))

		ret = {}
		ret['val_acc'] = val_acc
		ret['val_loss'] = val_loss
		ret['loss'] = loss
		ret['acc'] = train_acc
		ret['updates_per_epoch'] = updates_per_epoch
		ret['total_updates'] = total_updates
		ret['total_epochs'] = epoch
		return ret
	
	def setScoresMask(self, task):
		active_outputs = self.cumulative_split[task]
		active_outputs_bool = np.zeros(self.output_shape[0], dtype=np.bool)
		active_outputs_bool[active_outputs] = True
		self.classifier.setScoresMask(self.sess, active_outputs_bool)
		self.active_outputs = self.cumulative_split[task]

	def setDistillMask(self, task):
		active_outputs = self.cumulative_split[task]
		active_outputs_bool = np.zeros(self.output_shape[0], dtype=np.bool)
		active_outputs_bool[active_outputs] = True
		self.classifier.setDistillMask(self.sess, active_outputs_bool)
		self.distill_active_outputs = self.cumulative_split[task]

	# train on a range of tasks sequentially [start, end];
	def trainTasksInRange(self):
		start, end = self.hparams['training']['start'], self.hparams['training']['end']
		batch_size = self.hparams['training']['batch_size']
		test_accuracies = []
		# requires model to have been trained on task (start - 1) with same hparams passed to this class
		for i in range(start, end + 1):
			model_init_path = None
			if (i > 0):
				model_init_path = os.path.join(self.hparams['logging']['checkpoints_path'], str(i - 1))

			old_new_ratio = self.hparams['dataset']['old:new']
			if (old_new_ratio > 0):
				if i == 0:
					self.appended_task_list[i] = self.task_list[i]
				else:
					checkpoints_path = self.hparams['logging']['checkpoints_path']
					with open(os.path.join(checkpoints_path, str(i - 1), 'penultimate_output.dat'), 'rb') as f:
						old_penultimate_output = pickle.load(f)
					
					self.classifier.restoreModel(self.sess, os.path.join(checkpoints_path, str(i - 1)))
					cur_penultimate_output = self.getLayerOutput(i, batch_size, -2)
					self.appended_task_list[i] = dataset.getAppendedTask(i, self.task_list, batch_size=batch_size, 
																			old_penultimate_output=old_penultimate_output,
																			cur_penultimate_output=cur_penultimate_output, 
																			dataset_params=self.hparams['dataset'])
			else:
				self.appended_task_list[i] = self.task_list[i]

			# if (self.hparams['logging']['is_write_tensorboard']):
			# 	summaries_path = self.hparams['logging']['summaries_path']
			# 	writer = tensorboardX.SummaryWriter(log_dir=os.path.join(summaries_path, 'tensorboardX'))
			# 	fc_wt = self.sess.run([v for v in tf.all_variables() if 'dense' in v.name and 'kernel:0' in v.name])
			# 	writer.add_embedding(fc_wt[0][:, self.split[0]].T, 
			# 						metadata=self.split[0], 
			# 						tag='weights-init')

			if (i > 0 and (self.hparams['classifier'][i]['T'] is not None)):
				self.appended_task_list[i].loadFinalOutput(self.getAllLayerOutput(i, batch_size, -1)[0][:, self.cumulative_split[i - 1]])

			cur_result = self.train(i, 
									hparams=self.hparams['classifier'][i], logging_params=self.hparams['logging'], 
									training_params=self.hparams['training'], dataset_params=self.hparams['dataset'], 
									network_params=self.hparams['network'], model_init_path=model_init_path, 
									is_bf_phase=False)
			
			# if (self.is_write_tensorboard):
			# 	fc_wt = self.sess.run([v for v in tf.all_variables() if 'dense' in v.name and 'kernel:0' in v.name])
			# 	writer.add_embedding(fc_wt[0][:, self.split[0]].T, 
			# 						metadata=self.split[0], 
			# 						tag='weights-before_bf')

			cur_result_1 = None
			if (self.hparams['training']['epochs_bf'] > 0):
				cur_result_1 = self.train(i, 
											hparams=self.hparams['classifier'][i], logging_params=self.hparams['logging'], 
											training_params=self.hparams['training'], dataset_params=self.hparams['dataset'], 
											network_params=self.hparams['network'], model_init_path=model_init_path, 
											is_bf_phase=True)

			# if (self.is_write_tensorboard):
			# 	fc_wt = self.sess.run([v for v in tf.all_variables() if 'dense' in v.name and 'kernel:0' in v.name])
			# 	writer.add_embedding(fc_wt[0][:, self.split[0]].T, 
			# 						metadata=self.split[0], 
			# 						tag='weights-after_bf')
			# 	writer.close()

			cur_result = (cur_result, cur_result_1)
			total_updates = cur_result[0]['total_updates']
			if (self.hparams['training']['epochs_bf'] > 0):
				total_updates += cur_result[1]['total_updates']
			
			if (not os.path.exists(os.path.join(self.hparams['logging']['checkpoints_path'], str(i)))):
				os.makedirs(os.path.join(self.hparams['logging']['checkpoints_path'], str(i)))
			self.classifier.saveWeights(total_updates, self.sess, os.path.join(self.hparams['logging']['checkpoints_path'], str(i)))
			
			self.saveResults(cur_result, os.path.join(self.hparams['logging']['summaries_path'], str(i), 'results_dict.dat'))
			self.results_list[i] = cur_result

			if (self.hparams['dataset']['save_penultimate_output']):
				self.savePenultimateOutput(i, batch_size)
				# self.saveFinalOutput(i, batch_size, hparams)

			if (self.hparams['logging']['eval_test_dataset']):
				test_accuracies.append(self.valTestAccuracy(i, batch_size, type='test'))
				with open(os.path.join(self.hparams['logging']['summaries_path'], str(i), 'test_accuracies.dat'), 'wb') as f:
					pickle.dump(test_accuracies[-1], f)

		if (self.hparams['logging']['eval_test_dataset']):
			return test_accuracies
		else:
			return (None, )


	def savePenultimateOutput(self, i ,batch_size):
		print("calculating penultimate output...")
		start_time = time.time()
		penultimate_output = self.getAllLayerOutput(i, batch_size, -2)
		print("time taken: %f", time.time() - start_time)
		print("saving penultimate output...")
		with open(os.path.join(self.hparams['logging']['checkpoints_path'], str(i), 'penultimate_output.dat'), 'wb') as f:
			pickle.dump(penultimate_output, f)

	def saveFinalOutput(self, i ,batch_size):
		print("calculating final output...")
		start_time = time.time()
		final_output = self.getAllLayerOutput(i, batch_size, -1)
		print("time taken: %f", time.time() - start_time)
		print("saving final output...")
		with open(os.path.join(self.hparams['logging']['checkpoints_path'], str(i), 'final_output.dat'), 'wb') as f:
			pickle.dump(final_output, f)

	# get layer output of all layers till 't' using current parameters of network
	def getAllLayerOutput(self, t, batch_size, layer_index):
		total_elements = sum([task.train.images.shape[0] for task in self.task_list[0: t + 1]])
		output_size = int(self.classifier.layer_output[layer_index].shape[-1])
		output = np.empty(shape=(total_elements, output_size))
		offset = 0
		for i in range(t + 1):
			cur_num_elements = self.task_list[i].train.images.shape[0]
			cur_output = self.getLayerOutput(i, batch_size, layer_index)
			output[offset: offset + cur_num_elements, :] = cur_output
			offset += cur_num_elements
		
		return output
	
	# get output for task 't' using current parameters of network
	def getLayerOutput(self, t, batch_size, layer_index):
		num_elements = self.task_list[t].train.images.shape[0]
		output_size = int(self.classifier.layer_output[layer_index].shape[-1])
		output = np.empty(shape=(num_elements, output_size))
		offset = 0
		num_batches = math.ceil(1.0 * self.task_list[t].train.images.shape[0] / batch_size)
		task_train = self.task_list[t].train
		task_train.initializeIterator(batch_size)
		for j in range(num_batches):
			batch_xs, batch_ys = task_train.nextBatch()
			feed_dict = self.classifier.createFeedDict(batch_xs, batch_ys)
			cur_output = self.classifier.getLayerOutput(self.sess, feed_dict, layer_index)
			output[offset: offset + batch_size, :] = cur_output
			offset += batch_size
		
		return output


	# validation loss, accuracy till task 't'
	def valTestAccuracy(self, t, batch_size, type='validation', restore_path=None, get_loss=False):
		if restore_path is not None:
			self.classifier.restoreModel(self.sess, restore_path)

		if (self.hparams['network']['mask_softmax']):
			self.setScoresMask(t)
		accuracy = [None for _ in range(t + 1)]
		loss = [None for _ in range(t + 1)]
		for i in range(t + 1):
			if (type == 'validation'):
				num_batches = math.ceil(self.task_list[i].validation.images.shape[0] / batch_size)
				dataset = self.task_list[i].validation
			elif (type == 'test'):
				num_batches = math.ceil(self.task_list[i].test.images.shape[0] / batch_size)
				dataset = self.task_list[i].test
			else:
				print("valTestAccuracy : wrong type. Exiting...")
				sys.exit(0)
			dataset.initializeIterator(batch_size)
			cur_accuracy = 0.0
			cur_loss = 0.0
			for j in range(num_batches):
				batch_xs, batch_ys = dataset.nextBatch()
				if (self.hparams['network']['mask_softmax']):
					batch_ys = batch_ys[:, self.active_outputs]
				feed_dict = self.classifier.createFeedDict(batch_xs, batch_ys)
				eval_result = self.classifier.evaluateLossAccuracy(self.sess, feed_dict)
				cur_loss += eval_result[0] * batch_xs.shape[0]
				cur_accuracy += eval_result[1] * batch_xs.shape[0]
			if (type == 'validation'):
				cur_loss /= self.task_list[i].validation.images.shape[0]
				cur_accuracy /= self.task_list[i].validation.images.shape[0]
			elif (type == 'test'):
				cur_loss /= self.task_list[i].test.images.shape[0]
				cur_accuracy /= self.task_list[i].test.images.shape[0]
			else:
				print("valTestAccuracy : wrong type. Exiting...")
				sys.exit(0)
			accuracy[i] = cur_accuracy
			loss[i] = cur_loss

		if (get_loss):
			return loss, accuracy
		else:
			return accuracy

	# utility functions to save/load self.results_list
	def saveResults(self, result, file_path):
		with open(file_path, 'wb') as f:
			pickle.dump(result, f)

	# USEFUL : might help to convert dictionary of params to tuple with fixed order
	# def hparamsDictToTuple(self, hparams, tuner_hparams):
	# 	hparams_tuple = [v for k, v in sorted(hparams.items())]
	# 	hparams_tuple = hparams_tuple + [v for k, v in sorted(tuner_hparams.items())]
	# 	return tuple(hparams_tuple)

	# USEFUL : code for training on a single task
	# # Train on task 't' with hparams in self.hparams_list[t] and find the best one ; calls train() internally ; ; make sure all previous tasks have been trained
	# def tuneOnTask(self, t, batch_size, model_init_name=None, num_updates=-1, verbose=False, restore_params=True, 
	# 				equal_weights=False, apply_log=False,
	# 				save_weights_every_epoch=False, save_weights_end=True, 
	# 				final_train=False, random_crop_flip=False, 
	# 				is_sampling_reweighing=True, 
	# 				do_bf_finetuning=False, num_updates_bf=-1, 
	# 				bf_only_penultimate_train=False, 
	# 				only_calc_appended_task=False, 
	# 				sigma=None):
	# 	best_avg = 0.0 								# best average validation accuracy and hparams corresponding to it from self.hparams_list[t]
	# 	best_hparams = None
	# 	if model_init_name is None: 				# model with which to initialize weights
	# 		if (restore_params):
	# 			model_init_name = self.best_hparams[t - 1][-1] if t > 0 else None

	# 	# calculate weights of examples of previous tasks and append examples to current tasks
	# 	old_new_ratio = self.tuner_hparams['old:new']
	# 	if (old_new_ratio > 0):
	# 		if (t == 0):
	# 			self.appended_task_list[0] = self.task_list[0]
	# 		else:
	# 			self.appended_task_list[t] = self.getAppendedTask(t, model_init_name, batch_size, optimize_space=True, equal_weights=equal_weights, 
	# 																apply_log=apply_log, is_sampling_reweighing=is_sampling_reweighing, sigma=sigma)
	# 	else:
	# 		self.appended_task_list[t] = self.task_list[t]

	# 	if (not restore_params):
	# 		model_init_name = None

	# 	if (only_calc_appended_task):
	# 		return None, None

	# 	if (t > 0 and ('T' in self.hparams_list[t][0].keys())):
	# 		self.appended_task_list[i].loadFinalOutput(self.getAllLayerOutput(t, batch_size, -1)[0][:, self.cumulative_split[t - 1]])

	# 	# loop through self.hparams_list[t], train with it and find the best one
	# 	for hparams in self.hparams_list[t]:
	# 		cur_result = self.train(t, hparams, batch_size, model_init_name, num_updates=num_updates, verbose=verbose, random_crop_flip=random_crop_flip, 
	# 								save_weights_every_epoch=save_weights_every_epoch)
	# 		cur_result_bf = None
	# 		if (do_bf_finetuning):
	# 			cur_result_bf = self.train(t, hparams, batch_size, model_init_name, num_updates=num_updates_bf, verbose=verbose, random_crop_flip=random_crop_flip, 
	# 										save_weights_every_epoch=save_weights_every_epoch, is_bf_finetuning_phase=True, 
	# 										only_penultimate_train=bf_only_penultimate_train)
			
	# 		cur_result = (cur_result, cur_result_bf)
			

	# 		if ((save_weights_end or (not final_train)) and (not save_weights_every_epoch)):
	# 			total_updates = cur_result[0]['total_updates']
	# 			if (do_bf_finetuning):
	# 				total_updates += cur_result[1]['total_updates']
	# 			self.classifier.saveWeights(total_updates, self.sess, self.fileName(t, hparams, self.tuner_hparams))
	# 		self.saveResults(cur_result, self.fileName(t, hparams, self.tuner_hparams))
			
	# 		cur_best_avg = cur_result[1]['best_avg'] if cur_result[1] is not None else cur_result[0]['best_avg']
	# 		hparams_tuple = self.hparamsDictToTuple(hparams, self.tuner_hparams)
	# 		self.results_list[t][hparams_tuple] = cur_result
			
	# 		if (cur_best_avg > best_avg):
	# 			best_avg = cur_best_avg
	# 			best_hparams = hparams
		
	# 	# update best hparams
	# 	if (self.best_hparams[t] is None):
	# 		self.best_hparams[t] = (best_hparams, self.tuner_hparams, self.fileName(t, best_hparams, self.tuner_hparams))
	# 	else:
	# 		prev_best_hparams_tuple = self.hparamsDictToTuple(self.best_hparams[t][0], self.best_hparams[t][1])
	# 		prev_best_avg = self.results_list[t][prev_best_hparams_tuple]['best_avg']
	# 		if (best_avg > prev_best_avg):
	# 			self.best_hparams[t] = (best_hparams, self.tuner_hparams, self.fileName(t, best_hparams, self.tuner_hparams))

	# 	# retrain model for best_avg_updates for best hparam to get optimal validation accuracy
	# 	# TODO : Currently finetuning doesn't work with final_train ; ie early stopping
	# 	if final_train:
	# 		best_hparams_tuple = self.hparamsDictToTuple(best_hparams, self.tuner_hparams)
	# 		cur_result = self.train(t, best_hparams, batch_size, model_init_name,
	# 								num_updates=self.results_list[t][best_hparams_tuple]['best_epoch'] * self.results_list[t][best_hparams_tuple]['updates_per_epoch'], 
	# 								random_crop_flip=random_crop_flip)
	# 		self.classifier.saveWeights(self.results_list[t][best_hparams_tuple]['best_epoch'] * self.results_list[t][best_hparams_tuple]['updates_per_epoch'], 
	# 										self.sess, self.fileName(t, best_hparams, self.tuner_hparams))

	# 	# calculate penultimate output of all tasks till 't' and save to file
	# 	if (self.save_penultimate_output):
	# 		self.savePenultimateOutput(t, batch_size, hparams)
	# 		# self.saveFinalOutput(t, batch_size, hparams)

	# 	return best_avg, best_hparams
