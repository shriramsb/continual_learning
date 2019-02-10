import numpy as np
import pickle
from copy import deepcopy
from src.dataset import MyTask
import os

def splitDataset(dataset, dataset_split):
	task_list = []
	train_labels = np.argmax(dataset.train.labels, axis=1)
	validation_labels = np.argmax(dataset.validation.labels, axis=1)
	test_labels = np.argmax(dataset.test.labels, axis=1)
	for i in range(len(dataset_split)):
		cur_train_indices = np.isin(train_labels, dataset_split[i])
		cur_validation_indices = np.isin(validation_labels, dataset_split[i])
		cur_test_indices = np.isin(test_labels, dataset_split[i])

		task = deepcopy(dataset)
		task.train.images = task.train.images[cur_train_indices]
		task.train.labels = task.train.labels[cur_train_indices]
		task.validation.images = task.validation.images[cur_validation_indices]
		task.validation.labels = task.validation.labels[cur_validation_indices]
		task.test.images = task.test.images[cur_test_indices]
		task.test.labels = task.test.labels[cur_test_indices]
		task = MyTask(task)
		task_list.append(task)

	return task_list
	
def smoothLabels(dataset, label_smooth_param):
	train_labels = dataset.train.labels
	train_labels_argmax = np.argmax(train_labels, axis=1)
	train_labels = train_labels + label_smooth_param / (train_labels.shape[1] - 1)
	train_labels[range(train_labels.shape[0]), train_labels_argmax] = 1 - label_smooth_param
	dataset.train._labels = train_labels

class TempDataset(object):
	def __init__(self):
		self.images = None
		self.labels = None
	
class TempTask(object):
	def __init__(self):
		self.train = TempDataset()
		self.validation = TempDataset()
		self.test = TempDataset()
	
	
def cifar100Loader(data_path, num_class):
	"""
		Returns (train_data, train_labels(one-hot), test_data, test_labels(one-hot), label_names) in the format NHWC, after normalizing with whole of train data
	"""

	with open(os.path.join(data_path, 'train'), 'rb') as f:
		f_train_data = pickle.load(f, encoding='bytes')
		
	with open(os.path.join(data_path, 'test'), 'rb') as f:
		f_test_data = pickle.load(f, encoding='bytes')

	with open(os.path.join(data_path, 'meta'), 'rb') as f:
		f_label_names = pickle.load(f)

	temp_train_labels = np.array(f_train_data[b'fine_labels'], dtype=np.int32)
	temp_test_labels = np.array(f_test_data[b'fine_labels'], dtype=np.int32)
	f_train_data[b'fine_labels'] = np.zeros((temp_train_labels.shape[0], num_class))
	(f_train_data[b'fine_labels'])[range(temp_train_labels.shape[0]), temp_train_labels] = 1
	f_test_data[b'fine_labels'] = np.zeros((temp_test_labels.shape[0], num_class))
	(f_test_data[b'fine_labels'])[range(temp_test_labels.shape[0]), temp_test_labels] = 1
	f_train_data[b'data'] = np.reshape(f_train_data[b'data'], (-1, 3, 32, 32))
	f_test_data[b'data'] = np.reshape(f_test_data[b'data'], (-1, 3, 32, 32))
	f_train_data[b'data'] = np.transpose(f_train_data[b'data'], (0, 2, 3, 1))
	f_test_data[b'data'] = np.transpose(f_test_data[b'data'], (0, 2, 3, 1))
	
	tr_data = f_train_data[b'data']
	te_data = f_test_data[b'data']
	# normalizing data
	avg = np.mean(tr_data, axis=(0, 1, 2))
	std = np.std(tr_data, axis=(0, 1, 2))
	
	f_train_data[b'data'] = (tr_data - avg) / std
	f_test_data[b'data'] = (te_data - avg) / std
	
	return f_train_data[b'data'], f_train_data[b'fine_labels'], f_test_data[b'data'], f_test_data[b'fine_labels'], f_label_names['fine_label_names']



def readDatasets(data_path, num_class, label_shuffle_seed, 
					class_per_split, task_weights, 
					percent_validation, 
					dataset='cifar-100', 
					label_smooth_param=0):
	labels_list = list(range(num_class))
	np.random.seed(label_shuffle_seed)
	np.random.shuffle(labels_list)
	
	split = []
	pos = 0
	for i in range(len(class_per_split)):
		split.append(labels_list[pos : pos + class_per_split[i]])
		pos += class_per_split[i]

	num_tasks = len(split)

	if (dataset == 'cifar-100'):
		task = TempTask()
		train_data, train_labels, test_data, test_labels, label_names = cifar100Loader(data_path, num_class)
	
	# CHECK : assuming equal examples per class
	num_val_per_class = int(percent_validation / 100.0 * train_data.shape[0] / num_class)
	
	for i in range(num_class):
		pos = (np.argmax(train_labels, axis=1) == i)
		
		if (i == 0):
			task.validation.images = (train_data[pos])[0 : num_val_per_class]
			task.validation.labels = (train_labels[pos])[0 : num_val_per_class]

			task.train.images = (train_data[pos])[num_val_per_class : ]
			task.train.labels = (train_labels[pos])[num_val_per_class : ]
		else:
			task.validation.images = np.concatenate((task.validation.images, (train_data[pos])[0 : num_val_per_class]))
			task.validation.labels = np.concatenate((task.validation.labels, (train_labels[pos])[0 : num_val_per_class]))

			task.train.images = np.concatenate((task.train.images, (train_data[pos])[num_val_per_class : ]))
			task.train.labels = np.concatenate((task.train.labels, (train_labels[pos])[num_val_per_class : ]))
		
	task.test.images = test_data
	task.test.labels = test_labels
	
	# CHECK : haven't checked if it works
	if (label_smooth_param != 0):
		smoothLabels(task, label_smooth_param)
		
	task_list = splitDataset(task, split)

	for i in range(len(task_list)):
		shuffle_indices = np.random.permutation(task_list[i].train.images.shape[0])
		task_list[i].train.images = task_list[i].train.images[shuffle_indices]
		task_list[i].train.labels = task_list[i].train.labels[shuffle_indices]

	return split, num_tasks, task_weights, task_list, label_names