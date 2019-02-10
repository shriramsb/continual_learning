import numpy as np
import torch
import tensorflow as tf


# Dataset - helps access dataset batch-wise
class MyDataset(object):
	def __init__(self, images, labels, weights=None, new_task_size=None):
		self.images = images
		self.labels = labels
		self.batch_size = 0

		# sampling weight for each train data point 
		if weights is not None:
			self.weights = weights / np.sum(weights)
		else:
			self.weights = weights
		self.pos = 0                             	# pointer storing position starting from which to return data for nextBatch()

		if new_task_size is None or weights is None:
			self.old_task_equalized_weights = np.array([1 / images.shape[0] for _ in range(images.shape[0])])
		else:
			# old images appear before new images
			self.old_task_equalized_weights = np.empty(weights.shape)
			self.old_task_equalized_weights[-new_task_size : ] = weights[-new_task_size : ]
			old_task_size = images.shape[0] - new_task_size
			self.old_task_equalized_weights[ : old_task_size] = (1.0 - np.sum(weights[-new_task_size : ])) / old_task_size

		self.final_outputs = None

	# modify batch size and position pointer to the start of dataset 
	def initializeIterator(self, batch_size):
		self.batch_size = batch_size
		self.pos = 0

	# get next batch, in round-robin fashion
	def nextBatch(self, random_flip=False, random_crop=False):
		# last batch might be smaller than self.batch_size
		if (self.pos + self.batch_size >= self.images.shape[0]):
			ret = self.images[self.pos : ], self.labels[self.pos : ]
			self.pos = 0
		else:
			ret = self.images[self.pos : self.pos + self.batch_size], self.labels[self.pos : self.pos + self.batch_size]
			self.pos = self.pos + self.batch_size

		if random_flip:
			rnd = np.random.random(size=(ret[0].shape[0], ))
			for i in range(ret[0].shape[0]):
				if (rnd[i] < 0.5):
					ret[0][i] = np.flip(ret[0][i], axis=1) 		# assumed (N, H, W, C) format
		
		if random_crop:
			rnd = np.random.random(size=(ret[0].shape[0], ))
			for i in range(ret[0].shape[0]):
				if (rnd[i] < 0.5):
					ret[0][i] = self.randomCrop(ret[0][i], 4) 	# 4 - hard-coded

		return ret

	# Pads image by given pad and does random crop back to original size - assumed (N, H, W, C) format
	def randomCrop(self, image, pad):
		padded_image = np.pad(image, [(pad, pad), (pad, pad), (0, 0)], 'constant')
		r = np.random.random_integers(0, 2 * pad, size=(2, ))
		padded_image = padded_image[r[0] : r[0] + image.shape[0], r[1] : r[1] + image.shape[1]]
		return padded_image

	# sample next batch according to weights
	# with epsilon probability, samples uniformly
	def nextBatchSample(self, random_flip=False, random_crop=False, epsilon=0.0, sample_random=False):
		if (sample_random):
			total_examples = self.images.shape[0]
			sampled_indices = np.random.choice(range(total_examples), size=self.batch_size)
			batch_xs = self.images[sampled_indices]
			batch_ys = self.labels[sampled_indices]
			batch_weights = np.ones((self.batch_size, ), dtype=np.float32)
			batch_final_outputs = None
			if (self.final_outputs is not None):
				batch_final_outputs = self.final_outputs[sampled_indices]

		else:
			total_examples = self.images.shape[0]
			num_uniform_samples = int(self.batch_size * epsilon)
			num_selective_samples = self.batch_size - num_uniform_samples
			batch_xs = np.empty(tuple([self.batch_size] + list(self.images.shape[1 : ])))
			batch_ys = np.empty(tuple([self.batch_size] + list(self.labels.shape[1 : ])))
			batch_weights = np.empty((self.batch_size, ))
			if (self.final_outputs is not None):
				batch_final_outputs = np.empty(tuple([self.batch_size] + list(self.final_outputs.shape[1 : ])))
			else:
				batch_final_outputs = None

			if (num_uniform_samples > 0):
				sampled_indices = np.random.choice(range(total_examples), p=self.old_task_equalized_weights, size=num_uniform_samples)
				batch_xs[ : num_uniform_samples] = self.images[sampled_indices]
				batch_ys[ : num_uniform_samples] = self.labels[sampled_indices]
				batch_weights[ : num_uniform_samples] = 1.0 / self.old_task_equalized_weights[sampled_indices]
				if (self.final_outputs is not None):
					batch_final_outputs[ : num_uniform_samples] = self.final_outputs[sampled_indices]

			if (num_selective_samples > 0):
				sampled_indices = np.random.choice(range(total_examples), p = self.weights, size=num_selective_samples)
				batch_xs[num_uniform_samples : ] = self.images[sampled_indices]
				batch_ys[num_uniform_samples : ] = self.labels[sampled_indices]
				batch_weights[num_uniform_samples : ] = 1.0 / self.weights[sampled_indices]
				if (self.final_outputs is not None):
					batch_final_outputs[num_uniform_samples : ] = self.final_outputs[sampled_indices]

			if random_flip:
				rnd = np.random.random(size=(batch_xs.shape[0], ))
				for i in range(batch_xs.shape[0]):
					if (rnd[i] < 0.5):
						batch_xs[i] = np.flip(batch_xs[i], axis=1) 		# assumed (N, H, W, C) format
			
			if random_crop:
				for i in range(batch_xs.shape[0]):
					rnd = np.random.random(size=(batch_xs.shape[0], ))
					if (rnd[i] < 0.5):
						batch_xs[i] = self.randomCrop(batch_xs[i], 4) 	# 4 - hard-coded

		return batch_xs, batch_ys, batch_weights, batch_final_outputs

	# get data in [start, end)
	def getData(self, start, end):
		return self.images[start: end], self.labels[start: end]

# Task having dataset for train, dev, test
class MyTask(object):
	def __init__(self, task, train_images=None, train_labels=None, weights=None, new_task_size=None):        
		# MNIST dataset from tf loaded dataset
		if (type(task) == tf.contrib.learn.datasets.base.Datasets):
			weights = np.array([1 for _ in range(task.train._images.shape[0])]) / task.train._images.shape[0]
			self.train = MyDataset(task.train._images, task.train._labels, weights, new_task_size=new_task_size)
			self.validation = MyDataset(task.validation._images, task.validation._labels)
			self.test = MyDataset(task.test._images, task.test._labels)
		else:
			if train_images is None:
				if weights is None:
					weights = np.array([1 for _ in range(task.train.images.shape[0])]) / task.train.images.shape[0]
				self.train = MyDataset(task.train.images, task.train.labels, weights, new_task_size=new_task_size)
				self.validation = MyDataset(task.validation.images, task.validation.labels)
				self.test = MyDataset(task.test.images, task.test.labels)
			else:
				if weights is None:
					weights = np.array([1 for _ in range(train_images.shape[0])]) / train_images.shape[0]
				self.train = MyDataset(train_images, train_labels, weights, new_task_size=new_task_size)
				self.validation = MyDataset(task.validation.images, task.validation.labels)
				self.test = MyDataset(task.test.images, task.test.labels)

	def loadFinalOutput(self, final_output):
		self.train.final_outputs = final_output


# To be added before call to getAppendedTask()
# if (model_init_name is not None):	
# 			self.classifier.restoreModel(self.sess, model_init_name)
# 		# load penultimate output of previous tasks' examples for model_init_name 
# 		with open(self.checkpoint_path + model_init_name + '_penultimate_output.txt', 'rb') as f:
# 			old_penultimate_output, old_taskid_offset = pickle.load(f)
		
# 		# penultimate output of current task for model_init_name
# 		cur_penultimate_output, _ = self.getLayerOutput(t, batch_size, -2)

# append task with examples from previous tasks
def getAppendedTask(t, task_list, batch_size=None, old_penultimate_output=None, cur_penultimate_output=None, dataset_params=None):
	
	appended_task = None
	
	# if equal_weights is True, then assign equal weights to all points till current task. 
	# Otherwise, weigh each point of previous tasks by cosine similarity to current task's points
	if (not dataset_params['equal_weights']):
		cur_penultimate_output_norm = np.sqrt(np.sum((cur_penultimate_output ** 2), axis=1))
		old_penultimate_output_norm = np.sqrt(np.sum((old_penultimate_output ** 2), axis=1))
		
		# similarity - each row gives cosine similarity between an example from current task with all examples from previous tasks
		similarity = np.empty((cur_penultimate_output.shape[0], old_penultimate_output.shape[0]), np.float32)
		# use gpu for multiplication if available ; using PyTorch here
		if (dataset_params['use_gpu']):
			b = torch.Tensor(old_penultimate_output.T).cuda()
		for i in range(cur_penultimate_output.shape[0]):
			if (dataset_params['use_gpu']):
				a = torch.Tensor(np.expand_dims(cur_penultimate_output[i], axis=0)).cuda()
				similarity[i] = torch.mm(a, b).cpu()
				del a
			else:
				similarity[i] = np.matmul(cur_penultimate_output[i], old_penultimate_output.T)
			similarity[i] = similarity[i] / old_penultimate_output_norm / cur_penultimate_output_norm[i]
			if (dataset_params['sigma'] is not None):
				similarity[i] = torch.nn.functional.softmax(torch.from_numpy(similarity[i] * sigma), dim=0).numpy()
		if (dataset_params['use_gpu']):
			del b

		train_task = task_list[t].train

		old_new_ratio = dataset_params['old:new']
		
		old_task_weights = np.sum(similarity, axis=0)
		old_task_weights = old_task_weights / np.sum(old_task_weights)
		old_task_weights = old_task_weights * old_new_ratio / (old_new_ratio + 1)
		cur_task_weights = np.array([1.0 / (old_new_ratio + 1) for _ in range(train_task.images.shape[0])]) / train_task.images.shape[0]

		appended_images_shape = tuple([train_task.images.shape[0] + old_penultimate_output.shape[0]] + list(train_task.images.shape)[1: ])
		appended_labels_shape = tuple([train_task.labels.shape[0] + old_penultimate_output.shape[0]] + list(train_task.labels.shape)[1: ])
		appended_weights_shape = tuple([train_task.labels.shape[0] + old_penultimate_output.shape[0]])

		appended_images = np.empty(appended_images_shape)
		appended_labels = np.empty(appended_labels_shape)
		appended_weights = np.empty(appended_weights_shape)

		offset = 0
		for i in range(t + 1):
			appended_images[offset : offset + task_list[i].train.images.shape[0]] = task_list[i].train.images
			appended_labels[offset : offset + task_list[i].train.labels.shape[0]] = task_list[i].train.labels
			offset += task_list[i].train.images.shape[0]
		
		appended_weights[0 : old_task_weights.shape[0]] = old_task_weights
		appended_weights[ old_task_weights.shape[0] : ] = cur_task_weights
		
		appended_task = MyTask(task_list[t], train_images=appended_images, train_labels=appended_labels, weights=appended_weights, 
								new_task_size=cur_task_weights.shape[0])

	
	else:
		train_task = task_list[t].train
		num_elements = 0
		for i in range(t + 1):
			num_elements += task_list[i].train.images.shape[0]

		appended_images_shape = tuple([num_elements] + list(train_task.images.shape)[1: ])
		appended_labels_shape = tuple([num_elements] + list(train_task.labels.shape)[1: ])
		appended_weights_shape = tuple([num_elements])            
		
		appended_images = np.empty(appended_images_shape)
		appended_labels = np.empty(appended_labels_shape)
		appended_weights = np.empty(appended_weights_shape)

		offset = 0
		for i in range(t + 1):
			appended_images[offset : offset + task_list[i].train.images.shape[0]] = task_list[i].train.images
			appended_labels[offset : offset + task_list[i].train.labels.shape[0]] = task_list[i].train.labels
			offset += task_list[i].train.images.shape[0]
		
		appended_weights[ : ] = 1.0 / num_elements
		appended_task = MyTask(task_list[t], train_images=appended_images, train_labels=appended_labels, weights=appended_weights)

	
	return appended_task