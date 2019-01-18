import torch
import torch.utils.data
import numpy as np


class MNISTClassBiasedSampler(torch.utils.data.Sampler):
	def __init__(self, dataset, class_prob):
		train_labels = np.array([dataset[i][1].item() for i in range(len(dataset))])
		num_class = 10
		images_per_class = [np.sum(train_labels == i) for i in range(num_class)]

		self.weights = [0.0 for _ in range(len(dataset))]
		for i in range(len(dataset)):
			cur_label = dataset[i][1].item()
			self.weights[i] = class_prob[cur_label] / images_per_class[cur_label]

	def __iter__(self):
		return iter(i.item() for i in torch.multinomial(torch.DoubleTensor(self.weights), len(self.weights), replacement=True))

	def __len__(self):
		return len(self.weights)

class CIFARClassBiasedSampler(torch.utils.data.Sampler):
	def __init__(self, dataset, class_prob):
		train_labels = np.array([dataset[i][1] for i in range(len(dataset))])
		num_class = 10
		images_per_class = [np.sum(train_labels == i) for i in range(num_class)]

		self.weights = [0.0 for _ in range(len(dataset))]
		for i in range(len(dataset)):
			cur_label = dataset[i][1]
			self.weights[i] = class_prob[cur_label] / images_per_class[cur_label]

	def __iter__(self):
		return iter(i.item() for i in torch.multinomial(torch.DoubleTensor(self.weights), len(self.weights), replacement=True))

	def __len__(self):
		return len(self.weights)