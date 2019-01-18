import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

def trainStep(network, criterion, optimizer, X, y):
	"""
	One training step of the network: forward prop + backprop + update parameters
	Return: (loss, accuracy) of current batch
	"""
	optimizer.zero_grad()
	outputs = network(X)
	loss = criterion(outputs, y)
	loss.backward()
	optimizer.step()
	accuracy = float(torch.sum(torch.argmax(outputs, dim=1) == y).item()) / y.shape[0]
	return loss, accuracy

def getLossAccuracyOnDataset(network, dataset_loader, fast_device, criterion=None):
	"""
	Returns (loss, accuracy) of network on given dataset
	"""
	network.is_training = False
	accuracy = 0.0
	loss = 0.0
	dataset_size = 0
	for j, D in enumerate(dataset_loader, 0):
		X, y = D
		X = X.to(fast_device)
		y = y.to(fast_device)
		with torch.no_grad():
			pred = network(X)
			if criterion is not None:
				loss += criterion(pred, y) * y.shape[0]
			accuracy += torch.sum(torch.argmax(pred, dim=1) == y).item()
		dataset_size += y.shape[0]
	loss, accuracy = loss / dataset_size, accuracy / dataset_size
	network.is_training = True
	return loss, accuracy

def getConfusionMatrix(network, dataset_loader, fast_device, num_class):
	"""
	Return: Confusion matrix of network's prediction on given dataset
	"""
	network.is_training = False
	confusion_matrix = [[0 for _ in range(num_class)] for _ in range(num_class)]
	dataset_size = 0
	for j, D in enumerate(dataset_loader, 0):
		X, y = D
		X = X.to(fast_device)
		with torch.no_grad():
			pred = network(X)
			pred = torch.argmax(pred, dim=1)
			pred = pred.cpu()
			pred = pred.numpy()
			y = y.numpy()
			for i in range(y.shape[0]):
				confusion_matrix[y[i]][pred[i]] += 1
	return confusion_matrix

def printConfusionMatrix(confusion_matrix):
	num_class = len(confusion_matrix)
	print('%3d ' %(-1, ), end='')
	for i in range(num_class):
		print('%3d ' %(i, ), end='')
	print('')
	for i in range(num_class):
		print('%3d ' % (i, ), end='')
		for j in range(num_class):
			print('%3d ' % (confusion_matrix[i][j], ), end='')
		print('')

def trainTeacherOnHparam(teacher_net, hparam, num_epochs, 
						train_loader, val_loader, 
						print_every=0, 
						fast_device=torch.device('cpu')):
	"""
	Trains teacher on given hyperparameters for given number of epochs; Pass val_loader=None when not required to validate for every epoch 
	Return: List of training loss, accuracy for each update calculated only on the batch; List of validation loss, accuracy for each epoch
	"""
	train_loss_list, train_acc_list, val_loss_list, val_acc_list = [], [], [], []
	teacher_net.dropout_input = hparam['dropout_input']
	teacher_net.dropout_hidden = hparam['dropout_hidden']
	criterion = nn.CrossEntropyLoss()
	optimizer = optim.SGD(teacher_net.parameters(), lr=hparam['lr'], momentum=hparam['momentum'], weight_decay=hparam['weight_decay'])
	lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=hparam['lr_decay'])
	for epoch in range(num_epochs):
		lr_scheduler.step()
		if epoch == 0:
			if val_loader is not None:
				val_loss, val_acc = getLossAccuracyOnDataset(teacher_net, val_loader, fast_device, criterion)
				val_loss_list.append(val_loss)
				val_acc_list.append(val_acc)
				print('epoch: %d validation loss: %.3f validation accuracy: %.3f' %(epoch, val_loss, val_acc))
		for i, data in enumerate(train_loader, 0):
			X, y = data
			X, y = X.to(fast_device), y.to(fast_device)
			loss, acc = trainStep(teacher_net, criterion, optimizer, X, y)
			train_loss_list.append(loss)
			train_acc_list.append(acc)
		
			if print_every > 0 and i % print_every == print_every - 1:
				print('[%d, %5d/%5d] train loss: %.3f train accuracy: %.3f' %
					  (epoch + 1, i + 1, len(train_loader), loss, acc))
		
		if val_loader is not None:
			val_loss, val_acc = getLossAccuracyOnDataset(teacher_net, val_loader, fast_device, criterion)
			val_loss_list.append(val_loss)
			val_acc_list.append(val_acc)
			print('epoch: %d validation loss: %.3f validation accuracy: %.3f' %(epoch + 1, val_loss, val_acc))
	return {'train_loss': train_loss_list, 
			'train_acc': train_acc_list, 
			'val_loss': val_loss_list, 
			'val_acc': val_acc_list}

def studentTrainStep(teacher_net, student_net, studentLossFn, optimizer, X, y, T, alpha):
	"""
	One training step of student network: forward prop + backprop + update parameters
	Return: (loss, accuracy) of current batch
	"""
	optimizer.zero_grad()
	teacher_pred = None
	if (alpha > 0):
		with torch.no_grad():
			teacher_pred = teacher_net(X)
	student_pred = student_net(X)
	loss = studentLossFn(teacher_pred, student_pred, y, T, alpha)
	loss.backward()
	optimizer.step()
	accuracy = float(torch.sum(torch.argmax(student_pred, dim=1) == y).item()) / y.shape[0]
	return loss, accuracy

def trainStudentOnHparam(teacher_net, student_net, hparam, num_epochs, 
						train_loader, val_loader, 
						print_every=0, 
						fast_device=torch.device('cpu'), 
						only_penultimate_train=False):
	"""
	Trains teacher on given hyperparameters for given number of epochs; Pass val_loader=None when not required to validate for every epoch
	Return: List of training loss, accuracy for each update calculated only on the batch; List of validation loss, accuracy for each epoch
	"""
	train_loss_list, train_acc_list, val_acc_list = [], [], []
	T = hparam['T']
	alpha = hparam['alpha']
	student_net.dropout_input = hparam['dropout_input']
	student_net.dropout_hidden = hparam['dropout_hidden']
	if (only_penultimate_train):
		for param in student_net.parameters():
			param.requires_grad_(False)
		student_net.all_layers[-1].weight.requires_grad_(True)
		student_net.all_layers[-1].bias.requires_grad_(True)

	optimizer = optim.SGD(filter(lambda p: p.requires_grad, student_net.parameters()), lr=hparam['lr'], momentum=hparam['momentum'], weight_decay=hparam['weight_decay'])
	lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=hparam['lr_decay'])

	def studentLossFn(teacher_pred, student_pred, y, T, alpha):
		"""
		Loss function for student network: Loss = alpha * (distillation loss with soft-target) + (1 - alpha) * (cross-entropy loss with true label)
		Return: loss
		"""
		if (alpha > 0):
			loss = F.kl_div(F.log_softmax(student_pred / T, dim=1), F.softmax(teacher_pred / T, dim=1), reduction='batchmean') * (T ** 2) * alpha + F.cross_entropy(student_pred, y) * (1 - alpha)
		else:
			loss = F.cross_entropy(student_pred, y)
		return loss

	for epoch in range(num_epochs):
		lr_scheduler.step()
		if epoch == 0:
			if val_loader is not None:
				_, val_acc = getLossAccuracyOnDataset(student_net, val_loader, fast_device)
				val_acc_list.append(val_acc)
				print('epoch: %d validation accuracy: %.3f' %(epoch, val_acc))
		for i, data in enumerate(train_loader, 0):
			X, y = data
			X, y = X.to(fast_device), y.to(fast_device)
			loss, acc = studentTrainStep(teacher_net, student_net, studentLossFn, optimizer, X, y, T, alpha)
			train_loss_list.append(loss)
			train_acc_list.append(acc)
		
			if print_every > 0 and i % print_every == print_every - 1:
				print('[%d, %5d/%5d] train loss: %.3f train accuracy: %.3f' %
					  (epoch + 1, i + 1, len(train_loader), loss, acc))
	
		if val_loader is not None:
			_, val_acc = getLossAccuracyOnDataset(student_net, val_loader, fast_device)
			val_acc_list.append(val_acc)
			print('epoch: %d validation accuracy: %.3f' %(epoch + 1, val_acc))

	if (only_penultimate_train):
		for param in student_net.parameters():
			param.requires_grad_(True)

	return {'train_loss': train_loss_list, 
			'train_acc': train_acc_list, 
			'val_acc': val_acc_list}

def hparamToString(hparam):
	"""
	Convert hparam dictionary to string with deterministic order of attribute of hparam in output string
	"""
	hparam_str = ''
	for k, v in sorted(hparam.items()):
		hparam_str += k + '=' + str(v) + ', '
	return hparam_str[:-2]

def hparamDictToTuple(hparam):
	"""
	Convert hparam dictionary to tuple with deterministic order of attribute of hparam in output tuple
	"""
	hparam_tuple = [v for k, v in sorted(hparam.items())]
	return tuple(hparam_tuple)

def getTrainMetricPerEpoch(train_metric, updates_per_epoch):
	"""
	Smooth the training metric calculated for each batch of training set by averaging over batches in an epoch
	Input: List of training metric calculated for each batch
	Output: List of training matric averaged over each epoch
	"""
	train_metric_per_epoch = []
	temp_sum = 0.0
	for i in range(len(train_metric)):
		temp_sum += train_metric[i]
		if (i % updates_per_epoch == updates_per_epoch - 1):
			train_metric_per_epoch.append(temp_sum / updates_per_epoch)
			temp_sum = 0.0

	return train_metric_per_epoch