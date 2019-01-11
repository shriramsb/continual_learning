import pickle
import matplotlib.pyplot as plt
import sys
import numpy as np

class_per_task = int(sys.argv[1])
summaries_path_prefix = 'summaries_' + sys.argv[1] + '_'

num_perms = 5
i = int(sys.argv[2])

results_list = None
with open(summaries_path_prefix + str(i) + '/all_results', 'rb') as f:
	results_list =pickle.load(f)

task = int(sys.argv[3])
val_acc_mean = []
print("Training phase")
for k, v in results_list[task].items():
	val_acc = np.array(v[0]['val_acc'])
	for i in range(val_acc.shape[1]):
		print("epoch :", i + 1, val_acc[:, i])
		val_acc_mean.append(np.mean(val_acc[:, i]))

training_epochs = val_acc.shape[1]
print("Balanced finetuning phase")
for k, v in results_list[task].items():
	val_acc = np.array(v[1]['val_acc'])
	for i in range(val_acc.shape[1]):
		print("epoch :", training_epochs + i + 1, val_acc[:, i])
		val_acc_mean.append(np.mean(val_acc[:, i]))

plt.plot(val_acc_mean)
plt.savefig('val_acc.png')

