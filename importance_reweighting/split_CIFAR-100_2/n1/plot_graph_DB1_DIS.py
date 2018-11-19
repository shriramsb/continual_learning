import pickle
import matplotlib.pyplot as plt
import sys
import numpy as np

class_per_task = int(sys.argv[1])
summaries_path_prefix = 'summaries_' + sys.argv[1] + '_'

num_perms = 5

test_accuracies_DB1 = []
for i in range(num_perms):
	with open('DB1/' + summaries_path_prefix + str(i) + '/test_accuracy', 'rb') as f:
		test_accuracies_DB1.append(pickle.load(f))

test_accuracies_DIS = []
for i in range(num_perms):
	with open('DIS/' + summaries_path_prefix + str(i) + '/test_accuracy', 'rb') as f:
		test_accuracies_DIS.append(pickle.load(f))

test_accuracies_DB1 = np.array(test_accuracies_DB1)
test_accuracies_mean_DB1 = np.mean(test_accuracies_DB1, axis=0)
test_accuracies_std_DB1 = np.std(test_accuracies_DB1, axis=0)
test_accuracies_DIS = np.array(test_accuracies_DIS)
test_accuracies_mean_DIS = np.mean(test_accuracies_DIS, axis=0)
test_accuracies_std_DIS = np.std(test_accuracies_DIS, axis=0)
num_class = 100
x = list(range(class_per_task, num_class + 1, class_per_task))
plt.errorbar(x, test_accuracies_mean_DB1, yerr=test_accuracies_std_DB1, label='DB1')
plt.errorbar(x, test_accuracies_mean_DIS, yerr=test_accuracies_std_DIS, label='DIS')
plt.legend()
plt.savefig('test_' + str(class_per_task) + '.png')