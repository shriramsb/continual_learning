import pickle
import matplotlib.pyplot as plt
import sys
import numpy as np

class_per_task = int(sys.argv[1])
summaries_path_prefix = 'summaries_' + sys.argv[1] + '_'

num_perms = 5

test_accuracies = []
for i in range(num_perms):
	with open(summaries_path_prefix + str(i) + '/test_accuracy', 'rb') as f:
		test_accuracies.append(pickle.load(f))
		print(test_accuracies[i][-1])
		for j in range(len(test_accuracies[i])):
			test_accuracies[i][j] = np.mean(test_accuracies[i][j])

test_accuracies = np.array(test_accuracies)
test_accuracies_mean = np.mean(test_accuracies, axis=0)
test_accuracies_std = np.std(test_accuracies, axis=0)
num_class = 100
print("Mean test accuracy across tasks :", np.mean(test_accuracies_mean))
# x = list(range(class_per_task, num_class + 1, class_per_task))
# plt.errorbar(x, test_accuracies_mean, yerr=test_accuracies_std)
# plt.savefig('test_' + str(class_per_task) + '.png')