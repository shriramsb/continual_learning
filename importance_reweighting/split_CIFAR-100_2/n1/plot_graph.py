import pickle
import matplotlib.pyplot as plt
import sys
import numpy as np

class_per_task = int(sys.argv[1])
summaries_path_prefix = 'summaries_' + sys.argv[1] + '_'

num_perms = 5

test_accuracies_DIS1 = []
for i in range(num_perms):
	with open('DIS1/' + summaries_path_prefix + str(i) + '/test_accuracy', 'rb') as f:
		test_accuracies_DIS1.append(pickle.load(f))
	print(test_accuracies_DIS1[i][-1])
	for j in range(len(test_accuracies_DIS1[i])):
		test_accuracies_DIS1[i][j] = np.mean(test_accuracies_DIS1[i][j])

test_accuracies_DIS1_2 = []
for i in range(num_perms):
	with open('DIS1_2/' + summaries_path_prefix + str(i) + '/test_accuracy', 'rb') as f:
		test_accuracies_DIS1_2.append(pickle.load(f))
	print(test_accuracies_DIS1_2[i][-1])
	for j in range(len(test_accuracies_DIS1_2[i])):
		test_accuracies_DIS1_2[i][j] = np.mean(test_accuracies_DIS1_2[i][j])

test_accuracies_DIS1_3 = []
for i in range(num_perms):
	with open('DIS1_3/' + summaries_path_prefix + str(i) + '/test_accuracy', 'rb') as f:
		test_accuracies_DIS1_3.append(pickle.load(f))
	print(test_accuracies_DIS1_3[i][-1])
	for j in range(len(test_accuracies_DIS1_3[i])):
		test_accuracies_DIS1_3[i][j] = np.mean(test_accuracies_DIS1_3[i][j])

test_accuracies_DIS1 = np.array(test_accuracies_DIS1)
test_accuracies_mean_DIS1 = np.mean(test_accuracies_DIS1, axis=0)
test_accuracies_std_DIS1 = np.std(test_accuracies_DIS1, axis=0)

test_accuracies_DIS1_2 = np.array(test_accuracies_DIS1_2)
test_accuracies_mean_DIS1_2 = np.mean(test_accuracies_DIS1_2, axis=0)
test_accuracies_std_DIS1_2 = np.std(test_accuracies_DIS1_2, axis=0)

test_accuracies_DIS1_3 = np.array(test_accuracies_DIS1_3)
test_accuracies_mean_DIS1_3 = np.mean(test_accuracies_DIS1_3, axis=0)
test_accuracies_std_DIS1_3 = np.std(test_accuracies_DIS1_3, axis=0)
num_class = 100
# print("Mean test accuracy across tasks :", np.mean(test_accuracies_mean))
x = list(range(class_per_task, num_class + 1, class_per_task))
plt.errorbar(x, test_accuracies_mean_DIS1, yerr=test_accuracies_std_DIS1, color='b', label='DIS1')
plt.errorbar(x, test_accuracies_mean_DIS1_2, yerr=test_accuracies_std_DIS1_2, color='g', label='DIS1_2')
plt.errorbar(x, test_accuracies_mean_DIS1_3, yerr=test_accuracies_std_DIS1_3, color='r', label='DIS1_3')
plt.legend()
plt.savefig('testDIS1_12_13_' + str(class_per_task) + '.png')