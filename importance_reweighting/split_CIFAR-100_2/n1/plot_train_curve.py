import pickle
import matplotlib.pyplot as plt
import sys
import numpy as np

class_per_task = int(sys.argv[1])
summaries_path_prefix = 'summaries_' + sys.argv[1] + '_'

num_perms = 5
i = int(sys.argv[2])

DB1_results_list = None
with open('DB1/' + summaries_path_prefix + str(i) + '/all_results', 'rb') as f:
	DB1_results_list =pickle.load(f)

with open('DIS/' + summaries_path_prefix + str(i) + '/all_results', 'rb') as f:
	DIS_results_list =pickle.load(f)

task = int(sys.argv[3])
for k, v in DB1_results_list[task].items():
	plt.plot(v['loss'], label='DB1')
	break

for k, v in DIS_results_list[task].items():
	plt.plot(v['loss'], label='DIS')
	break

plt.legend()
# plt.ylim(ymin=0, ymax=12)
plt.savefig('loss_curve.png')

plt.clf()

for k, v in DB1_results_list[task].items():
	plt.plot(v['val_acc'][-1], label='DB1')
	break

for k, v in DIS_results_list[task].items():
	plt.plot(v['val_acc'][-1], label='DIS')
	break

plt.legend()
plt.savefig('val_acc_curve.png')

