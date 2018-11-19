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

task = int(sys.argv[3])
for k, v in DB1_results_list[task].items():
	plt.plot(v['loss'], label='train_loss')
	inc = len(v['loss']) / len(v['val_loss'][-1])
	plt.plot(np.arange(inc , len(v['loss']) + inc , inc), v['val_loss'][-1], label='val_loss')
	break

plt.legend()
# plt.ylim(ymin=0, ymax=12)
plt.savefig('train_val_loss.png')

plt.clf()

for k, v in DB1_results_list[task].items():
	inc = len(v['acc']) / len(v['val_acc'][-1])
	plt.plot(np.arange(inc , len(v['acc']) + inc , inc), v['val_acc'][-1], label='val_acc')
	plt.plot(v['acc'], label='train_acc')
	break

plt.legend()
plt.savefig('train_val_acc.png')

