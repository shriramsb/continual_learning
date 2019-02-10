from src.trainer import MultiTaskTrainer
from src.networks import ResNet32

import importlib
import argparse
import os, shutil
import pickle

import tensorflow as tf

parser = argparse.ArgumentParser()
parser.add_argument('--config', required=True, help='path to file containing config params, with path in python module format')
parser.add_argument('--checkpoints_path', help='path to store checkpoints')
parser.add_argument('--summaries_path', help='path to store summaries (logs, validation accuracies, etc)')
parser.add_argument('--write_tensorboard', type=bool, help='whether to write losses and accuracies to tensorboardX')
parser.add_argument('--verbose', type=bool, help='whether to print statistics of training')
parser.add_argument('--eval_test_dataset', type=bool, help='whether to evaluate on test dataset at the end of training on each task')
parser.add_argument('--cuda_visible_devices', default='', help='comma separated gpu device ids to be used for training in tensorflow')
args = parser.parse_args()

hparams = importlib.import_module(args.config).hparams
if (args.checkpoints_path is not None):
    hparams['logging']['checkpoints_path'] = args.checkpoints_path
if (args.summaries_path is not None):
    hparams['logging']['summaries_path'] = args.summaries_path
if (args.write_tensorboard is not None):
    hparams['logging']['is_write_tensorboard'] = args.write_tensorboard
if (args.verbose is not None):
    hparams['logging']['verbose'] = args.verbose
if (args.eval_test_dataset is not None):
    hparams['logging']['eval_test_dataset'] = args.eval_test_dataset

for i in range(hparams['training']['start'], hparams['training']['end'] + 1):
    if (os.path.exists(os.path.join(hparams['logging']['summaries_path'], str(i)))):
        shutil.rmtree(os.path.join(hparams['logging']['summaries_path'], str(i)))
    if (os.path.exists(os.path.join(hparams['logging']['checkpoints_path'], str(i)))):
        shutil.rmtree(os.path.join(hparams['logging']['checkpoints_path'], str(i)))


os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_visible_devices
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

network = ResNet32()

trainer = MultiTaskTrainer(sess, network=network, input_shape=(32, 32, 3), output_shape=(100, ), hparams=hparams)
test_accuracy = trainer.trainTasksInRange()
with open(os.path.join(hparams['logging']['summaries_path'], 'test_accuracies.dat'), 'wb') as f:
    pickle.dump(test_accuracy, f)
