import argparse
import tensorflow as tf

from tuner import HyperparameterTuner

from tensorflow.contrib import tpu
from tensorflow.contrib.cluster_resolver import TPUClusterResolver


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--hidden_layers', type=int, default=2, help='the number of hidden layers')
    parser.add_argument('--hidden_units', type=int, default=800, help='the number of units per hidden layer')
    parser.add_argument('--num_perms', type=int, default=5, help='the number of tasks')
    parser.add_argument('--trials', type=int, default=50, help='the number of hyperparameter trials per task')
    parser.add_argument('--epochs', type=int, default=100, help='the number of training epochs per task')
    parser.add_argument('--use_tpu', type=bool, default=False, help='whether to use tpu')
    parser.add_argument('--tpu_name', type=str, default="", help='name of tpu in the project')
    return parser.parse_args()


def main():
    args = parse_args()
    
    if (not args.use_tpu):
        with tf.Session() as sess:
            tuner = HyperparameterTuner(sess=sess, hidden_layers=args.hidden_layers, hidden_units=args.hidden_units,
                                        num_perms=args.num_perms, trials=args.trials, epochs=args.epochs)
            tuner.search()
            print(tuner.best_parameters)
    else:
        # Get the TPU's location
        tpu_cluster = TPUClusterResolver(tpu=[args.tpu_name]).get_master()
        with tf.Session(tpu_cluster) as sess:
            sess.run(tpu.initialize_system())
            tuner = HyperparameterTuner(sess=sess, hidden_layers=args.hidden_layers, hidden_units=args.hidden_units,
                                        num_perms=args.num_perms, trials=args.trials, epochs=args.epochs)
            tuner.search()
            print(tuner.best_parameters)
            sess.run(tpu.shutdown_system())

if __name__ == "__main__":
    main()



