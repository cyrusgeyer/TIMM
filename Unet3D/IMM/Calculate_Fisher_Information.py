import tables
import pickle
from ..unet3d.training import load_old_model
from keras.backend import tensorflow_backend
import tensorflow as tf
import numpy as np
from ..unet3d.metrics import weighted_dice_coefficient
import h5py

import gpustat
import time
import os
import argparse


def get_free_gpus(num, except_gpu = 2):
    gpu_list = gpustat.GPUStatCollection.new_query()
    free_gpu = []
    while not len(free_gpu) == num:
        for gpu in gpu_list:
            if not gpu.processes:
                if not gpu.index == except_gpu:
                    free_gpu.append(gpu.index)
            if len(free_gpu) == num:
                break
        if not len(free_gpu) == num:
            free_gpu = []
            print('Not enough GPUs avaialble at this time. Waiting ....')
            time.sleep(20)

    return str(free_gpu)[1:-1]


def CalculateFisherMatrix(FM_file, sess, data, graph, weight, logits_name, output_name, num_labels, L_max=1, **extras):
    inp = graph.get_tensor_by_name('input_1:0')
    output_before_sigmoid = graph.get_tensor_by_name(output_name)
    shape = list(output_before_sigmoid.shape)
    shape[0] = -1

    yl = tf.reshape(output_before_sigmoid, [-1, num_labels])

    sampled_labels = tf.reshape(tf.one_hot(tf.multinomial(yl, 1), num_labels), shape)
    logits = tf.reshape(graph.get_tensor_by_name(logits_name), shape)

    probability = weighted_dice_coefficient(sampled_labels, logits)

    FM_list = FM_file.require_dataset(weight.name, tuple([L_max] + weight.get_shape().as_list()), dtype='f')

    W_grad = tf.squeeze(tf.square(tf.gradients(tf.log(tf.clip_by_value(probability, 1e-10, 1.0)), weight)), axis=0)

    for L in range(0, L_max):
        FM = np.zeros(weight.get_shape().as_list())
        for i in range(data.shape[0]):
            W_grad_val = sess.run(W_grad, feed_dict={inp: np.expand_dims(data[i], 0)})
            FM += W_grad_val
        FM += 1e-10
        FM_list[L] = FM
        print('FM of ' + weight.name + ' sampled' + str(L) + ' times')
    return 0

def main(FLAGS,model,sess):
    tensorflow_backend.set_session(sess)

    data = tables.open_file(FLAGS.data_file)
    FM = os.path.dirname(FLAGS.model_file) + '/FM.h5'
    FM_file = h5py.File(FM, "a")
    p = pickle.load(open(FLAGS.validation_file, mode='r'))
    vali_data = data.root['data'][:][p]

    graph = sess.graph

    for weight in model.weights:
        CalculateFisherMatrix(FM_file, sess, vali_data, graph, weight, FLAGS.logits, FLAGS.output_before_sigmoid,
                              num_labels=len(FLAGS.labels))

    FM_file.close()

    sess.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--num_GPU',
        type=int,
        default=1,
        help='How many GPUs to use'
    )
    parser.add_argument(
        '--logits',
        type=str,
        default='activation_3/Sigmoid:0',
        help='name of the logits output tensor'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='add_7/add:0',
        help='Name of the tensor that feeds into the logits output (ouptut before sigmoid)'
    )
    parser.add_argument(
        '--data_file',
        type=str,
        default='/local/home/geyerr/3DUnetCNN/IMM_Experiments/Neerav/Neerav_data_hist_norm.h5',
        help='data_set that was used for training'
    )
    parser.add_argument(
        '--ids',
        type=str,
        default='/local/home/geyerr/3DUnetCNN/IMM_Experiments/Neerav/2_fold_split/Split_1/training_ids.pkl',
        help='Ids from the dataset on which to run the FM calculations'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='/local/home/geyerr/3DUnetCNN/Model_Neerav_1234567.h5',
        help='Model for which to run the FM calculations'
    )
    parser.add_argument(
        '--FM',
        type=str,
        default="None",
        help='String; The transfer model to start training with'
    )
    parser.add_argument(
        '--labels',
        type=lambda s: tuple([float(item) for item in s.split(',')]),
        default=(1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14.),
        help='Which labels to detect'
    )

    FLAGS, unparsed = parser.parse_known_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = get_free_gpus(FLAGS.num_GPU)
    print('The labels that the Fisher info is computed on:')
    print(FLAGS.labels)

    main(FLAGS)
