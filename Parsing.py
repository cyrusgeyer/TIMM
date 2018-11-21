import tensorflow as tf

from Unet3D.Adaptive import train_isensee2017_EWC as train_isensee2017
import gpustat
import time
import os
import argparse


def get_free_gpus(num, except_gpu=None):
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

    return free_gpu


def main():
    FLAGS.GPU = get_free_gpus(FLAGS.num_GPU)
    train_isensee2017.config.update(vars(FLAGS))
    model = train_isensee2017.main(overwrite=False)
    train_isensee2017.model


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--num_GPU',
        type=int,
        default=2,
        help='Integer; defines how many GPUs to use'
    )
    parser.add_argument(
        '--Base_directory',
        type=str,
        default='Brain_Region_Segmentation',
        help='String; New Folder where The Model will be stored.'
    )
    parser.add_argument(
        '--logging_file',
        type=str,
        default='None',
        help='String; names the file that stores the training progress'
    )
    parser.add_argument(
        '--image_shape',
        type=lambda s: tuple([int(item) for item in s.split(',')]),
        default=(160, 208, 160),
        help='Tuple; The dimensions of the input image, e.g. : 160,208,160'
    )
    parser.add_argument(
        '--normalize',
        type=str,
        default='hist_norm',
        help='options: mean_and_std, No, zero_one, hist_norm'
    )
    parser.add_argument(
        '--deconvolution',
        type=bool,
        default=True,
        help='Bool; whether to use convolution or upsampling'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=1,
        help='Int; batch size'
    )
    parser.add_argument(
        '--transfer_model_file',
        type=str,
        default=None,
        help='String; The transfer model to start training with'
    )
    parser.add_argument(
        '--model_file',
        type = str,
        default='None',
        help='String; The name for the new model'
    )
    parser.add_argument(
        '--data_file',
        type=str,
        default='Data',
        help='String; The name for the data_file'
    )
    parser.add_argument(
        '--data_directory',
        type=str,
        default='None',
        help='String; The name for the data_file'
    )
    parser.add_argument(
        '--training_file',
        type=str,
        default='None',
        help='String; The name for the new model'
    )
    parser.add_argument(
        '--validation_file',
        type=str,
        default='None',
        help='String; The name for the new model'
    )
    parser.add_argument(
        '--flip',
        type=bool,
        default=False,
        help='whether to randomly flip axis during training'
    )
    parser.add_argument(
        '--distort',
        type=bool,
        default=False,
        help='whether to randomly flip axis during training'
    )
    parser.add_argument(
        '--return_name_only',
        type=bool,
        default=False,
        help='Return name of the Model only'
    )
    parser.add_argument(
        '--Except_layers',
        type=lambda s: tuple([str(item) for item in s.split(',')]),
        default=('conv3d_30', 'conv3d_22', 'conv3d_26'),
        help='Return name of the Model only'
    )
    parser.add_argument(
        '--ID',
        type=str,
        default='exp_2',
        help='In case the same config is used twice'
    )
    parser.add_argument(
        '--train_decoder_only',
        type=bool,
        default=False,
        help='Whether to train the decoder only'
    )
    parser.add_argument(
        '--Load_optimizer',
        type=bool,
        default=False,
        help='Whether to load the optimizer weights too'
    )
    parser.add_argument(
        '--fisher_multiplier',
        type=float,
        default=1000,
        help='How to scale the Fisher Info'
    )
    parser.add_argument(
        '--labels',
        type= str,
        default='1,2,3,4,5,6,7,8,9,10,11,12,13,14',
        help='Which labels to detect'
    )
    parser.add_argument(
        '--training_modalities',
        type=lambda s: tuple([str(item) for item in s.split(',')]),
        default=('t1', 't1ce', 'flair', 't2'),
        help='names of the training modality files. Also sets number of input channels.'
    )
    parser.add_argument(
        '--EWC',
        type=bool,
        default=False,
        help='Whether to use EWC to regularize parameters.'
    )

    FLAGS, unparsed = parser.parse_known_args()

    if not FLAGS.data_file:
        print('Please specify the argument data_file')

    FLAGS.Base_directory = os.path.join(os.getcwd(),'Data_and_Pretrained_Models',FLAGS.Base_directory)
    FLAGS.data_file = os.path.join(FLAGS.Base_directory,FLAGS.data_file)
    FLAGS.nb_channels = len(FLAGS.training_modalities)
    FLAGS.patch_shape = None
    FLAGS.truth_channel = FLAGS.nb_channels
    FLAGS.input_shape = tuple([FLAGS.nb_channels] + list(FLAGS.image_shape))
    FLAGS.augment = FLAGS.flip or FLAGS.distort
    if FLAGS.train_decoder_only:
        FLAGS.non_trainable_list = ['conv3d_' + str(s) for s in range(1, 15)]
    else:
        FLAGS.non_trainable_list = None

    f1 = lambda s: ''.join([(item) for item in s.split(',')])
    f2 = lambda s: tuple([float(item) for item in s.split(',')])

    new_exp_name = os.path.join(FLAGS.Base_directory,'Trained_models')

    if FLAGS.model_file == 'None':
        new_exp_name = os.path.join(new_exp_name, 'Labels_' + f1(FLAGS.labels))
        if not FLAGS.transfer_model_file:
            new_exp_name = os.path.join(new_exp_name, 'No_Transfer')
        else:
            tr_model = os.path.relpath(FLAGS.transfer_model_file, os.getcwd())
            new_exp_name = os.path.join(new_exp_name,
                                        'Transfer',
                                        tr_model.translate(None, '.').translate(None, '/') + '_as_transfer_model')
        if FLAGS.ID != 'None':
            new_exp_name += FLAGS.ID
            print('Experiment with ' + FLAGS.ID + ' is run')

        if (not os.path.isdir(new_exp_name)) and (not FLAGS.return_name_only):
            os.makedirs(new_exp_name)
        elif FLAGS.return_name_only:
            pass
        else:
            print('The Experiment and the specified ID already exist.')
            raise EnvironmentError

        FLAGS.model_file = os.path.join(new_exp_name, 'Model.h5')
        FLAGS.logging_file = os.path.join(new_exp_name, 'Log.csv')

    if FLAGS.training_file == 'None':
        FLAGS.training_file = os.path.join(FLAGS.Base_directory, 'training_ids.pkl')
    if FLAGS.validation_file == 'None':
        FLAGS.validation_file = os.path.join(FLAGS.Base_directory, 'validation_ids.pkl')

    FLAGS.labels = f2(FLAGS.labels)
    FLAGS.n_labels = len(FLAGS.labels)

    if not FLAGS.return_name_only:
        main()
    else:
        print(FLAGS.model_file)