from functools import partial

from keras.layers import Input, LeakyReLU, Add, UpSampling3D, Activation, SpatialDropout3D, Conv3D
from keras.engine import Model
from keras.optimizers import Adam

from .unet import concatenate, EWC, create_convolution_block
from ..metrics import weighted_dice_coefficient_loss
import tensorflow as tf

create_convolution_block = partial(create_convolution_block, activation=LeakyReLU, instance_normalization=True)

def isensee2017_model(input_shape=(4, 128, 128, 128), n_base_filters=16, depth=5,
                           dropout_rate=0.3, n_segmentation_levels=3, n_labels=4, optimizer=Adam,
                           initial_learning_rate=5e-4, loss_function=weighted_dice_coefficient_loss,
                           activation_name="sigmoid", non_trainable_list=None, ewc=False, gpu=2,
                           FM = None, M_old = None, fisher_multiplier=None):
    '''
    This function builds a model proposed by Isensee et al. for the BRATS 2017 competition:
    https://www.cbica.upenn.edu/sbia/Spyridon.Bakas/MICCAI_BraTS/MICCAI_BraTS_2017_proceedings_shortPapers.pdf

    It allows for dictribution on multiple GPU and enables the use of Elastic weight consolidation.

    This network is highly similar to the model proposed by Kayalibay et al. "CNN-based Segmentation of Medical
    Imaging Data", 2017: https://arxiv.org/pdf/1701.03056.pdf

    :param FM:
    :param M_old:
    :param fisher_multiplier:
    :param input_shape:
    :param n_base_filters:
    :param depth:
    :param dropout_rate:
    :param n_segmentation_levels:
    :param n_labels:
    :param optimizer:
    :param initial_learning_rate:
    :param loss_function:
    :param activation_name:
    :param non_trainable_list:
    :param ewc:
    :param gpu:
    :return:
    '''
    if ewc:
        ewc = EWC(M_old=M_old,
                  FM=FM,
                  fisher_multiplier=fisher_multiplier,
                  activation=LeakyReLU,
                  instance_normalization=True)

    inputs = Input(input_shape)

    current_layer = inputs
    level_output_layers = list()
    level_filters = list()

    encoder_gpu = "/device:GPU:0"
    decoder_gpu = "/device:GPU:0" if gpu == 1 else "/device:GPU:1"

    index = 1

    with tf.device(encoder_gpu):

        for level_number in range(depth):
            n_level_filters = (2 ** level_number) * n_base_filters
            level_filters.append(n_level_filters)

            if current_layer is inputs:
                if ewc:
                    in_conv = ewc.create_regularized_convolution_block(current_layer, n_level_filters, index)
                    index += 1
                else:
                    in_conv = create_convolution_block(current_layer, n_level_filters)

            else:
                if ewc:
                    in_conv = ewc.create_regularized_convolution_block(current_layer, n_level_filters, index,
                                                                       strides=(2, 2, 2))
                    index += 1
                else:
                    in_conv = create_convolution_block(current_layer, n_level_filters, strides=(2, 2, 2))

            if ewc:
                context_output_layer = create_regularized_context_module(in_conv, n_level_filters, ewc, index,
                                                                         dropout_rate=dropout_rate)
                index += 2
            else:
                context_output_layer = create_context_module(in_conv, n_level_filters, dropout_rate=dropout_rate)

            summation_layer = Add()([in_conv, context_output_layer])
            level_output_layers.append(summation_layer)
            current_layer = summation_layer

    segmentation_layers = list()

    with tf.device(decoder_gpu):

        for level_number in range(depth - 2, -1, -1):
            if ewc:
                up_sampling = create_regularized_up_sampling_module(current_layer, level_filters[level_number], ewc,
                                                                    index)
                index += 1
                concatenation_layer = concatenate([level_output_layers[level_number], up_sampling], axis=1)
                localization_output = create_regularized_localization_module(concatenation_layer,
                                                                             level_filters[level_number], ewc,
                                                                             index)
                index += 2
            else:
                up_sampling = create_up_sampling_module(current_layer, level_filters[level_number])
                concatenation_layer = concatenate([level_output_layers[level_number], up_sampling], axis=1)
                localization_output = create_localization_module(concatenation_layer, level_filters[level_number])
            current_layer = localization_output
            if level_number < n_segmentation_levels:
                segmentation_layers.insert(0, Conv3D(n_labels, (1, 1, 1))(current_layer))
                index += 1

        output_layer = None
        for level_number in reversed(range(n_segmentation_levels)):
            segmentation_layer = segmentation_layers[level_number]
            if output_layer is None:
                output_layer = segmentation_layer
            else:
                output_layer = Add()([output_layer, segmentation_layer])

            if level_number > 0:
                output_layer = UpSampling3D(size=(2, 2, 2))(output_layer)

        activation_block = Activation(activation_name)(output_layer)

    model = Model(inputs=inputs, outputs=activation_block)

    if non_trainable_list:

        for layer in model.layers:
            if layer.name in non_trainable_list:
                layer.trainable = False
                print(str(layer.name) + ' is frozen')

    model.compile(optimizer=optimizer(lr=initial_learning_rate),
                  loss=loss_function,
                  metrics=[weighted_dice_coefficient_loss])

    return model


def create_regularized_localization_module(input_layer, n_filters, ewc, index):
    convolution1 = ewc.create_regularized_convolution_block(input_layer, n_filters, index=index)
    index += 1
    convolution2 = ewc.create_regularized_convolution_block(convolution1, n_filters, kernel=(1, 1, 1), index=index)
    return convolution2


def create_regularized_up_sampling_module(input_layer, n_filters, ewc, index, size=(2, 2, 2)):
    up_sample = UpSampling3D(size=size)(input_layer)
    convolution = ewc.create_regularized_convolution_block(up_sample, n_filters, index=index)
    return convolution


def create_regularized_context_module(input_layer, n_level_filters, ewc, index, dropout_rate=0.3,
                                      data_format="channels_first"):
    convolution1 = ewc.create_regularized_convolution_block(input_layer=input_layer, n_filters=n_level_filters,
                                                            index=index)
    index += 1
    dropout = SpatialDropout3D(rate=dropout_rate, data_format=data_format)(convolution1)
    convolution2 = ewc.create_regularized_convolution_block(input_layer=dropout, n_filters=n_level_filters, index=index)
    return convolution2


def create_localization_module(input_layer, n_filters):
    convolution1 = create_convolution_block(input_layer, n_filters)
    convolution2 = create_convolution_block(convolution1, n_filters, kernel=(1, 1, 1))
    return convolution2


def create_up_sampling_module(input_layer, n_filters, size=(2, 2, 2)):
    up_sample = UpSampling3D(size=size)(input_layer)
    convolution = create_convolution_block(up_sample, n_filters)
    return convolution


def create_context_module(input_layer, n_level_filters, dropout_rate=0.3, data_format="channels_first"):
    convolution1 = create_convolution_block(input_layer=input_layer, n_filters=n_level_filters)
    dropout = SpatialDropout3D(rate=dropout_rate, data_format=data_format)(convolution1)
    convolution2 = create_convolution_block(input_layer=dropout, n_filters=n_level_filters)
    return convolution2
