import os
import glob
import tables

from Unet3D.unet3d.data_adaptive import write_data_to_file, open_data_file
from Unet3D.unet3d.generator import get_training_and_validation_generators
from Unet3D.unet3d.model.isensee2017_GPU_EWC import isensee2017_model
from Unet3D.unet3d.training import load_old_model, train_model

import tensorflow as tf
from keras.backend import tensorflow_backend

config = dict()

config["n_base_filters"] = 16
config["deconvolution"] = True  # if False, will use upsampling instead of deconvolution
config["batch_size"] = 1
config["validation_batch_size"] = 1
config["n_epochs"] = 500  # cutoff the training after this many epochs
config["patience"] = 10  # learning rate will be reduced after this many epochs if the validation loss is not improving
config["early_stop"] = 50  # training will be stopped after this many epochs without the validation loss improving
config["initial_learning_rate"] = 5e-4
config["learning_rate_drop"] = 0.5  # factor by which the learning rate will be reduced
config["validation_split"] = 0.8  # portion of the data that will be used for training
config["permute"] = False  # data shape must be a cube. Augments the data by permuting in various directions
config["validation_patch_overlap"] = 0  # if > 0, during training, validation patches will be overlapping
config["training_patch_start_offset"] = (16, 16, 16)  # randomly offset the first patch index by up to this offset
config["skip_blank"] = True  # if True, then patches without any target will be skipped


def fetch_training_data_files(return_subject_ids=False):
    training_data_files = list()
    subject_ids = list()

    if os.path.isdir(config['data_directory']):
        if os.path.isfile(glob.glob(os.path.join(config['data_directory'], "*","*"))[0]):
            Directories = glob.glob(os.path.join(config['data_directory'], "*"))
            truth = "truth"
            ending = ".nii.gz"
        elif os.path.isfile(glob.glob(os.path.join(config['data_directory'], "*","*","*"))[0]):
            Directories = glob.glob(os.path.join(config['data'], "*","*"))
            truth = "truth"
            ending = ".nii.gz"
        else:
            print(
                'Please provide a file structure that obeys the following rules: \n Data_directory \n --> Patient_directory'
                ' \n      --> Patient_file'
                ' \n OR'
                ' \n Data_directory'
                ' \n --> Subdirectory'
                ' \n     --> Patient_directory'
                ' \n         --> Patient_file'
                ' \nPatient File must be modality name provided in config + .nii.gz and truth.nii.gz for ground truth images.'
                )
            raise EnvironmentError
    else:
        print(
            config['data_directory'] + " does not exist"
            '\n Please provide a file structure that obeys the following rules: \n Data_directory \n --> Patient_directory'
                ' \n      --> Patient_file'
                ' \n OR'
                ' \n Data_directory'
                ' \n --> Subdirectory'
                ' \n     --> Patient_directory'
                ' \n         --> Patient_file'
                ' \nPatient File must be modality name provided in config + .nii.gz and truth.nii.gz for ground truth images.'
        )
        raise EnvironmentError
    for subject_dir in Directories:
        subject_ids.append(os.path.basename(subject_dir))
        subject_files = list()
        for modality in config["training_modalities"] + [truth]:
            subject_files.append(os.path.join(subject_dir, modality + ending))
        training_data_files.append(tuple(subject_files))
    if return_subject_ids:
        return training_data_files, subject_ids
    else:
        return training_data_files



def main(overwrite=False):

    if overwrite or not os.path.exists(config["data_file"]):
        print('specified data_file does not exist yet at' + config["data_file"] + '. Trying to build a data_file from '
              'patient data at '+config['data_directory'])
        training_files, subject_ids = fetch_training_data_files(return_subject_ids=True)

        write_data_to_file(training_files,
                           config["data_file"],
                           image_shape=config["image_shape"],
                           subject_ids=subject_ids,
                           normalize=config['normalize'])

    data_file_opened = open_data_file(config["data_file"])

    os.environ["CUDA_VISIBLE_DEVICES"] = str(config["GPU"])[1:-1]

    sess = tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True)))

    tensorflow_backend.set_session(sess)

    if config["EWC"]:
        try:
            FM = tables.open_file(os.path.dirname(config['transfer_model_file']) + '/FM.h5').root
        except:
            'There appears to be no Fisher Information at: ' + os.path.dirname(
                config['transfer_model_file']) + '/FM.h5' + ' It is therefor not possible to run EWC.'
            raise AttributeError

        M_old = tables.open_file(config['transfer_model_file']).root

        model = isensee2017_model(input_shape=config["input_shape"],
                                  n_labels=config["n_labels"],
                                  initial_learning_rate=config["initial_learning_rate"],
                                  n_base_filters=config["n_base_filters"],
                                  non_trainable_list=config['non_trainable_list'],
                                  gpu=len(config["GPU"]),
                                  FM=FM,
                                  fisher_multiplier=config['fisher_multiplier'],
                                  M_old=M_old)

    else:

        model = isensee2017_model(input_shape=config["input_shape"],
                                  n_labels=config["n_labels"],
                                  initial_learning_rate=config["initial_learning_rate"],
                                  n_base_filters=config["n_base_filters"],
                                  non_trainable_list=config['non_trainable_list'],
                                  gpu=len(config["GPU"]))

    if config["transfer_model_file"]:

        transfer_model = load_old_model(config["transfer_model_file"])

        for layer, t_layer in zip(model.layers, transfer_model.layers):
            if len(layer.weights) > 0:
                if layer.name not in config["Except_layers"]:
                    try:
                        model.get_layer(layer.name).set_weights(transfer_model.get_layer(layer.name).get_weights())
                    except:
                        print(layer.name + ' was not set as the dimensions do not match')
                else:
                    print(layer.name + ' was not set as it was excluded by Except_layers')

        # if config['Load_optimizer']:
        #     model.optimizer.set_weights(transfer_model.optimizer.get_weights())

    if not overwrite and os.path.exists(config["model_file"]):
        model.load_weights(config["model_file"])
        print('loading_old_weights')

    # get training and testing generators
    train_generator, validation_generator, n_train_steps, n_validation_steps = get_training_and_validation_generators(
        data_file_opened,
        batch_size=config["batch_size"],
        data_split=config["validation_split"],
        overwrite=overwrite,
        validation_keys_file=config["validation_file"],
        training_keys_file=config["training_file"],
        n_labels=config["n_labels"],
        labels=config["labels"],
        patch_shape=config["patch_shape"],
        validation_batch_size=config["validation_batch_size"],
        validation_patch_overlap=config["validation_patch_overlap"],
        training_patch_start_offset=config["training_patch_start_offset"],
        permute=config["permute"],
        augment=config["augment"],
        skip_blank=config["skip_blank"],
        augment_flip=config["flip"],
        augment_distortion_factor=config["distort"])

    # run training
    train_model(model=model,
                model_file=config["model_file"],
                training_generator=train_generator,
                validation_generator=validation_generator,
                steps_per_epoch=n_train_steps,
                validation_steps=n_validation_steps,
                initial_learning_rate=config["initial_learning_rate"],
                learning_rate_drop=config["learning_rate_drop"],
                learning_rate_patience=config["patience"],
                early_stopping_patience=config["early_stop"],
                n_epochs=1,
                logging_file=config["logging_file"],
                tensorboard_logdir=os.path.join(os.path.dirname(config["model_file"]), 'logdir'))
    data_file_opened.close()

    return model,sess
