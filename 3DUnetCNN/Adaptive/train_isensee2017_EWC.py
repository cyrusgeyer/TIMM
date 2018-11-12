import os
import glob
import tables

from unet3d.data_adaptive import write_data_to_file, open_data_file
from unet3d.generator import get_training_and_validation_generators
from unet3d.model.isensee2017_3GPU import isensee2017_model_3GPU
from unet3d.model.isensee2017_2GPU_EWC import isensee2017_model_2GPU
from unet3d.training import load_old_model, train_model

import tensorflow as tf
from keras.backend import tensorflow_backend

config = dict()

####### data independent cofigurations:

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

    if config['data'] == 'brats':
        Directories = glob.glob(os.path.join("./brats", "data", "preprocessed", "*", "*"))
        truth = "truth"
        ending = ".nii.gz"
    elif config['data'] == 'mrbrains':
        Directories = glob.glob(os.path.join("./mrbrains", "MRBrainS13DataNii", "preprocessed", "*", "*"))
        truth = "LabelsForTraining"
        ending = ".nii.gz"
    elif config['data'] == 'lupus':
        Directories = glob.glob(os.path.join("./Lupus", "data", "original", "*", "*"))
        truth = "LabelsForTraining"
        ending = ".nii.gz"
    elif config['data'] == 'Neerav':
        Directories = glob.glob(os.path.join("./Neerav_data", "*"))
        truth = "preprocessed_gt15"
        ending = ".nii.gz"
    elif config['data'] == 'Brats_2018':
        Directories = glob.glob(os.path.join("./Brats2018", "MICCAI_BraTS_2018_Data_Training_preprocessed", "*", "*"))
        truth = "truth"
        ending = ".nii.gz"
    else:
        print('data must be Brats or MrBrains or Lupus')
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
    # convert input images into an hdf5 file

    config["data_file"] = config["data_file"][:-3] + '_' + config["normalize"] + config["data_file"][-3:]

    if overwrite or not os.path.exists(config["data_file"]):
        training_files, subject_ids = fetch_training_data_files(return_subject_ids=True)

        write_data_to_file(training_files,
                            config["data_file"],
                            image_shape=config["image_shape"],
                            subject_ids=subject_ids,
                            normalize=config['normalize'])

    data_file_opened = open_data_file(config["data_file"])

    os.environ["CUDA_VISIBLE_DEVICES"] = config["GPU"]

    sess = tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True)))

    tensorflow_backend.set_session(sess)

    if config["GPU_mode"] == 'auto_2':

        FM = tables.open_file(os.path.dirname(config['transfer_model_file'])+'/FM.h5').root
        M_old = tables.open_file(config['transfer_model_file']).root

        model = isensee2017_model_2GPU(input_shape=config["input_shape"],
                                  n_labels=config["n_labels"],
                                  initial_learning_rate=config["initial_learning_rate"],
                                  n_base_filters=config["n_base_filters"],
                                  non_trainable_list=config['non_trainable_list'],
                                  FM = FM, fisher_multiplier = config['fisher_multiplier'], M_old = M_old)

    if config["GPU_mode"] == 'auto_3':

        model = isensee2017_model_3GPU(input_shape=config["input_shape"],
                                  n_labels=config["n_labels"],
                                  initial_learning_rate=config["initial_learning_rate"],
                                  n_base_filters=config["n_base_filters"])

    if os.path.exists(config["transfer_model_file"]):

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
                n_epochs=config["n_epochs"],
                logging_file=config["logging_file"],
                tensorboard_logdir = config["Base_directory"]+'/logdir')
    data_file_opened.close()

if __name__ == "__main__":

    # If this script is run directly, please specify the following:
    config["GPU"] = '0,2,7'
    config["logging_file"] = 'LOG.csv'
    config["image_shape"] = (160,208,160)  # This determines what shape the images will be cropped/resampled to.
    config["flip"] = False  # augments the data by randomly flipping an axis during
    config["patch_shape"] = None  # switch to None to train on the whole image
    config["data_file"] = os.path.abspath("MRBrainS_data_cube.h5")
    config["all_modalities"] = ["t1", "t1Gd", "flair", "t2"]
    config["labels"] = (1, 2, 4)
    config["transfer_model_file"] = None
    config["model_file"] = os.path.abspath("isensee_MRBRainS_2017_model_3_GPU_cube.h5")
    config["training_file"] = os.path.abspath("isensee_MRBRainS_training_ids_span_norm_2_cube.pkl")
    config["validation_file"] = os.path.abspath("isensee_MRBRainS_validation_ids_span_norm_2_cube.pkl")
    config["overwrite"] = False  # If True, will previous files. If False, will use previously written files.
    config["n_labels"] = len(config["labels"])
    config["training_modalities"] = config[
        "all_modalities"]  # change this if you want to only use some of the modalities
    config["nb_channels"] = len(config["training_modalities"])

    if "patch_shape" in config and config["patch_shape"] is not None:
        config["input_shape"] = tuple([config["nb_channels"]] + list(config["patch_shape"]))
    else:
        config["input_shape"] = tuple([config["nb_channels"]] + list(config["image_shape"]))
    config["truth_channel"] = config["nb_channels"]

    main(overwrite=config["overwrite"])