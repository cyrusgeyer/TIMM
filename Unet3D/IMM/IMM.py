import tensorflow as tf
import numpy as np
import h5py
import tables
import argparse

def MODE_TRANSFER(Model_directories, FM_directories, New_weight_directory, alpha, model_basis):

    FM = [h5py.File(d, 'r') for d in FM_directories]

    if not alpha:
        alpha = [1.0/len(FM)]*len(FM)

    Gesamt_W = {}
    for key1 in list(FM[0].keys()):
        if all(key1 in fm for fm in FM):
            for key2 in list(FM[0][key1].keys()):
                shape = FM[0][key1][key2][:].shape
                if all(fm[key1][key2][:].shape == shape for fm in FM):
                    Gesamt_W[key1 + '/' + key2] = alpha[0] * (np.mean(FM[0][key1][key2][:], 0)/np.sum(FM[0][key1][key2][:]))
                    for i in range(1, len(FM)):
                        Gesamt_W[key1 + '/' + key2] += (alpha[i] * (np.mean(FM[i][key1][key2][:], 0)/np.sum(FM[i][key1][key2][:])))

    W_Faktor = [{} for _ in range(len(FM))]

    for i in range(len(FM)):
        for key1 in list(FM[0].keys()):
            if all(key1 in fm for fm in FM):
                for key2 in list(FM[0][key1].keys()):
                    shape = FM[0][key1][key2][:].shape
                    if all(fm[key1][key2][:].shape == shape for fm in FM):
                        W_Faktor[i][key1 + '/' + key2] = alpha[i] * (np.mean(FM[i][key1][key2][:], 0)/np.sum(FM[i][key1][key2][:])) / Gesamt_W[
                            key1 + '/' + key2]

    opened_files = [tables.open_file(d) for d in Model_directories]

    Dicts = [f.root['model_weights'] for f in opened_files]
    opened_files[model_basis].copy_file(New_weight_directory, overwrite=True)
    opened_file_new_weights = tables.open_file(New_weight_directory,mode='a')
    Neues_W = opened_file_new_weights.root['model_weights']

    for key1 in list(FM[0].keys()):
        if all(key1 in fm for fm in FM):
            for key2 in list(FM[0][key1].keys()):
                shape = FM[0][key1][key2][:].shape
                if all(fm[key1][key2][:].shape == shape for fm in FM):
                    Neues_W[key1][key1][key2][:] = Dicts[0][key1][key1][key2][:] * W_Faktor[0][key1 + '/' + key2]
                    for i in range(1, len(FM)):
                        Neues_W[key1][key1][key2][:] += Dicts[i][key1][key1][key2][:] * W_Faktor[i][key1 + '/' + key2]
                else:
                    print(key1 + '/' + key1 + '/' + key2 + ' Was not mixed according to IMM as the dimensions do not'
                                                           'match. Careful when using these weights!')

    [f.close() for f in opened_files]
    opened_file_new_weights.close()

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--FM_directories',
        type=lambda s: tuple([str(item) for item in s.split(',')]),
        default=('/local/home/geyerr/3DUnetCNN/IMM_Experiments/Neerav/2_fold_split/Split_1/Labels_135791113/No_Transfer/FM.h5', '/local/home/geyerr/3DUnetCNN/IMM_Experiments/Neerav/2_fold_split/Split_2/Labels_2468101214/Transfer/Split_2Labels_2468101214TransferSplit_1Labels_135791113No_Transfertrain_decoder/FM.h5'),
        help='The directories where the Fisher information is stored'
    )
    parser.add_argument(
        '--Model_directories',
        type=lambda s: tuple([str(item) for item in s.split(',')]),
        default=('/local/home/geyerr/3DUnetCNN/IMM_Experiments/Neerav/2_fold_split/Split_1/Labels_135791113/No_Transfer/Model.h5','/local/home/geyerr/3DUnetCNN/IMM_Experiments/Neerav/2_fold_split/Split_2/Labels_2468101214/Transfer/Split_2Labels_2468101214TransferSplit_1Labels_135791113No_Transfertrain_decoder/Model.h5'),
        help='The directories where the trained models are stored'
    )
    parser.add_argument(
        '--New_weight_directory',
        type=str,
        default='/local/home/geyerr/3DUnetCNN/IMM_Experiments/Neerav/2_fold_split/IMM_model/IMM_model_135791113.h5',
        help='Directory to store the new '
    )
    parser.add_argument(
        '--Alphas',
        type=lambda s: tuple([float(item) for item in s.split(',')]),
        default=None,
        help='The directories where the trained models are stored'
    )
    parser.add_argument(
        '--Model_basis',
        type=int,
        default=0,
        help='Which of the models to use as a basis'
    )

    FLAGS, unparsed = parser.parse_known_args()

    MODE_TRANSFER(FLAGS.Model_directories,
                  FLAGS.FM_directories,
                  FLAGS.New_weight_directory,
                  FLAGS.Alphas,
                  FLAGS.Model_basis)
