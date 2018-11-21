import os

import numpy as np
import tables

from .normalize import normalize_data_storage, reslice_image_set, get_cropping_parameters, reslice_image_set_mrbrains


def create_data_file(out_file, n_channels, n_samples, image_shape):
    hdf5_file = tables.open_file(out_file, mode='w')
    filters = tables.Filters(complevel=5, complib='blosc')
    data_shape = tuple([0, n_channels] + list(image_shape))
    truth_shape = tuple([0, 1] + list(image_shape))
    data_storage = hdf5_file.create_earray(hdf5_file.root, 'data', tables.Float32Atom(), shape=data_shape,
                                           filters=filters, expectedrows=n_samples)
    truth_storage = hdf5_file.create_earray(hdf5_file.root, 'truth', tables.UInt8Atom(), shape=truth_shape,
                                            filters=filters, expectedrows=n_samples)
    affine_storage = hdf5_file.create_earray(hdf5_file.root, 'affine', tables.Float32Atom(), shape=(0, 4, 4),
                                             filters=filters, expectedrows=n_samples)
    return hdf5_file, data_storage, truth_storage, affine_storage

def write_image_data_to_file(image_files, data_storage, truth_storage, padding, crop_slice,n_channels, affine_storage,
                             truth_dtype=np.uint8, normalize = 'No', image_shape = (160,208,160)):


    if all(p > 0 for p in traverse(padding)):
        # No negative values in padding means the images fit inside the defined shape
        image_shape = None

    for set_of_files in image_files:
        images = reslice_image_set_mrbrains(set_of_files,
                                            label_indices=len(set_of_files) - 1,
                                            crop_slices=crop_slice,
                                            image_shape=image_shape)

        if normalize == 'hist_norm' or normalize == 'zero_one':
            subject_data = []
            for j in range(len(images)):

                if all(p > 0 for p in traverse(padding)):
                    # Pad the image with zeros.
                    img = np.pad(np.asarray(images[j].get_data()),
                                 pad_width=padding,
                                 mode='constant',
                                 constant_values=0.0)
                else:
                    img = np.asarray(images[j].get_data())

                if j < len(images)-1:
                    # The data
                    img[img < 0] = 0
                    img = img.astype(np.float)
                    img /= np.max(img)
                    if normalize == 'hist_norm':
                        subject_data.append(hist_match(img))
                    elif normalize == 'zero_one':
                        subject_data.append(img)
                else:
                    # The truth
                    subject_data.append(img)
        else:
            subject_data = [
                np.pad(np.asarray(image.get_data()), pad_width=padding, mode='constant', constant_values=0.0)
                if all(p > 0 for p in traverse(padding))
                else np.asarray(image.get_data())
                for image in images]
        add_data_to_storage(data_storage, truth_storage, affine_storage, subject_data, np.eye(4), n_channels,truth_dtype)
    return data_storage, truth_storage


def add_data_to_storage(data_storage, truth_storage, affine_storage, subject_data, affine, n_channels, truth_dtype):
    data_storage.append(np.asarray(subject_data[:n_channels])[np.newaxis])
    truth_storage.append(np.asarray(subject_data[n_channels], dtype=truth_dtype)[np.newaxis][np.newaxis])
    affine_storage.append(np.asarray(affine)[np.newaxis])


def get_crop_slice_and_image_shape(training_data_files):
    starts_list = []
    stops_list = []
    diff_list = []
    for set_of_files in training_data_files:
        crop = get_cropping_parameters([set_of_files])
        starts_list.append([sl.start for sl in crop])
        stops_list.append([sl.stop for sl in crop])
    start = np.min(np.asarray(starts_list), 0)
    stop = np.max(np.asarray(stops_list), 0)

    crop_slice = [slice(st, sp, None) for st, sp in zip(start, stop)]

    image_shape = list(stop - start)

    return crop_slice,image_shape

def write_data_to_file(training_data_files, out_file, truth_dtype=np.uint8, subject_ids=None,
                       normalize= 'No', image_shape = (208,208,208)):
    """
    Takes in a set of training images and writes those images to an hdf5 file.
    :param training_data_files: List of tuples containing the training data files. The modalities should be listed in
    the same order in each tuple. The last item in each tuple must be the labeled image.
    Example: [('sub1-T1.nii.gz', 'sub1-T2.nii.gz', 'sub1-truth.nii.gz'),
              ('sub2-T1.nii.gz', 'sub2-T2.nii.gz', 'sub2-truth.nii.gz')]
    :param out_file: Where the hdf5 file will be written to.
    :param image_shape: Shape of the images that will be saved to the hdf5 file.
    :param truth_dtype: Default is 8-bit unsigned integer.
    :return: Location of the hdf5 file with the image data written to it.
    """
    n_samples = len(training_data_files)
    n_channels = len(training_data_files[0]) - 1

    crop_slice, min_dim = get_crop_slice_and_image_shape(training_data_files)

    pad = padding(crop_slice,image_shape)

    try:
        hdf5_file, data_storage, truth_storage, affine_storage = create_data_file(out_file,
                                                                                  n_channels=n_channels,
                                                                                  n_samples=n_samples,
                                                                                  image_shape=image_shape)

    except Exception as e:
        # If something goes wrong, delete the incomplete data file
        os.remove(out_file)
        raise e

    write_image_data_to_file(training_data_files,
                              data_storage,
                              truth_storage,
                              pad,
                              crop_slice,
                              truth_dtype=truth_dtype,
                              n_channels=n_channels,
                              affine_storage=affine_storage,
                              normalize=normalize)
    if subject_ids:
        hdf5_file.create_array(hdf5_file.root,
                               'subject_ids',
                               obj=subject_ids)
    if normalize == 'mean_and_std':
        normalize_data_storage(data_storage)
    hdf5_file.close()
    return out_file


def open_data_file(filename, readwrite="r"):
    return tables.open_file(filename, readwrite)


def padding(crop_slices,desired_shape):

    frame = [s.stop-s.start for s in crop_slices]
    div = [int(s - f) / 2 for s, f in zip(desired_shape, frame)]
    rem = [int(s - f) % 2 for s, f in zip(desired_shape, frame)]

    return [(a,a+b) for a,b in zip(div,rem)]

def hist_match(source, template = None):
    """
    Adjust the pixel values of a grayscale image such that its histogram
    matches that of a target image

    Arguments:
    -----------
        source: np.ndarray
            Image to transform; the histogram is computed over the flattened
            array
        template: np.ndarray
            Template image; can have different dimensions to source
    Returns:
    -----------
        matched: np.ndarray
            The transformed output image
    """

    oldshape = source.shape
    source = source.ravel()


    # get the set of unique pixel values and their corresponding indices and
    # counts
    s_values, bin_idx, s_counts = np.unique(source, return_inverse=True,
                                            return_counts=True)

    if template:
        template = template.ravel()
        t_values, t_counts = np.unique(template, return_counts=True)
        t_values = t_values[1:]
        t_quantiles = np.cumsum(t_counts[1:]).astype(np.float64)
        t_quantiles /= t_quantiles[-1]
    else:
        t_quantiles = np.linspace(0,1,len(source))[1:]
        t_values = np.linspace(0,1,len(source))[1:]

    # take the cumsum of the counts and normalize by the number of pixels to
    # get the empirical cumulative distribution functions for the source and
    # template images (maps pixel value --> quantile)
    s_quantiles = np.cumsum(s_counts[1:]).astype(np.float64)
    s_quantiles /= s_quantiles[-1]


    # interpolate linearly to find the pixel values in the template image
    # that correspond most closely to the quantiles in the source image
    interp_t_values = np.interp(s_quantiles, t_quantiles, t_values)

    return np.hstack((np.asarray([0]),interp_t_values))[bin_idx].reshape(oldshape)

def traverse(item):
    try:
        for i in iter(item):
            for j in traverse(i):
                yield j
    except TypeError:
        yield item