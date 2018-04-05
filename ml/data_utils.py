import os
import imageio
import numpy as np
import h5py
import tarfile

image_size = 45 # size for this data doesn't change

def divide_data(dataset_X, dataset_Y):
    """
    Divide `dataset` into 3 datasets, train, test, and validation
    """
    n = dataset_X.shape[0]
    cut1 = round(n * 0.6)
    cut2 = round(n * 0.8)

    p = np.random.permutation(n)
    shuffle_X = dataset_X[p]
    shuffle_Y = dataset_Y[p]
    
    train_X = shuffle_X[:cut1]
    train_Y = shuffle_Y[:cut1]

    test_X = shuffle_X[cut1:cut2]
    test_Y = shuffle_Y[cut1:cut2]

    valid_X = shuffle_X[cut2:]
    valid_Y = shuffle_Y[cut2:]

    return {
        "train_X": train_X,
        "train_Y": train_Y,
        "test_X": test_X,
        "test_Y": test_Y,
        "valid_X": valid_X,
        "valid_Y": valid_Y
    }

def load_all_data():
    """
    Load each datapoint from the math symbols dataset and cache the formatted
    arrays in an h5 file. If the data is already cached, simply load from
    the h5 file.

    returns
        A dictionary containing
        - dataset_X: 3D numpy array of images
        - dataset_Y: 2D numpy array of integer labels corresponding to dataset_X.
                     The integer label corresponds to an index in `labels`
        - labels: Names of labels
    """
    full_path = os.path.join(os.path.dirname(__file__), "data.h5")
    if os.path.isfile(full_path):
        print("Loading dataset from file")
        # read h5 file
        h5f = h5py.File("data.h5", "r")
        return {
            "dataset_X": h5f["dataset_X"].value,
            "dataset_Y": h5f["dataset_Y"].value,
            "labels": h5f["labels"].value
        }
    else:
        print("Loading dataset and saving as data.h5")
        return package_dataset()

def load_dir(dirname, maxn=None):
    """
    Loads all the images from the directory for symbol `dirname` into
    a 3D numpy array

    returns
        A 3D numpy array containing all images for symbol `dirname`
    """
    os.path.split(__file__)
    full_path = os.path.join(os.path.dirname(__file__), "data", dirname)
    files = os.listdir(full_path)

    if maxn == None:
        ndata = len(files)
    else:
        ndata = min(maxn, len(files))

    dataset = np.ndarray(shape=(ndata, image_size ** 2),
                         dtype=np.float32)

    nimages = 0
    for i in range(ndata):
        impath = os.path.join(full_path, files[i])
        idata = imageio.imread(impath)

        if idata.shape != (image_size, image_size):
            raise Exception('Unexpected image shape: %s' % str(idata.shape))

        dataset[i, :] = idata.reshape((image_size ** 2, ))
        nimages += 1

    return dataset

def package_dataset():
    """
    Loads each symbol in the dataset and saves the whole set as numpy
    arrays in an h5 file. The arrays are as follows
        - dataset_X: 3D numpy array of images
        - dataset_Y: 2D numpy array of integer labels corresponding to dataset_X.
                     The integer label corresponds to an index in `labels`
        - labels: Names of labels
    """
    data_path = os.path.join(os.path.dirname(__file__), "data")
    dirnames = sorted(os.listdir(data_path))

    # Get total number of images
    nimg = 0
    for d in dirnames:
        imgs = os.listdir(os.path.join(data_path, d))
        nimg += len(imgs)

    print("Images in dataset:", nimg)
    dataset_X = np.zeros((nimg, image_size ** 2))
    dataset_Y = np.zeros((nimg, len(dirnames)))

    offset = 0
    maxn = 1000 # max images to use per letter
    for i in range(len(dirnames)):
        data = load_dir(dirnames[i], maxn)
        dataset_X[offset:offset+data.shape[0], :] = data
        dataset_Y[offset:offset+data.shape[0], i] = 1
        print("%s loaded" % dirnames[i])
        offset += data.shape[0]

    print("Images used:", offset)

    dataset_X = dataset_X[:offset]
    dataset_Y = dataset_Y[:offset]

    ret = {
        "dataset_X": dataset_X,
        "dataset_Y": dataset_Y,
        "labels": dirnames
    }

    h5f = h5py.File("data.h5", "w")
    h5f.create_dataset("dataset_X", data=dataset_X)
    h5f.create_dataset("dataset_Y", data=dataset_Y)
    h5f.create_dataset("labels", data=np.asarray(dirnames, dtype=np.dtype("S")))

    h5f.close()

    return ret

