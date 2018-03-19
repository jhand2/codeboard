import os
import imageio
import numpy as np

def load_dir(dirname, image_size, maxn=None):
    os.path.split(__file__)
    full_path = os.path.join(os.path.dirname(__file__), "data", dirname)
    files = os.listdir(full_path)

    if maxn == None:
        ndata = len(files)
    else:
        ndata = min(maxn, len(files))

    dataset = np.ndarray(shape=(ndata, image_size, image_size),
                         dtype=np.float32)

    nimages = 0
    for i in range(len(files)):
        if i >= ndata:
            break

        impath = os.path.join(full_path, files[i])
        idata = imageio.imread(impath)

        if idata.shape != (image_size, image_size):
            raise Exception('Unexpected image shape: %s' % str(idata.shape))

        dataset[i, :, :] = idata
        nimages += 1

    dataset = dataset[0:nimages, :, :] # only return as many images as we loaded

    return dataset

def load_all_data(image_size):
    data_path = os.path.join(os.path.dirname(__file__), "data")
    dirnames = os.listdir(data_path)

    # Get total number of images
    nimg = 0
    for d in dirnames:
        imgs = os.listdir(os.path.join(data_path, d))
        nimg += len(imgs)

    print("Images in dataset:", nimg)
    train_X = np.zeros((nimg, image_size, image_size))
    train_Y = np.zeros((nimg, 1))

    # TODO: Split dataset into train, test, valid. Return all 3

    # Sample code for shuffling. will use later
    # p = numpy.random.permutation(len(a))

    ret = {
        "train_X": train_X,
        "train_Y": train_Y,
        "labels": dirnames
    }

    return ret


