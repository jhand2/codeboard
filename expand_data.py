#!/usr/bin/python3
"""
A tool to take a handwritten math symbols dataset and construct missing symbols
often used in handwritten code by manupulating the original symbols.

Original dataset: https://www.kaggle.com/xainano/handwrittenmathsymbols
"""

import cv2
import numpy as np
import sys
import os
import imageio
import scipy.ndimage
import math

# file globals
image_size = 45

# Main functios to fill in extra symbol data

def build_dot():
    print("Building dataset for . symbol")
    dataset = load_dir("!", maxn=None)
    print(dataset.shape)

    dataset[:, 0:round(image_size * 0.9), :] = 255
    write_dataset(dataset, "dot")


def build_mod():
    print("Building dataset for % symbol")
    dataset = load_dir("div", maxn=None)
    print(dataset.shape)

    rotated = scipy.ndimage.interpolation.rotate(dataset, 45, (2, 1), reshape=False, cval=255., order=1).astype(np.uint8)

    write_dataset(rotated, "%")


def build_times():
    print("Building dataset for * symbol")
    x_dataset = load_dir("times", maxn=None).astype(np.uint8)
    print(x_dataset.shape)

    rotated = scipy.ndimage.rotate(x_dataset, 45, (2, 1), reshape=False, cval=255., order=1).astype(np.uint8)
    print(rotated.shape)

    combined = np.bitwise_and(x_dataset, rotated)
    print(combined.shape)
    write_dataset(combined, "*")


def build_semi_colon():
    print("Building dataset for ; symbol")
    dataset = load_dir("!", maxn=None)
    print(dataset.shape)

    # flip
    rotated = np.rot90(dataset, 2, axes=(1,2))
    print(rotated.shape)

    # roll bottom 30% progressively
    cutoff = int(image_size * 0.6)
    for i in range(cutoff, image_size):
        # Add exponentially increasing roll amount. Scaled so no black pixels
        # will roll off the end
        mag = math.floor(-1 * (2/image_size) * (i - cutoff) ** 2)
        rotated[:, i, :] = np.roll(rotated[:,i,:], mag, axis=(1))

    write_dataset(rotated, ";")


def write_dataset(dataset, symname):
    dest_dir = os.path.join("data", symname)
    if not os.path.exists(dest_dir):
        os.mkdir(dest_dir)

    for i in range(dataset.shape[0]):
        imageio.imwrite(os.path.join(dest_dir, '%s_%d.jpg' % (symname, i)), dataset[i, :, :].astype(np.uint8))
    print("Done!")

def load_dir(d, maxn=None):
    full_path = os.path.join(".", "data", d)
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


fmap = {
    "dot": build_dot,
    "mod": build_mod,
    "times": build_times,
    "semi_colon": build_semi_colon
}

def usage(msg):
    print("Error: %s" % msg)
    print("Usage: %s [dot|mod|times|semi_colon]" % sys.argv[0])

if __name__ == "__main__":
    if len(sys.argv) < 2:
        usage("Too few arguments to %s" % sys.argv[0])
        exit(1)

    if sys.argv[1] in fmap:
        # If we have a function to construct data for this symbol,
        # execute it
        fmap[sys.argv[1]]()
    else:
        usage("Don't know how to build data for symbol \"%s\"" % sys.argv[1])

