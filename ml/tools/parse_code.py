#!/usr/bin/python3

import sys
import os
import imageio

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from im_utils import parse_code_im

char_size = 45

if __name__ == "__main__":
    if len(sys.argv) > 1:
        impath = sys.argv[1]
    else:
        print("Please supply an image path")
        exit(1);

    idata = imageio.imread(impath)
    parse_code_im(idata, char_size)

