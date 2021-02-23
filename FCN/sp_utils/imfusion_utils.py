# Important at start of the project to run the following
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/ImFusionLib/plugins

import os
import imfusion
import numpy as np


def open_sweep(filename):
    imageset = imfusion.open(filename)

    # if the length is higher than one it means that probably the file also contains the labels - take only the
    # sweep
    sweep = np.array(imageset[0])
    sweep = np.squeeze(sweep)  # now we only have the NFrames x Width x Length sequence
    return sweep


imfusion.init()
print("initialized")
a = imfusion.open("/media/maryviktory/My Passport/IROS 2020 TUM/DATASETs/Dataset/initial data set all/Maria_V/spinous_process/spinous_process.imf")
b = np.array(a[0])
print(b.shape)
#
#
#
#
# print(len(a))
# print("opened")

# reader = SweepReader()
# imageset = imfusion.SharedImageSet(np.zeros([10, 600, 800, 1], dtype='uint8'))
#
# reader.open_sweep("/home/maria/Desktop/prova.png")