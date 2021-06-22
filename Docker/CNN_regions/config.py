from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
# import yaml

import numpy as np
from easydict import EasyDict as edict


config = edict()
config.Augmentation = True
config.BATCH_SIZE = 12