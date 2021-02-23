# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
# import yaml

import numpy as np
from easydict import EasyDict as edict


config = edict()

config.OUTPUT_DIR = ''
config.LOG_DIR = ''
config.DATA_DIR = ''
config.GPUS = '0'
config.WORKERS = 0
config.PRINT_FREQ = 20

# testing
config.TEST = edict()

config.TEST.Windows = True
# size of images for each device
config.TEST.BATCH_SIZE = 1
# Test Model
config.TEST.PLOT = False #plots each frame
config.TEST.VIDEO = True #record video with detected point and label
config.TEST.PLOT_VIDEO = False #plot each frame of the video
config.TEST.labels_exist = True #True if the sweep is labelled
# config.TRAIN.SWEEP_TRJ_PLOT = True
config.TEST.SAVE_NPZ_FILE = False
config.TEST.PLOT_SMOOTH_LABEL_TRAJECTORY = True #True if the file already exists and need to be loaded
## Augmentation of data
config.TEST.enable_transform = False


config.TEST.MODEL_FILE = "/media/maryviktory/My Passport/spine navigation Polyu 2021/DATASET_polyu/models_FCN/best_model_exp40184.pt"
config.TEST.Windows_MODEL_FILE = "D:\spine navigation Polyu 2021\DATASET_polyu\models_FCN\\human_best_model_exp40184.pt"

config.TEST.PROBE_SIZE = 80 #old wifi probe 48mm, new probe 80
config.TEST.ORIGINAL_IMAGE_SIZE = 640 #width, IFL - 480, Polyu - 640
config.TEST.ORIGINAL_IMAGE_HEIGHT = 480
config.TEST.input_im_size = 224
config.TEST.heatmap_size = 56
config.TEST.normalization = True
config.TEST.JSON_WS = {
    "Command": "Us_Config",
    "US_module": 2,  # "US_DEVICE_UVF = 1", "US_DEVICE_PALM = 2", "US_DEVICE_TERASON = 3"
    "Posture_module": 1,  # POSTURE_SENSOR_UVF = 1 ,POSTURE_SENSOR_TRAKSTAR =2,POSTURE_SENSOR_REALSENSE =3
    "US_module_config": "",
    "Posture_module_config": ".\\test.uvf"
}

def _update_dict(k, v):
    if k == 'DATASET':
        if 'MEAN' in v and v['MEAN']:
            v['MEAN'] = np.array([eval(x) if isinstance(x, str) else x
                                  for x in v['MEAN']])
        if 'STD' in v and v['STD']:
            v['STD'] = np.array([eval(x) if isinstance(x, str) else x
                                 for x in v['STD']])
    if k == 'MODEL':
        if 'EXTRA' in v and 'HEATMAP_SIZE' in v['EXTRA']:
            if isinstance(v['EXTRA']['HEATMAP_SIZE'], int):
                v['EXTRA']['HEATMAP_SIZE'] = np.array(
                    [v['EXTRA']['HEATMAP_SIZE'], v['EXTRA']['HEATMAP_SIZE']])
            else:
                v['EXTRA']['HEATMAP_SIZE'] = np.array(
                    v['EXTRA']['HEATMAP_SIZE'])
        if 'IMAGE_SIZE' in v:
            if isinstance(v['IMAGE_SIZE'], int):
                v['IMAGE_SIZE'] = np.array([v['IMAGE_SIZE'], v['IMAGE_SIZE']])
            else:
                v['IMAGE_SIZE'] = np.array(v['IMAGE_SIZE'])
    for vk, vv in v.items():
        if vk in config[k]:
            config[k][vk] = vv
        else:
            raise ValueError("{}.{} not exist in config.py".format(k, vk))


# def update_config(config_file):
#     exp_config = None
#     with open(config_file) as f:
#         exp_config = edict(yaml.load(f))
#         for k, v in exp_config.items():
#             if k in config:
#                 if isinstance(v, dict):
#                     _update_dict(k, v)
#                 else:
#                     if k == 'SCALES':
#                         config[k][0] = (tuple(v))
#                     else:
#                         config[k] = v
#             else:
#                 raise ValueError("{} not exist in config.py".format(k))


# def gen_config(config_file):
#     cfg = dict(config)
#     for k, v in cfg.items():
#         if isinstance(v, edict):
#             cfg[k] = dict(v)
#
#     with open(config_file, 'w') as f:
#         yaml.dump(dict(cfg), f, default_flow_style=False)


def update_dir(model_dir, log_dir, data_dir):
    if model_dir:
        config.OUTPUT_DIR = model_dir

    if log_dir:
        config.LOG_DIR = log_dir

    if data_dir:
        config.DATA_DIR = data_dir

    config.DATASET.ROOT = os.path.join(
            config.DATA_DIR, config.DATASET.ROOT)

    config.TEST.COCO_BBOX_FILE = os.path.join(
            config.DATA_DIR, config.TEST.COCO_BBOX_FILE)

    config.MODEL.PRETRAINED = os.path.join(
            config.DATA_DIR, config.MODEL.PRETRAINED)


def get_model_name(cfg):
    name = cfg.MODEL.NAME
    full_name = cfg.MODEL.NAME
    extra = cfg.MODEL.EXTRA
    if name in ['pose_resnet']:
        name = '{model}_{num_layers}'.format(
            model=name,
            num_layers=extra.NUM_LAYERS)
        deconv_suffix = ''.join(
            'd{}'.format(num_filters)
            for num_filters in extra.NUM_DECONV_FILTERS)
        full_name = '{height}x{width}_{name}_{deconv_suffix}'.format(
            height=cfg.MODEL.IMAGE_SIZE[1],
            width=cfg.MODEL.IMAGE_SIZE[0],
            name=name,
            deconv_suffix=deconv_suffix)
    else:
        raise ValueError('Unkown model: {}'.format(cfg.MODEL))

    return name, full_name


if __name__ == '__main__':
    import sys
    # gen_config(sys.argv[1])