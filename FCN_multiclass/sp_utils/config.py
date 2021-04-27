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


# pose_resnet related params
POSE_RESNET = edict()
POSE_RESNET.NUM_LAYERS = 18
POSE_RESNET.DECONV_WITH_BIAS = False
POSE_RESNET.NUM_DECONV_LAYERS = 3
POSE_RESNET.NUM_DECONV_FILTERS = [256, 256, 256]
POSE_RESNET.NUM_DECONV_KERNELS = [4, 4, 4]
POSE_RESNET.FINAL_CONV_KERNEL = 1
POSE_RESNET.TARGET_TYPE = 'gaussian'
POSE_RESNET.HEATMAP_SIZE = [64, 64]  # width * height, ex: 24 * 32
POSE_RESNET.SIGMA = 2

MODEL_EXTRAS = {
    'pose_resnet': POSE_RESNET,
}

# common params for NETWORK
config.MODEL = edict()
config.MODEL.NAME = 'pose_resnet'
config.MODEL.INIT_WEIGHTS = True
config.MODEL.PRETRAINED = "D:\spine navigation Polyu 2021\DATASET_polyu\FCN_PWH_train_dataset_heatmaps\data_19subj_multiclass_heatmap\pretrained model\model_best_resnet_fixed_False_pretrained_True_Multiclass_spine_4classes_exp6106.pt"
#'/media/maryviktory/My Passport/IROS 2020 TUM/DATASETs/models/spinous_best_18_retrain.pt'
config.MODEL.PRETRAINED_NAS = "SpinousProcessData/FCN_PWH_train_dataset_heatmaps/model_best_resnet_fixed_False_pretrained_True_data_19subj_2_exp_36776.pt"
#"SpinousProcessData/spinous_best_18_retrain.pt"
config.MODEL.Imagenet_pretrained = "C:\\Users\Administrator\Documents\dataset_multiclass_FCN\pretrained model\\resnet18-5c106cde.pth"

config.MODEL.NUM_JOINTS = 1
config.MODEL.num_classes = 4
config.MODEL.IMAGE_SIZE = [256, 256]  # width * height, ex: 192 * 256
config.MODEL.EXTRA = MODEL_EXTRAS[config.MODEL.NAME]

config.MODEL.STYLE = 'pytorch'

config.LOSS = edict()
config.LOSS.USE_TARGET_WEIGHT = True

# DATASET related params
config.DATASET = edict()
config.DATASET.PATH = "C:\\Users\Administrator\Documents\dataset_multiclass_FCN"
#"/media/maryviktory/My Passport/spine navigation Polyu 2021/DATASET_polyu/FCN_PWH_train_dataset_heatmaps/data_19subj_2"
#"/media/maryviktory/My Passport/IROS 2020 TUM/Spine navigation vertebrae tracking/FCN_spine_point_regression/Small_dataset_heatmaps"
config.DATASET.OUTPUT_PATH = os.path.join(config.DATASET.PATH,"output")
#"/media/maryviktory/My Passport/spine navigation Polyu 2021/DATASET_polyu/FCN_PWH_train_dataset_heatmaps/data_19subj_2"
#"/media/maryviktory/My Passport/IROS 2020 TUM/Spine navigation vertebrae tracking/FCN_spine_point_regression/Small_dataset_heatmaps"
config.DATASET.PATH_NAS = 'SpinousProcessData/FCN_PWH_train_dataset_heatmaps/data_19subj_2'
#"SpinousProcessData/Heatmaps_spine"


# train
config.TRAIN = edict()

config.TRAIN.POLYAXON = False
config.TRAIN.BATCH_SIZE = 12
config.TRAIN.VAL_BATCH_SIZE = 12
config.TRAIN.THRESHOLD = 0.5 #distance error maximum allowed - 2.4mm (threshold = 0.5)
config.TRAIN.Augmentation = True
config.TRAIN.UPDATE_WEIGHTS = True
config.TRAIN.LR = 0.001
config.TRAIN.END_EPOCH = 100
config.TRAIN.SWEEP_TRJ_PLOT = True

#EXTRA
config.TRAIN.LR_FACTOR = 0.1
config.TRAIN.LR_STEP = [90, 110]
config.TRAIN.OPTIMIZER = 'adam'
config.TRAIN.MOMENTUM = 0.9
config.TRAIN.WD = 0.0001
config.TRAIN.loss_alpha = 5000
config.TRAIN.weight_decay = 0.1
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
config.TRAIN.SWEEP_TRJ_PLOT = True
config.TEST.SAVE_NPZ_FILE = False
config.TEST.PLOT_SMOOTH_LABEL_TRAJECTORY = False #True if the file already exists and need to be loaded
## Augmentation of data
config.TEST.enable_transform = False
config.TEST.THRESHOLD = 0.5

config.TEST.data_dir = "D:\spine navigation Polyu 2021\DATASET_polyu\FCN_PWH_train_dataset_heatmaps\data_19subj_multiclass_heatmap "
config.TEST.Windows_data_dir = "D:\spine navigation Polyu 2021\DATASET_polyu\FCN_PWH_train_dataset_heatmaps\data_19subj_2"
#"/media/maryviktory/My Passport/IROS 2020 TUM/Spine navigation vertebrae tracking/FCN_spine_point_regression/Heatmaps_spine"
# config.TEST.sweep_data_dir = "/media/maryviktory/My Passport/IROS 2020 TUM/DATASETs/Dataset/DATA_toNas_for CNN_IPCAI/data set patients images/Maria_T/Images" #images_subset
config.TEST.sweep_data_dir = "/media/maryviktory/My Passport/spine navigation Polyu 2021/DATASET_polyu/FCN_PWH_dataset_heatmaps_all"
config.TEST.Windows_sweep_data_dir = "D:\spine navigation Polyu 2021\DATASET_polyu\PWH_sweeps\Subjects dataset"
#"/media/maryviktory/My Passport/IROS 2020 TUM/Spine navigation vertebrae tracking/FCN_spine_point_regression/Dataset_Heatmaps_all_subjects"
# config.TEST.data_dir_w_out_labels = "/media/maryviktory/My Passport/IROS 2020 TUM/Spine navigation vertebrae tracking/FCN_spine_point_regression/Spinous_and_Gap_without_labels"
config.TEST.data_dir_w_out_labels = "/media/maryviktory/My Passport/IROS 2020 TUM/Spine navigation vertebrae tracking/FCN_spine_point_regression/small_test"
config.TEST.Windows_data_dir_w_out_labels = "D:\spine navigation Polyu 2021\DATASET_polyu\FCN_PWH_dataset_heatmaps_all"

config.TEST.save_dir = "/media/maryviktory/My Passport/spine navigation Polyu 2021/DATASET_polyu/FCN_PWH_train_dataset_heatmaps/output"
#"/media/maryviktory/My Passport/IROS 2020 TUM/Spine navigation vertebrae tracking/FCN_spine_point_regression/Spinous_positions_sweeps"
config.TEST.Windows_save_dir = "D:\spine navigation Polyu 2021\DATASET_polyu\FCN_PWH_train_dataset_heatmaps\output"
config.TEST.MODEL_FILE = "/media/maryviktory/My Passport/spine navigation Polyu 2021/DATASET_polyu/models_FCN/best_model_exp40184.pt"
config.TEST.Windows_MODEL_FILE = "C:\\Users\Administrator\Documents\dataset_multiclass_FCN\output\\2\model30.pt"
#"D:\spine navigation Polyu 2021\DATASET_polyu\models_FCN\\human_best_model_exp40184.pt"
#'/media/maryviktory/My Passport/spine navigation Polyu 2021/DATASET_polyu/models_FCN/best_model_exp36817.pt'
#"/media/maryviktory/My Passport/IROS 2020 TUM/DATASETs/models/best_model.pt"
config.TEST.data_npz = "/media/maryviktory/My Passport/spine navigation Polyu 2021/DATASET_polyu/FCN_PWH_train_dataset_heatmaps/output"
config.TEST.Windows_data_npz="D:\spine navigation Polyu 2021\DATASET_polyu\FCN_PWH_train_dataset_heatmaps\output"

config.TEST.PROBE_SIZE = 80 #old wifi probe 48mm, new probe 80
config.TEST.ORIGINAL_IMAGE_SIZE = 640 #width, IFL - 480, Polyu - 640
config.TEST.ORIGINAL_IMAGE_HEIGHT = 480
config.TEST.input_im_size = 224
config.TEST.heatmap_size = 56
config.TEST.normalization = True


# config.TEST.FLIP_TEST = False
# config.TEST.POST_PROCESS = True
# config.TEST.SHIFT_HEATMAP = True


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