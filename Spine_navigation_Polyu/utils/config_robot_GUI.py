from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
# import yaml

import numpy as np
from easydict import EasyDict as edict
import urx

config = edict()
config.Mode_Develop = True #Allows to launch code without robot
config.IP_ADRESS = "192.168.0.100" # "158.132.172.194"
config.FOV = 0.045 #Field of view of the robot
config.alpha = 0.1
config.robot_TCP = (0, 0, 0.315, 0, 0, 0) #306 for smaller holder
config.robot_payload = 1 #KG

config.w = 400
config.hg = 360
config.Trajectory_n = 'FCN_force'
config.default_distance = 0.3 #meters
config.maximum_distance = 2 #meters
config.VELOCITY_up = 0.002 #0.004
# config.robot = urx.Robot(config.IP_ADRESS, use_rt=True)
# print("robot in config")
config.MODE = edict()
config.MODE.Develop = False #Allows to launch code without robot
config.MODE.FCN = True
config.MODE.FCN_vcontrol = True
config.MODE.FORCE = True
config.MODE.BASE_csys = False
config.MODE.exp_smoothing_velocity = True

config.MODE.median_filter = False
config.Median_kernel_size = 31

config.FORCE = edict()

config.FORCE.Fref_first_move = 4 #Force [N]
config.FORCE.Fref = 5 #Force [N]
config.FORCE.Fmax = 30 #Force [N]
config.FORCE.Fcrit = 35 #Force [N]
config.FORCE.K_delta = 0.0005
config.FORCE.Kf = 0.0004 #0.002  Kunal - 0.0004
config.FORCE.Kz = 0.6
config.FORCE.v = 0.007 #move first point
config.FORCE.a = 0.01 #move first point
config.FORCE.thr = 0.004
config.FORCE.K_torque = 0.07 #0.07

config.IMAGE = edict()

config.IMAGE.K_im_out = 0.5
config.IMAGE.K_im_near = 0.2
# config.IMAGE.FCN = False
config.IMAGE.subject_mode = "phantom" #"human" #phantom

config.IMAGE.JSON_WS = {
    "Command": "Us_Config",
    "US_module": 2,  # "US_DEVICE_UVF = 1", "US_DEVICE_PALM = 2", "US_DEVICE_TERASON = 3"
    "Posture_module": 1,  # POSTURE_SENSOR_UVF = 1 ,POSTURE_SENSOR_TRAKSTAR =2,POSTURE_SENSOR_REALSENSE =3
    "US_module_config": "",
    "Posture_module_config": ".\\test.uvf"
}

config.IMAGE.LOCAL_HOST = "ws://localhost:4100"
config.IMAGE.Windows_MODEL_FILE = "D:\spine navigation Polyu 2021\DATASET_polyu\models_FCN\\human_best_model_exp40184.pt"
config.IMAGE.Windows_MODEL_FILE_PHANTOM = "D:\spine navigation Polyu 2021\DATASET_polyu\models_FCN\\best_model_exp49560_phantom_6scans.pt"
config.IMAGE.PROBE_SIZE = 0.08 #old wifi probe 48mm, new probe 80 mm = 0.08m
config.IMAGE.ORIGINAL_IMAGE_SIZE = 640 #width, IFL - 480, Polyu - 640
config.IMAGE.ORIGINAL_IMAGE_HEIGHT = 480
config.IMAGE.input_im_size = 224
config.IMAGE.heatmap_size = 56
config.IMAGE.normalization = True

config.IMAGE.Windows = True
# size of images for each device
config.IMAGE.BATCH_SIZE = 1
config.IMAGE.probability_threshold = 0.5
# Test Model
config.IMAGE.input_im_size = 224
config.IMAGE.PLOT = False #plots each frame
config.IMAGE.VIDEO = True #record video with detected point and label
config.IMAGE.PLOT_VIDEO = False #plot each frame of the video
config.IMAGE.labels_exist = True #True if the sweep is labelled
# config.TRAIN.SWEEP_TRJ_PLOT = True
config.IMAGE.SAVE_NPZ_FILE = True
config.IMAGE.SAVE_PATH = "D:\spine navigation Polyu 2021\\robot_trials_output"
config.IMAGE.PLOT_SMOOTH_LABEL_TRAJECTORY = True #True if the file already exists and need to be loaded
## Augmentation of data
config.IMAGE.enable_transform = False

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

# common params for NETWORK
config.MODEL = edict()
config.MODEL.NAME = 'pose_resnet'
config.MODEL.INIT_WEIGHTS = True
# config.MODEL.PRETRAINED = '/media/maryviktory/My Passport/spine navigation Polyu 2021/DATASET_polyu/model_best_resnet_fixed_False_pretrained_True_data_19subj_2_2_classes_exp6105.pt'
#'/media/maryviktory/My Passport/IROS 2020 TUM/DATASETs/models/spinous_best_18_retrain.pt'
# config.MODEL.PRETRAINED_NAS = "SpinousProcessData/FCN_PWH_train_dataset_heatmaps/model_best_resnet_fixed_False_pretrained_True_data_19subj_2_exp_36776.pt"
#"SpinousProcessData/spinous_best_18_retrain.pt"
config.MODEL.NUM_JOINTS = 1
config.MODEL.IMAGE_SIZE = [256, 256]  # width * height, ex: 192 * 256
# config.MODEL.EXTRA = MODEL_EXTRAS[config.MODEL.NAME]

config.MODEL.STYLE = 'pytorch'

# config.LOSS = edict()
# config.LOSS.USE_TARGET_WEIGHT = True