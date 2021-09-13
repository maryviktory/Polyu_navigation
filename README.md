# Spinous process localization in ultrasound images with FCN + robotic navigation

This work is an implementation of the paper: "Follow the Curve: Robotic-Ultrasound Navigation with Learning Based Localization of Spinous Processes for Scoliosis Assessment"
Authors: Maria Victorova, Michael Ka-Shing Lee, David Navarro-Alarcon and Yongping Zheng

## "FCN" folder
This folder contains the script to train the network to detect Spinous Process. The network input is ultrasound images of spinous process and correspondent labels in a form of Gaussian drawn around the spinous process location. The image shows the label placed manually on the location of spinous process (a), the generated label (b) generated with "sp_utils/label_spinous_for_segmentation.py" from the previous image and (c) shows the resulted image with detected spinous on a single frame, which can be computed with "FCN/test.py" (enable in "config.py" config.TEST.PLOT = True).

![alt text](https://github.com/maryviktory/Polyu_navigation/blob/master/FCN_single_image_output.png?raw=true)

When you have an ultrasound scan of the back there are images of both "vertebrae" and "intervertebral gap", after you label each image with the spinous process, putting a point where the location is you end up with a folder with folders: 
-Name_of_subject
--Images
--Labels

Since the input for the network is only the images containing the spinous process, you can split the images in the scan to classes "vertebrae" and "intervertebral gap" with script "sp_utils/split_dataset_to_classes_for_CNN.py"

When all the sweeps are processed the dataset can be split to train, val, test with "sp_utils/create_data_sets_heatmaps.py".

The dataset now is ready for training with "run_train_polyu.py". The training can be performed locally or with use of Polyaxon, otherwise you might choose using docker to load on a server, then use Dockerfile from "Docker" folder and launch the 


