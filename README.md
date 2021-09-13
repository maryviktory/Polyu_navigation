# Spinous process localization in ultrasound images with FCN + robotic navigation

This work is an implementation of the paper: "Follow the Curve: Robotic-Ultrasound Navigation with Learning Based Localization of Spinous Processes for Scoliosis Assessment"
Authors: Maria Victorova, Michael Ka-Shing Lee, David Navarro-Alarcon and Yongping Zheng

## "FCN" folder
This folder contains the script to train the network to detect Spinous Process. The network input is ultrasound images of spinous process and correspondent labels in a form of Gaussian drawn around the spinous process location. The image shows the label placed manually on the location of spinous process (a), the generated label (b) generated with "sp_utils/label_spinous_for_segmentation.py" from the previous image and (c) shows the resulted image with detected spinous on a single frame, which can be computed with "FCN/test.py" (enable in "config.py" config.TEST.PLOT = True).

![alt text](https://github.com/maryviktory/Polyu_navigation/blob/master/FCN_single_image_output.png?raw=true)
