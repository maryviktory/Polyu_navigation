#Spinous Process localization in Ultrasound Images with Heatmap Approach

- The network model is similar with Simple Baselines for Human Pose Estimation and Tracking. Bin Xiao1∗, Haiping Wu2∗†, and Yichen We  https://arxiv.org/pdf/1804.06208.pdf 
- The data set consists of 19 subjects with around 2000 Ultrasound Linear B-mode images of vertebrae.
- The model is based on the ResNet backbone, which is ResNet 18, pretrained on Imagenet, then fine-tuned on Ultrasound images for classification task to classify "vertebra" and "intervertebral gap".
- The additional 3 deconvolutional layers are added on top of ResNet for Heatmap generation.

#Current functionality:
- Train model (The input of the network - Ultrasound Imgages of spine, label - 2D gaussian around the landmark (spinous process))
- Test model: 

1) images with label to test accuracy - The accuracy is given in mm, taking in account the probe size of 48mm and image width of 480px.
2) images without label to visually estimate the error
