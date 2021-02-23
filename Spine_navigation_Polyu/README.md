In this project robot was scanning the surface of human spine phantom to capture Ultrasound images and respective robot coordinates for further 3D reconstruction.

The work was presented at IRC2019 named "3D Ultrasound Imaging of Scoliosis with Force-Sensitive Robotic Scanning", [publication website](https://ieeexplore.ieee.org/document/8675657).
![Alt text](Navigation_robotic_Polyu/GUI/Setup.png?raw=true "Setup")

![Alt text](Navigation_robotic_Polyu\GUI\Capture.JPG?raw=true "GUI")

![GitHub Logo](/Navigation_robotic_Polyu\GUI\Capture.JPG)
Robotic control for two cases:
1) (Force control) Robot moves along pre-recorded trajectory and applies force control to assure that it always applies constant preasure to the phantom's skin.

2) (Force-Image control) Robot's first and last points are assigned at the beguinning and at the end of the spine. From first to the last point robot moves applying the constant force and adjust the positions so that the spine is in the center of the ultrasound image 

**Force control**

File to launch: *Move_force_trajectory.py*


**Force-Image control**

File to launch: *Image_force_control_UR5.py*
Method example of center of spine detection for further navigation can be seen
at *Examples/Center_spine_detection_Max_Mean_point.py*