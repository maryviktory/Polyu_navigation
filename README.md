# Spinous process localization in ultrasound images with FCN + robotic navigation

This work is an implementation of the paper: "Follow the Curve: Robotic-Ultrasound Navigation with Learning Based Localization of Spinous Processes for Scoliosis Assessment"
Authors: Maria Victorova, Michael Ka-Shing Lee, David Navarro-Alarcon and Yongping Zheng

![alt text](https://github.com/maryviktory/Polyu_navigation/blob/master/FCN_architechture_full.png?raw=true)

### Spinous Process Localization
## "FCN" folder
This folder contains the script to train the network to detect Spinous Process. The network input is ultrasound images of spinous process and correspondent labels in a form of Gaussian drawn around the spinous process location. The image shows the label placed manually on the location of spinous process (a), the generated label (b) generated with "sp_utils/label_spinous_for_segmentation.py" from the previous image and (c) shows the resulted image with detected spinous on a single frame, which can be computed with "FCN/test.py" (enable in "config.py" config.TEST.PLOT = True).

![alt text](https://github.com/maryviktory/Polyu_navigation/blob/master/FCN_single_image_output.png?raw=true)

When you have an ultrasound scan of the back there are images of both "vertebrae" and "intervertebral gap", after you label each image with the spinous process, putting a point where the location is you end up with a folder with folders: 
-Name_of_subject
--Images
--Labels

Since the input for the network is only the images containing the spinous process, you can split the images in the scan to classes "vertebrae" and "intervertebral gap" with script "sp_utils/split_dataset_to_classes_for_CNN.py"

When all the sweeps are processed the dataset can be split to train, val, test with "sp_utils/create_data_sets_heatmaps.py".

The dataset now is ready for training with "run_train_polyu.py". The training can be performed locally or with use of Polyaxon, otherwise you might choose using docker to load on a server, then use Dockerfile from "Docker" folder and launch the "Docker/FCN_one_class/run_train.py"

## "FCN_multiclass"
This is a two headed network for localization and classification tasks.
![alt text](https://github.com/maryviktory/Polyu_navigation/blob/master/region_network_two_heads.png?raw=true)

```run_train_multiclass_polyu.py```
The dataset looks as follows:
```-train #(ultrasound images)
 -Sacrum
 -Thoracic
 -Lumbar
 
-val #(ultrasound images)
 -Sacrum
 -Thoracic
 -Lumbar
Labels_heatmaps #labels for all images included in training and val set
```
## "Docker" folder
Containes Dockerfile to build a Docker image. The docker-compose.yaml is a convenient way to build Docker image with specific parameters and sequncially launch .py scripts inside this image. The docker folder contains scripts for spinous process localization (FCN_one_class) and multitask training for 2 heads (localization + classication, run_train_multiclass_polyu.py) as well as experimental script for 3 heads (localization of spinous process, sacrum and classification) -  run_train_multiclass_polyu_3heads.py


```$ (if you launch docker container through terminal)
$docker run -it --shm-size 8G -p 10000:6006 --name docker_container_fcn_cnn --gpus all -v /storage/maryviktory/FCN_CNN:/workspace maryviktory:fcn_cnn
$ (if you launch docker-compose.yaml with all parameters specified inside, just run, the container will be removed after the execution and the outputs will be saved in "runs" folder)
$docker-compose run --rm cms
$ (to use tensorboard - example)
$cd \workspace
$ tensorboard --bind_all --logdir=./data_19subj_15train_no_sacrum_lab/runs/```


###Robotic navigation
The script is based on multiprocessing classes which are controlled by GUI. 
*class Window* is a GUI, which starts the processes on the button "move" click.
*Move_thread* is the main thread, which collect the information from *Force_Thread*, which received the data from force sensor at 32fps, and from *FCN_thread*, which yields the spinous process location on the images received from the websocket communication process *get_image_fun*. All received information is used for the robotic velocity calculation and send the command with correspondent move. The tread also includes Kalman filtering of the spinous process locations. 

###US_probe_communication

Is used to stream raw data from USB Ultrasound probe to python script through WebSocket using multiprocessing. The image is reconstructed with openCV and sent to another process for FCN processing. 


## Contact
maryviktory@gmail.com
