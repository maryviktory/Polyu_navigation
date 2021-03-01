import json
import websocket
from PIL import  Image
import numpy as np
import cv2
import torch
# import torchvision
import FCN.sp_utils as utils
import logging
import keyboard
import os
import sys
import time
from threading import Thread
import pandas as pd
from multiprocessing import Process
import multiprocessing
from Spine_navigation_Polyu.utils.config_robot_GUI import config
from Spine_navigation_Polyu.utils.functions import run_FCN_streamed_image,save_video, AverageMeter
#https://techtutorialsx.com/2018/11/08/python-websocket-client-sending-binary-content/




message = config.IMAGE.JSON_WS
json_mylist = json.dumps(message, separators=(',', ':'))

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(os.path.join(config.IMAGE.SAVE_PATH,'output_original.avi'), fourcc, 3.0, (1280, 480))  # for images of size 480*640

ws = websocket.WebSocket()
ws.connect("ws://localhost:4100")
ws.send(json_mylist)

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.info("STARTING TEST")

# cpu_count = multiprocessing.cpu_count()
# print("CPU counting", cpu_count)

class Get_Image(Thread):
    '''# we need to receive the data of length 307329'''
    def __init__(self,Stop_Thread):
        Thread.__init__(self)

        self.image = np.zeros(0)
        self.num = 0
        self.frame_num = 0
        self.stop_thread = Stop_Thread
        # self.threads_stopper = self.stop_thread.threads_stopper

    def run(self):
          # to stop all threads at the same time
        time_start = time.time()

        num_text = 0
        while True: #self.num in range (100)

            if self.stop_thread.threads_stopper ==False:
                # print("received")
                # print(self.stop_thread.threads_stopper)
                binAnswer = ws.recv_frame()

                if websocket.ABNF.OPCODE_MAP[binAnswer.opcode] == "binary":
                    # print("image received")
                    image_byte_array = (binAnswer.data)[129:]
                    # Create a PIL Image from our pixel array.
                    image = Image.frombuffer('L',(640, 480),image_byte_array)
                    if image.size:
                        self.image = image
                        self.frame_num = self.frame_num + 1
                    image_display = np.array(self.image)

                else:
                    print("Text output")
                    num_text = num_text+1
                    if num_text ==2:
                        print("Ready to stream")

                self.num=self.num+1
                '''To interupt the thread '''
                # global threads_stopper
                # print("receive")

            else:
                 # or stop_threads==True
                print("Get_Image thread is stopped")
                time_thread = time.time() - time_start
                fps_im = self.num/time_thread
                print("fps Get Image thread {}, average time per cycle {}".format(fps_im, 1/fps_im))
                # ws.close()
                break

class FCN_Thread(Thread):
    '''Run FCN pre-trained model on online streamed images
    calculate the coordinate'''
    def __init__(self,Get_Image,Stop_Thread):
        Thread.__init__(self)

        self.result_im = np.zeros(0)
        self.get_image = Get_Image
        self.image_flag = False
        self.num = 0
        self.stop_thread = Stop_Thread
        self.threads_stopper = self.stop_thread.threads_stopper
    def run(self):

          # to stop all threads at the same time
        probability = np.zeros(0)
        X = np.zeros(0)
        Y = np.zeros(0)

        num= 0
        patient = "US_probe"
        time_inference = AverageMeter()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if config.IMAGE.Windows == True:
            model = utils.model_pose_resnet.get_pose_net(config.IMAGE.Windows_MODEL_FILE, is_train=False)
            logger.info('=> loading model from {}'.format(config.IMAGE.Windows_MODEL_FILE))
            model.load_state_dict(
                torch.load(config.IMAGE.Windows_MODEL_FILE, map_location=device)['model_state_dict']) #map_location=torch.device('cpu')
            model.eval()  # Super important for testing! Otherwise the result would be random
            if torch.cuda.is_available():
                model.cuda()

            # logger.info("Setting model to eval. It is important for testing")
        else:
            print("Model is not defined")
            model = []

        time_start = time.time()
        n = 0
        with torch.no_grad():
            while True:

                if self.get_image.image.size: #image we get from WebSocket connection
                    # print("image received")
                    self.image_flag = True
                    # print("frame num of received image in FCN thread", self.get_image.frame_num)
                    start_time = time.time()

                    inputs,pred,probability, X, Y, frame_probability = run_FCN_streamed_image(self.get_image.image,model, device, probability,X,Y,logger,config)
                    # print("x",X)
                    if config.IMAGE.VIDEO == True:
                        self.result_im = save_video(out, inputs, pred, frame_probability, patient,config, target=None, labels=None)
                    #TODO: accumulate array of robot positions and force values to write it to npz file
                    time_one = time.time() - start_time
                    time_inference.update(time_one)
                    n = n+1
                    self.num = self.num + 1
                else:
                    self.image_flag = False

                '''To interrupt the thread'''

                if self.stop_thread.threads_stopper==True:
                    print("FCN_Thread stopped")
                    if time_inference.avg !=0:
                        print("fps FCN thread {}, average time per cycle {}".format(1/time_inference.avg, time_inference.avg))

                    time_thread = time.time() - time_start
                    fps_im = self.num / time_thread
                    print("fps FCN thread {}, average time per cycle {}".format(fps_im, 1 / fps_im))
                    print("NUM FCN ", self.num, n)
                    # print(X)
                    if config.IMAGE.SAVE_NPZ_FILE:
                        while os.path.exists("FCNthread_output%s.npz" % num):
                            num = num+ 1
                        np.savez(os.path.join(config.IMAGE.SAVE_PATH, "FCNthread_output%s.npz" % num),
                                 probability=probability, x=X, y=Y)

                        pd_frame = pd.DataFrame({"frame_probability":probability, "X":X, "Y":Y})
                        pd_frame.to_csv(os.path.join(config.IMAGE.SAVE_PATH, "FCNthread_output%s.csv" % num))
                    # out.release()
                    # ws.close()
                    # os._exit(0)
                    break


class Stop_Thread(Thread):
    '''# To set a flag to stop the treads. By initiating the separate thread we make sure that the keyboard
    pressed value is received immediately'''

    def __init__(self):
        Thread.__init__(self)
        self.threads_stopper = False # to stop all threads at the same time
        self.num = 0
    def run(self):
        print("wait")
        time_start = time.time()
        i = 0
        while True:
            self.num = self.num +1
            # print(i)
            if keyboard.is_pressed('c') or (self.threads_stopper == True):
                self.threads_stopper = True
                print("Killing threads")
                time_thread = time.time() - time_start
                fps_im = self.num / time_thread
                print("Total time {},fps Stop_Thread thread {}, average time per cycle {}".format(time_thread,fps_im, 1 / fps_im))

                time.sleep(3)
                out.release()
                # ws.close()

                break


if __name__ == '__main__':
    stop_threads = Stop_Thread()
    # stop_threads2 = Stop_Thread()
    get_image = Get_Image(stop_threads)
    fcn_thread = FCN_Thread(get_image, stop_threads)
    stop_threads.start()

    get_image.start()
    # stop_threads2.start()
    try:

        fcn_thread.start()

        print("RUn")

    except KeyboardInterrupt:
        print('Hello user you have pressed ctrl-c button.')
        stop_threads.threads_stopper = True
        time.sleep(1)
        print("releasing out and ws")
        out.release()
        ws.close()
        os._exit(0)