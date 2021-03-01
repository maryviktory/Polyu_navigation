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







class Get_Image(Process):
    '''# we need to receive the data of length 307329'''
    def __init__(self,Stop_Thread,q_im):
        Process.__init__(self)

        self.image = np.zeros(0)
        self.num = 0
        self.frame_num = 0
        self.stop_thread = Stop_Thread
        self.q_image = q_im
        # self.threads_stopper = self.stop_thread.threads_stopper

    def run(self):
        # test_dir = "D:\spine navigation Polyu 2021\DATASET_polyu\FCN_PWH_dataset_heatmaps_all"
        # for patient in ["US_probe_output"]:  # Empty_frames
        #     test_dir_patient = os.path.join(test_dir, patient, "Images")
        #     test_list = [os.path.join(test_dir_patient, item) for item in os.listdir(test_dir_patient)]
        #
        # # to stop all threads at the same time
        time_start = time.time()

        num_text = 0
        while True: #self.num in range (100)
            if self.stop_thread.threads_stopper ==False:
                # for test_im in test_list:
                #     # print("from for")
                #     self.image = Image.open(test_im)
                #     self.frame_num = self.frame_num + 1

                binAnswer = ws.recv_frame()

                if websocket.ABNF.OPCODE_MAP[binAnswer.opcode] == "binary":
                    # print("image received")
                    image_byte_array = (binAnswer.data)[129:]
                    # Create a PIL Image from our pixel array.
                    image = Image.frombuffer('L',(640, 480),image_byte_array)
                    if image.size:
                        self.q_image.put(image)
                        # self.image = image
                        # self.frame_num = self.frame_num + 1
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

class FCN_Thread(Process):
    '''Run FCN pre-trained model on online streamed images
    calculate the coordinate'''
    def __init__(self,Get_Image,Stop_Thread,device,model):
        Process.__init__(self)

        self.result_im = np.zeros(0)
        self.get_image = Get_Image
        self.image_flag = False
        self.num = 0
        self.stop_thread = Stop_Thread
        self.threads_stopper = self.stop_thread.threads_stopper
        self.model = model
        self.device = device
    def run(self):

          # to stop all threads at the same time
        probability = np.zeros(0)
        X = np.zeros(0)
        Y = np.zeros(0)
        num= 0
        patient = "US_probe"
        time_inference = AverageMeter()

        time_start = time.time()
        n = 0
        while True:
            with torch.no_grad():

               # print("frame num of received image in FCN thread", self.get_image.frame_num)
                if self.get_image.image.size: #image we get from WebSocket connection
                    # print("image received")
                    self.image_flag = True
                    # print("frame num of received image in FCN thread", self.get_image.frame_num)
                    start_time = time.time()

                    inputs,pred,probability, X, Y, frame_probability = run_FCN_streamed_image(self.get_image.image,self.model, self.device, probability,X,Y,logger,config)
                    # print("x",X)
                    time_one = time.time() - start_time
                    time_inference.update(time_one)
                    if config.IMAGE.VIDEO == True:
                        self.result_im = save_video(out, inputs, pred, frame_probability, patient,config, target=None, labels=None)
                    #TODO: accumulate array of robot positions and force values to write it to npz file

                    n = n+1
                    self.num = self.num + 1
                else:
                    self.image_flag = False

                '''To interrupt the thread'''

                if bool(self.stop_thread.stopper.value) == True:
                    print("Allocated:", round(torch.cuda.memory_allocated(0) / 10243, 1),"GB")
                    # print('Cached: ', round(torch.cuda.memory_reserved(0) / 10243, 1))

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


class Stop_Thread(Process):
    '''# To set a flag to stop the treads. By initiating the separate thread we make sure that the keyboard
    pressed value is received immediately'''

    def __init__(self,stopper,q):
        Process.__init__(self)
        self.threads_stopper = False # to stop all threads at the same time
        self.num = 0
        self.q = q
        self.stopper = stopper

    def run(self):
        print("wait")
        time_start = time.time()
        i = 0

        while True:
            self.num = self.num +1
            # print(i)

            if keyboard.is_pressed('q') or (self.threads_stopper == True):
                self.stopper.value = 1
                self.q.put("KILL WORKER")
                self.threads_stopper = True
                print("Killing threads")
                time_thread = time.time() - time_start
                fps_im = self.num / time_thread
                print("Total time {},fps Stop_Thread thread {}, average time per cycle {}".format(time_thread,fps_im, 1 / fps_im))

                time.sleep(3)
                # out.release()
                # ws.close()

                break

class Shadow_Image(Process):
    '''# we need to receive the data of length 307329'''
    def __init__(self,Stop_Thread, Get_Image):
        Process.__init__(self)

        self.image = np.zeros(0)
        self.num = 0
        self.frame_num = 0
        self.stop_thread = Stop_Thread
        self.get_image = Get_Image
        # self.threads_stopper = self.stop_thread.threads_stopper

    def run(self):
        time_start = time.time()
        while True:
            if bool(self.stop_thread.stopper.value) == False:
                print(self.get_image.q_image.get())
                self.num = self.num + 1
            else:
                time_thread = time.time() - time_start
                fps_im = self.num / time_thread
                print("Total time {},fps Shadow_image thread {}, average time per cycle {}".format(time_thread,fps_im, 1 / fps_im))
                break


def initialization():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = "cpu"
    print(device)
    # a = torch.cuda.FloatTensor(10000)
    # print("Allocated:", round(torch.cuda.memory_allocated(0) / 10243, 1), "GB")
    print(torch.cuda.get_device_name(0))

    if config.IMAGE.Windows == True:
        model = utils.model_pose_resnet.get_pose_net(config.IMAGE.Windows_MODEL_FILE, is_train=False)
        logger.info('=> loading model from {}'.format(config.IMAGE.Windows_MODEL_FILE))
        model.load_state_dict(
            torch.load(config.IMAGE.Windows_MODEL_FILE, map_location=device)[
                'model_state_dict'])  # map_location=torch.device('cpu')
        model.eval()  # Super important for testing! Otherwise the result would be random

        model.to(device)
        print("Model on cuda: ", next(model.parameters()).is_cuda)
        # logger.info("Setting model to eval. It is important for testing")
    else:
        print("Model is not defined")
        model = []
    return device,model

if __name__ == '__main__':

    q = multiprocessing.Queue()
    q_im = multiprocessing.Queue()
    thread_stopper_bool = multiprocessing.Value('i', 0) #False
    device, model = initialization()

    stop_thread = Stop_Thread(thread_stopper_bool,q)
    # stop_threads2 = Stop_Thread()
    get_image = Get_Image(stop_thread,q_im)
    # fcn_thread = FCN_Thread(get_image, stop_threads, device, model)
    shadow = Shadow_Image(stop_thread, get_image)

    stop_thread.daemon = True
    get_image.daemon = True
    shadow.daemon = True

    stop_thread.start()
    get_image.start()
    shadow.start()
    # fcn_thread.start()
    # stop_threads2.start()
    try:
        print("RUn")
        while True:
            msg = stop_thread.q.get()
            if msg == "KILL WORKER":
                print("[MAIN]: Terminating slacking WORKER")
                stop_thread.terminate()
                get_image.terminate()
                shadow.terminate()
                time.sleep(0.1)
                if not stop_thread.is_alive():
                    print("[MAIN]: WORKER is a goner")
                    stop_thread.join(timeout=1.0)
                    get_image.join(timeout=1.0)
                    shadow.join(timeout=1.0)
                    print("[MAIN]: Joined WORKER successfully!")
                    q.close()
                    q_im.close()
                    thread_stopper_bool.close()
                    print("releasing out and ws")
                    out.release()
                    ws.close()
                    break  # watchdog process daemon gets terminated

    except KeyboardInterrupt:
        print('Hello user you have pressed ctrl-c button.')
        # stop_threads.threads_stopper = True
        time.sleep(1)
        print("releasing out and ws")
        out.release()
        ws.close()
        os._exit(0)