import numpy as np
import torch
import os
import FCN.sp_utils as utils
import time
import logging
import json
import cv2
import websocket
from threading import Thread
from FCN.sp_utils.config import config as config_FCN
import pandas as pd
# from multiprocessing import Process
import multiprocessing
# from Spine_navigation_Polyu.utils.config_robot_GUI import config
from Spine_navigation_Polyu.utils.functions import run_FCN_streamed_image, save_video, AverageMeter
from Spine_navigation_Polyu.utils.config_robot_GUI import config
from FCN.sp_utils.run_test_without_labels import run_test_without_labels
# https
import torch.multiprocessing as mp
from torch.multiprocessing import Process
import keyboard
from PIL import Image
import socket

message = config.IMAGE.JSON_WS
json_mylist = json.dumps(message, separators=(',', ':'))

# fourcc = cv2.VideoWriter_fourcc(*'XVID')
# out = cv2.VideoWriter(os.path.join(config.IMAGE.SAVE_PATH, 'output_original.avi'), fourcc, 3.0,
#                       (1280, 480))  # for images of size 480*640

ws = websocket.WebSocket()
# ws.connect("ws://localhost:4100")
# ws.send(json_mylist)

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.info("STARTING TEST")


class Get_Image_socketPC(Process):
    '''# we need to receive the data of length 307329'''

    def __init__(self,threads_stopper,q_im):
        Process.__init__(self)

        self.image = np.zeros(0)
        self.process_image = q_im
        self.threads_stopper = threads_stopper
        self.num = 0

    def run(self):
        # Create a TCP/IP socket
        HOST = "192.168.0.101"
        PORT = 6666
        buffer_size = 307200  # 307329 - 129

        server_address = (HOST, PORT)
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.bind(server_address)

        # Listen for incoming connections
        sock.listen()  # It specifies the number of unaccepted connections that the system will allow before refusing new connections.

        bytes_from_socket = ''
        connection, client_address = sock.accept()
        print("connection from {}".format(client_address))
        time_start = time.time()
        while True:

            try:
                bytes_from_socket = connection.recv(307200) #connection.recv()
            except:
                "No connection"
                connection, client_address = sock.accept()
                print("connection from {}".format(client_address))

            # print(len(bytes_from_socket))
            if len(bytes_from_socket) == buffer_size:
                print(len(bytes_from_socket))
                image = Image.frombuffer('L', (640, 480), bytes_from_socket)
                self.process_image.put(image)
                # print(image)
                image_np = np.array(image)
                cv2.imshow("image", image_np)
                cv2.waitKey(1)

                self.num = self.num + 1

            if bool(self.threads_stopper.value) == True:
                print("Terminate server")
                time_thread = time.time() - time_start
                fps_im = self.num/time_thread
                print("fps Server thread {}, average time per cycle {}".format(fps_im, 1/fps_im))
                sock.close()
                break


class Get_Image(Thread):
    '''# we need to receive the data of length 307329'''

    def __init__(self):
        Thread.__init__(self)

        self.image = np.zeros(0)
        self.num = 0
        self.frame_num = 0
        # self.stop_thread = thread_stopper
        # self.q_image = q_im
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
        while True:  # self.num in range (100)
            # if bool(self.stop_thread.value) ==False:
            #     # for test_im in test_list:
            #     # print("from for")
            #     self.image = Image.open(test_im)
            #     self.frame_num = self.frame_num + 1

            binAnswer = ws.recv_frame()

            if websocket.ABNF.OPCODE_MAP[binAnswer.opcode] == "binary":
                # print("image received")
                image_byte_array = (binAnswer.data)[129:]
                # Create a PIL Image from our pixel array.
                image = Image.frombuffer('L', (640, 480), image_byte_array)
                if image.size:
                    # self.q_image.put(image)
                    self.image = image
                    # self.frame_num = self.frame_num + 1
                image_display = np.array(self.image)

            else:
                print("Text output")
                num_text = num_text + 1
                if num_text == 2:
                    print("Ready to stream")

            self.num = self.num + 1
            time.sleep(0.1)
            '''To interupt the thread '''
            # global threads_stopper
            # print("receive")

        # else:
        #      # or stop_threads==True
        #     print("Get_Image thread is stopped")
        #     time_thread = time.time() - time_start
        #     fps_im = self.num/time_thread
        #     print("fps Get Image thread {}, average time per cycle {}".format(fps_im, 1/fps_im))
        #     # ws.close()
        #     break


class FCN_Thread(Process):
    '''Run FCN pre-trained model on online streamed images
    calculate the coordinate'''

    def __init__(self, threads_stopper, q_im):
        Process.__init__(self)

        self.result_im = np.zeros(0)

        self.image_flag = False
        self.num = 0

        self.threads_stopper = threads_stopper
        self.q_image = q_im

    def run(self):
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

        # to stop all threads at the same time
        probability = np.zeros(0)
        X = np.zeros(0)
        Y = np.zeros(0)
        num = 0
        patient = "US_probe"
        time_inference = AverageMeter()

        time_start = time.time()
        n = 0

        # test_dir = "D:\spine navigation Polyu 2021\DATASET_polyu\FCN_PWH_dataset_heatmaps_all"
        # for patient in ["US_probe_output"]:  # Empty_frames
        #     test_dir_patient = os.path.join(test_dir, patient, "Images")
        #     test_list = [os.path.join(test_dir_patient, item) for item in os.listdir(test_dir_patient)]

        with torch.no_grad():
            while True:

                # print("frame num of received image in FCN thread", self.get_image.frame_num)
                start_time = time.time()
                # input_data = Image.open(data)
                # print(input_data)
                print("from FCN", self.q_image.get())
                inputs,pred,pobability,X,Y,frame_probability = run_FCN_streamed_image(self.q_image.get(),model,device,probability,X,Y,logger,config)
                # self.q_image.put(input_data) #should be after image usage, otherwise it affects it
                # print("x",X)
                time_one = time.time() - start_time
                time_inference.update(time_one)

                n = n + 1
                self.num = self.num + 1

                '''To interrupt the thread'''

                if bool(self.threads_stopper.value) == True:
                    print("Allocated:", round(torch.cuda.memory_allocated(0) / 10243, 1), "GB")
                    # print('Cached: ', round(torch.cuda.memory_reserved(0) / 10243, 1))

                    print("FCN_Thread stopped")
                    if time_inference.avg != 0:
                        print("fps FCN thread {}, average time per cycle {}".format(1 / time_inference.avg,
                                                                                    time_inference.avg))

                    time_thread = time.time() - time_start
                    fps_im = self.num / time_thread
                    print("fps FCN thread {}, average time per cycle {}".format(fps_im, 1 / fps_im))
                    print("NUM FCN ", self.num, n)
                    # print(X)

                    # ws.close()
                    # os._exit(0)
                    break


class Stop_Thread(Process):
    '''# To set a flag to stop the treads. By initiating the separate thread we make sure that the keyboard
    pressed value is received immediately'''

    def __init__(self, stopper):
        Process.__init__(self)
        self.threads_stopper = False  # to stop all threads at the same time
        self.num = 0

        self.stopper = stopper

    def run(self):
        print("wait")
        time_start = time.time()
        i = 0

        while True:
            self.num = self.num + 1
            # print(i)

            if keyboard.is_pressed('q') or (self.threads_stopper == True):
                self.stopper.value = 1
                # self.q.put("KILL WORKER")
                self.threads_stopper = True
                print("Killing threads")
                time_thread = time.time() - time_start
                fps_im = self.num / time_thread
                print("Total time {},fps Stop_Thread thread {}, average time per cycle {}".format(time_thread, fps_im,
                                                                                                  1 / fps_im))

                time.sleep(3)
                # out.release()
                # ws.close()

                break


class Shadow_Image(Process):
    '''# we need to receive the data of length 307329'''

    def __init__(self, thread_stopper_bool, q_im):
        Process.__init__(self)

        self.image = np.zeros(0)
        self.num = 0
        self.frame_num = 0
        self.stop_thread = Stop_Thread
        self.get_image = q_im
        self.threads_stopper = thread_stopper_bool

    def run(self):
        fps_im = 0
        time_start = time.time()
        while True:
            if bool(self.threads_stopper.value) == False:
                image = self.get_image.get()

                print(image)
                self.num = self.num + 1
            else:
                time_thread = time.time() - time_start
                if time_thread != 0:
                    fps_im = self.num / time_thread
                print("Total time {},fps Shadow_image thread {}, average time per cycle {}".format(time_thread, fps_im,
                                                                                                   1 / fps_im))
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
    return device, model


def get_image_fun(q_im):
    num_text = 0
    while True:  # self.num in range (100)
        # if self.stop_thread.threads_stopper ==False:
        # for test_im in test_list:
        #     # print("from for")
        #     self.image = Image.open(test_im)
        #     self.frame_num = self.frame_num + 1

        binAnswer = ws.recv_frame()

        if websocket.ABNF.OPCODE_MAP[binAnswer.opcode] == "binary":
            # print("image received")
            image_byte_array = (binAnswer.data)[129:]
            # Create a PIL Image from our pixel array.
            image = Image.frombuffer('L', (640, 480), image_byte_array)
            # print("received")
            if image.size:
                q_im.put(image)
                image = image
                # self.frame_num = self.frame_num + 1
            image_display = np.array(image)

        else:
            print("Text output")
            num_text = num_text + 1
            if num_text == 2:
                print("Ready to stream")


if __name__ == '__main__':

    q_im = mp.Queue()
    thread_stopper_bool = mp.Value('i', 0)  # False
    server_init = Get_Image_socketPC(thread_stopper_bool,q_im)
    stop_thread = Stop_Thread(thread_stopper_bool)
    FCN = FCN_Thread(thread_stopper_bool, q_im)

    server_init.daemon = True
    stop_thread.daemon = True
    FCN.daemon = True

    server_init.start()
    stop_thread.start()
    FCN.start()
    try:
        # print("PID of Main process is: {}".format(multiprocessing.current_process().pid))

        for i in range (400):
            time.sleep(0.1)
            if bool(thread_stopper_bool.value)== True:
                time.sleep(0.5)
                server_init.terminate()
                stop_thread.terminate()
                FCN.terminate()

        server_init.terminate()
        stop_thread.terminate()
        FCN.terminate()
        time.sleep(1)
        if not stop_thread.is_alive():
            print("[MAIN]: WORKER is a goner")
            FCN.join(timeout=0.1)
            stop_thread.join(timeout=0.1)
            server_init.join(timeout=0.1)

            q_im.close()

            print("releasing out and ws")
            # out.release()
            ws.close()
            os._exit(0)

    except KeyboardInterrupt:
        # sock.close()
        os._exit(0)




    # mp.set_start_method("spawn")
    # q = mp.Queue()
    # q_im = mp.Queue()
    # thread_stopper_bool = mp.Value('i', 0)  # False
    # # device, model = initialization()
    # # stop_thread = Stop_Thread(thread_stopper_bool, q)
    # # get_image = Get_Image()
    # FCN = FCN_Thread(thread_stopper_bool, q_im)
    # shadow = Shadow_Image(thread_stopper_bool, q_im)
    #
    # # print(thread_stopper_bool.value)
    # shadow.daemon = True
    # # get_image.daemon = True
    # FCN.daemon = True
    # # stop_thread.daemon = True
    # # FCN.start()
    # # stop_thread.start()
    # shadow.start()
    # # shadow.start()
    # # get_image.start()
    #
    # FCN.start()
    # get_image_fun(q_im)

    #
    # # stop_threads2.start()
    # try:
    #     print("RUn")
    #     for i in range(200):
    #         # print(get_image.image)
    #         # q_im.put(get_image.image)
    #         time.sleep(0.1)
    #
    #     else:
    #         os._exit(0)

    # msg = stop_thread.q.get()
    # if msg == "KILL WORKER":
    #     print("[MAIN]: Terminating slacking WORKER")
    #
    #     # thread_stopper_bool.value = 1
    #     time.sleep(0.05)
    #     # FCN.terminate()
    #     # stop_thread.terminate()
    #     shadow.terminate()
    #     # get_image.terminate()
    #
    #     time.sleep(0.2)
    #     if not stop_thread.is_alive():
    #         print("[MAIN]: WORKER is a goner")
    #         FCN.join(timeout=0.1)
    #         stop_thread.join(timeout=0.1)
    #         shadow.join(timeout=0.1)
    #         get_image.join(timeout=0.1)
    #
    #         print("[MAIN]: Joined WORKER successfully!")
    #         q.close()
    #         q_im.close()
    #         # thread_stopper_bool.close()
    #         print("releasing out and ws")
    #         os._exit(0)

    # except KeyboardInterrupt:
    #     print('Hello user you have pressed ctrl-c button.')
    #     # stop_threads.threads_stopper = True
    #     time.sleep(1)
    #     print("releasing out and ws")
    #     # out.release()
    #     # ws.close()
    #     os._exit(0)