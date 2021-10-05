import numpy as np
import torch
import os
import Spine_navigation_Polyu.utils as utils
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
from Spine_navigation_Polyu.utils.functions import run_FCN_streamed_image,save_video, AverageMeter
from Spine_navigation_Polyu.utils.config_robot_GUI import config
from FCN.sp_utils.run_test_without_labels import run_test_without_labels
#https
import torch.multiprocessing as mp
from torch.multiprocessing import Process
import keyboard
from PIL import Image
import ctypes



logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.info("STARTING TEST")


class FCN_Thread(Process):
    '''Run FCN pre-trained model on online streamed images
    calculate the coordinate'''
    def __init__(self,q_im_raw,threads_stopper,q_im_inputs,q_im_pred,q_frame_probability,v_num,start_send_image,q_im_pred_sacrum,q_frame_probability_sacrum,
                                       q_frame_class):
        Process.__init__(self)

        self.result_im = np.zeros(0)

        self.image_flag = False
        self.num = 0
        self.q_im_pred_sacrum = q_im_pred_sacrum
        self.q_frame_probability_sacrum = q_frame_probability_sacrum
        self.q_frame_class = q_frame_class
        self.q_im_raw = q_im_raw
        self.threads_stopper = threads_stopper
        # self.q_image = q_im
        self.q_im_inputs = q_im_inputs
        self.q_im_pred = q_im_pred
        self.q_frame_probability = q_frame_probability
        # self.q_num = q_num
        self.v_num = v_num
        self.start_send_image = start_send_image


    def initialization(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # device = "cpu"
        print(self.device)
        # a = torch.cuda.FloatTensor(10000)
        # print("Allocated:", round(torch.cuda.memory_allocated(0) / 10243, 1), "GB")
        print(torch.cuda.get_device_name(0))

        if config.TRAIN.three_heads == True or config.TRAIN.two_heads == True:
            if config.MODE.subject_mode == "phantom":
                config.IMAGE.Windows_MODEL_FILE = config.IMAGE.Windows_MODEL_FILE_MULTITASK_PHANTOM

            self.model = utils.model_pose_resnet_two_heads.get_pose_net(config,config.IMAGE.Windows_MODEL_FILE, is_train=False)
            logger.info('=> loading model from {}'.format(config.IMAGE.Windows_MODEL_FILE))
            print('=> loading model from {}'.format(config.IMAGE.Windows_MODEL_FILE))
            self.model.load_state_dict(
                torch.load(config.IMAGE.Windows_MODEL_FILE, map_location=self.device)[
                    'model_state_dict'])  # map_location=torch.device('cpu')

            self.model.eval()  # Super important for testing! Otherwise the result would be random

            self.model.to(self.device)
            print("Model on cuda: ", next(self.model.parameters()).is_cuda)
            # logger.info("Setting model to eval. It is important for testing")

            if config.TRAIN.two_heads_for_class_only == True:
                config.IMAGE.Windows_MODEL_FILE=config.IMAGE.Windows_MODEL_FILE_best_FCN
                self.model_FCN = utils.model_pose_resnet.get_pose_net(config.IMAGE.Windows_MODEL_FILE, is_train=False)
                logger.info('=> loading model from {}'.format(config.IMAGE.Windows_MODEL_FILE))
                print('=> loading model from {}'.format(config.IMAGE.Windows_MODEL_FILE))
                self.model_FCN.load_state_dict(
                    torch.load(config.IMAGE.Windows_MODEL_FILE, map_location=self.device)[
                        'model_state_dict'])  # map_location=torch.device('cpu')

                self.model_FCN.eval()  # Super important for testing! Otherwise the result would be random

                self.model_FCN.to(self.device)
                print("Model on cuda: ", next(self.model_FCN.parameters()).is_cuda)

                return self.device, self.model, self.model_FCN
        else:
            if config.MODE.subject_mode == "phantom":
                config.IMAGE.Windows_MODEL_FILE = config.IMAGE.Windows_MODEL_FILE_PHANTOM

            self.model = utils.model_pose_resnet.get_pose_net(config.IMAGE.Windows_MODEL_FILE, is_train=False)
            logger.info('=> loading model from {}'.format(config.IMAGE.Windows_MODEL_FILE))
            print('=> loading model from {}'.format(config.IMAGE.Windows_MODEL_FILE))
            self.model.load_state_dict(
                torch.load(config.IMAGE.Windows_MODEL_FILE, map_location=self.device)[
                    'model_state_dict'])  # map_location=torch.device('cpu')

            self.model.eval()  # Super important for testing! Otherwise the result would be random

            self.model.to(self.device)
            print("Model on cuda: ", next(self.model.parameters()).is_cuda)
            # logger.info("Setting model to eval

        # fourcc = cv2.VideoWriter_fourcc(*'XVID')  # "DIVX" 'XVID'
        # print(fourcc)  # fourcc tag 0x44495658/'XVID' codec_id 000C
        # self.out = cv2.VideoWriter(os.path.join(config.IMAGE.SAVE_PATH, 'output_original.avi'), fourcc, 5.0,
        #                       (1280, 480))  # for images of size 480*640

        return self.device, self.model


    def run(self):

        if config.TRAIN.two_heads_for_class_only == True:
            self.device, self.model, self.model_FCN = self.initialization()
        else:
            self.device, self.model = self.initialization()

          # to stop all threads at the same time
        probability = np.zeros(0)
        X = np.zeros(0)
        Y = np.zeros(0)
        num= 0
        patient = "US_probe"
        # time_inference = AverageMeter()

        time_start = time.time()
        n = 0

        with torch.no_grad():
            while True:

                # start_time = time.time()
                # input_data = Image.open(data)
                # print(input_data)
                # print("from FCN",self.q_image.get())
                # image_from_probe = self.q_image.get()
                arr = (np.frombuffer(self.q_im_raw.get_obj())).astype('uint8')
                # make it two-dimensional
                # image_from_probe = arr.reshape((480, 640, -1))
                image_from_probe = Image.frombuffer("L",(640,480),arr)

                if config.TRAIN.three_heads == True:
                    inputs, pred, pobability, X, Y, frame_probability,pred_sacrum,frame_probability_sacrum,X_sac,Y_sac,classification = run_FCN_streamed_image(image_from_probe,
                                                                                               self.model, self.device,
                                                                                               probability, X, Y,
                                                                         logger, config)
                elif config.TRAIN.two_heads == True and config.TRAIN.two_heads_for_class_only == False:
                    inputs, pred, pobability, X, Y, frame_probability, classification = run_FCN_streamed_image(
                        image_from_probe,
                        self.model, self.device,
                        probability, X, Y,logger, config)

                elif config.TRAIN.two_heads == True and config.TRAIN.two_heads_for_class_only == True:
                    inputs, pred, pobability, X, Y, frame_probability, classification = run_FCN_streamed_image(
                        image_from_probe,
                        self.model, self.device,
                        probability, X, Y, logger, config,model2=self.model_FCN)

                else:
                    inputs,pred,pobability,X,Y,frame_probability = run_FCN_streamed_image(image_from_probe,self.model,self.device,probability,X,Y,logger,config)
                # self.q_image.put(input_data) #should be after image usage, otherwise it affects it
                cv2.imshow("Raw",np.array(image_from_probe))
                cv2.waitKey(1)

                if bool(self.start_send_image.value)== True:
                    # print((np.array(inputs).reshape(-1)).size)

                    self.q_im_inputs[:] = np.array(inputs).flatten()

                    # self.q_num.put(self.num)
                    self.q_im_pred[:] = pred[0][0]

                    self.q_frame_probability.value = frame_probability

                    if config.TRAIN.three_heads == True:
                        self.q_im_pred_sacrum[:] = pred_sacrum[0][0]

                        self.q_frame_probability_sacrum.value = frame_probability_sacrum
                        # print(classification)
                        self.q_frame_class[:] = classification[0]

                    elif config.TRAIN.two_heads == True:
                        self.q_frame_class[:] = classification[0]
                        self.q_im_pred_sacrum[:] = pred[0][0]

                        self.q_frame_probability_sacrum.value = 0
                        # print(classification)
                    else:
                        self.q_frame_class[:] = [0,0,0]
                        self.q_im_pred_sacrum[:] = pred[0][0]

                        self.q_frame_probability_sacrum.value = 0
                    # self.v_num.value = self.num
                    # print("numbers FCN: ", self.num )

                # time_one = time.time() - start_time
                # time_inference.update(time_one)

                # if config.IMAGE.VIDEO == True:
                #     self.result_im = save_video(self.out, inputs, pred, frame_probability, patient, config, target=None,
                #                                 labels=None)

                n = n+1
                self.num = self.num + 1


                '''To interrupt the thread'''

                if bool(self.threads_stopper.value) == True:
                    print("Allocated:", round(torch.cuda.memory_allocated(0) / 10243, 1),"GB")
                    # print('Cached: ', round(torch.cuda.memory_reserved(0) / 10243, 1))

                    print("FCN_Thread stopped")
                    # if time_inference.avg !=0:
                    #     print("fps FCN thread {}, average time per cycle {}".format(1/time_inference.avg, time_inference.avg))

                    time_thread = time.time() - time_start
                    fps_im = self.num / time_thread
                    print("fps FCN thread {}, average time per cycle {}".format(fps_im, 1 / fps_im))
                    # print("NUM FCN ", self.num, n)
                    if config.IMAGE.SAVE_NPZ_FILE:
                        while os.path.exists("FCNthread_output%s.npz" % num):
                            num = num+ 1
                        np.savez(os.path.join(config.IMAGE.SAVE_PATH, "FCNthread_output%s.npz" % num),
                                 probability=probability, x=X, y=Y)
                        print(len(probability),len(X),len(Y))
                        pd_frame = pd.DataFrame({"frame_probability":frame_probability, "X":X, "Y":Y})
                        # pd_frame.to_csv(os.path.join(config.IMAGE.SAVE_PATH, "FCNthread_output%s.csv" % num))
                    # self.out.release()
                    # print("out released")



                    # print(X)

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

                # time.sleep(3)
                # out.release()
                # ws.close()

                break

class Shadow_Image(Process):
    '''# we need to receive the data of length 307329'''
    def __init__(self,thread_stopper_bool,q_im):
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
        print("RUN SHADOW")
        while True:
            if bool(self.threads_stopper.value) == False:
                image = self.get_image.get()

                print(image)
                self.num = self.num + 1
            else:
                # out.release()
                time_thread = time.time() - time_start
                if time_thread != 0:
                    fps_im = self.num / time_thread
                print("Total time {},fps Shadow_image thread {}, average time per cycle {}".format(time_thread,fps_im, 1 / fps_im))
                break



def get_image_fun(q_im_raw,threads_stopper):
    message = config.IMAGE.JSON_WS
    json_mylist = json.dumps(message, separators=(',', ':'))
    ws = websocket.WebSocket()
    ws.connect("ws://localhost:4100")
    ws.send(json_mylist)
    num_text = 0
    start_time = time.time()
    num = 0



    while True:  # self.num in range (100)
        # if self.stop_thread.threads_stopper ==False:
        # for test_im in test_list:
        #     # print("from for")
        #     self.image = Image.open(test_im)
        #     self.frame_num = self.frame_num + 1

        binAnswer = ws.recv_frame()
        # print("ws",ws)
#ws <websocket.WebSocket object at 0x000002807685C4A8>
        if websocket.ABNF.OPCODE_MAP[binAnswer.opcode] == "text":
            print("text received")

        if websocket.ABNF.OPCODE_MAP[binAnswer.opcode] == "binary":
            # print("image received")
            image_byte_array = (binAnswer.data)[129:]
            # Create a PIL Image from our pixel array.
            image = Image.frombuffer('L', (640, 480), image_byte_array)
            image_display = np.array(image)
            # print("received")
            if image.size:
                # q_im.put(image)



                q_im_raw[:] = np.array(image).flatten()
                # print(image.size) #(640,480)
                if config.MODE.FCN == False:
                    cv2.imshow("Raw", np.array(image_display))
                    cv2.waitKey(1)

                # image = image
                # self.frame_num = self.frame_num + 1
            image_display = np.array(image)

        else:
            print("Text output")
            num_text = num_text + 1
            if num_text == 2:
                print("Ready to stream")
        num = num+1
        if bool(threads_stopper.value) == True:
            # out.release()
            time_thread = time.time() - start_time
            print("Get image stopped")
            fps_im = num / time_thread
            print("Total time {},fps Get Image thread {}, average time per cycle {}".format(time_thread, fps_im,
                                                                                              1 / fps_im))
            break


if __name__ == '__main__':



    # mp.set_start_method("spawn")
    q = mp.Queue()
    q_im = mp.Array(ctypes.c_double, 640*480)
    thread_stopper_bool = mp.Value('i', 0) #False
    # device, model = initialization()
    stop_thread = Stop_Thread(thread_stopper_bool, q)
    # get_image = Get_Image()
    # q_im = 0
    # device, model = initialization()

    # FCN = FCN_Thread(q_im,thread_stopper_bool)
    shadow = Shadow_Image(thread_stopper_bool,q_im)


    # print(thread_stopper_bool.value)
    # shadow.daemon = True

    # get_image.daemon = True
    # FCN.daemon = True
    stop_thread.daemon = True
    # FCN.start()
    stop_thread.start()

    # shadow.start()

    p = Process(target=get_image_fun,args=(q_im,thread_stopper_bool,))
    # p2 = Process(target=shadow.run)
    p.daemon = True
    p.start()
    # p.join()
    # p2.start()

    print("RUN")
    try:
        while True:
            msg = stop_thread.q.get()
            if msg == "KILL WORKER":
                print("[MAIN]: Terminating slacking WORKER")

                thread_stopper_bool.value = 1
                time.sleep(0.05)
                # FCN.terminate()
                stop_thread.terminate()

                p.terminate()
                # p2.terminate()

                time.sleep(1)
                if not stop_thread.is_alive():
                    print("[MAIN]: WORKER is a goner")
                    # FCN.join(timeout=0.1)
                    stop_thread.join(timeout=0.1)

                    print("[MAIN]: Joined WORKER successfully!")
                    q.close()
                    q_im.close()
                    # thread_stopper_bool.close()
                    # print("releasing out and ws")
                    # out.release()




                    # ws.close()
                    # os._exit(0)
                    break

    except KeyboardInterrupt:
        print('Hello user you have pressed ctrl-c button.')
        # stop_threads.threads_stopper = True
        time.sleep(1)
        print("releasing out and ws")
        # out.release()
        # ws.close()
        os._exit(0)