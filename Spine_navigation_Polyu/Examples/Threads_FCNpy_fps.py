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
import socket
from multiprocessing import Process
import multiprocessing
from Spine_navigation_Polyu.utils.config_robot_GUI import config
from Spine_navigation_Polyu.utils.functions import run_FCN_streamed_image,save_video, AverageMeter
#https://techtutorialsx.com/2018/11/08/python-websocket-client-sending-binary-content/
import pprofile

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

prof = pprofile.Profile()

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
    def __init__(self,Get_Image,Stop_Thread,device,model):
        Thread.__init__(self)

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

                    if config.IMAGE.VIDEO == True:
                        self.result_im = save_video(out, inputs, pred, frame_probability, patient,config, target=None, labels=None)
                    #TODO: accumulate array of robot positions and force values to write it to npz file

                    n = n+1
                    self.num = self.num + 1
                    time_one = time.time() - start_time
                    time_inference.update(time_one)
                else:
                    self.image_flag = False

                '''To interrupt the thread'''

                if self.stop_thread.threads_stopper==True:
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


            if keyboard.read_key() == "q" or (self.threads_stopper == True): #keyboard.is_pressed('q')
                self.threads_stopper = True
                print("Killing threads")
                time_thread = time.time() - time_start
                fps_im = self.num / time_thread
                print("Total time {},fps Stop_Thread thread {}, average time per cycle {}".format(time_thread,fps_im, 1 / fps_im))

                time.sleep(3)
                out.release()
                # ws.close()

                break

class Shadow_Image(Thread):
    '''# we need to receive the data of length 307329'''
    def __init__(self,Stop_Thread, Get_Image):
        Thread.__init__(self)

        self.image = np.zeros(0)
        self.num = 0
        self.frame_num = 0
        self.stop_thread = Stop_Thread
        self.get_image = Get_Image
        # self.threads_stopper = self.stop_thread.threads_stopper

    def run(self):
        time_start = time.time()
        while True:
            if self.stop_thread.threads_stopper == False:
                # print(self.get_image.image)
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


class Force_Thread(Thread):
    '''Force sensor has different coordinate system, it reads measurements in tool coordintate system,
        so that when it touches the surface perpendicular to the tool, it reads the Fz

        To get exact forces in the robot coordinate system, it is better to multiply Forces from the Force sensor with
        the transformation matrix from the robot, Universal robot can provide it at each moment of time.

        To symplify the task just the proportion between the Fz and Fset is used to generate the velocity
        for the robot control. ?Velocity should be set also in the tool frame? Otherwise the axis of the robot
        and the force sensor should be aligned during robot manipulation

        Now is (check sensor and base alignment):
        x robot - z force sensor
        y robot - x force sensor
        z robot - y force sensor

        #TODO: To change it the transformation matrix should be introduced.
        '''
    def __init__(self,Stop_Thread):
        Thread.__init__(self)
        self.Fz = 0
        self.Fdelta = 0
        self.Rx_next = 0
        self.Mx_raw = 0
        # self.point_x_next = 0
        # self.force_tread_stopper = False
        # self.stop_movement = False
        self.num = 0
        # self.q_Force = q_Force
        # self.q_Full_Force = q_Full_Force
        self.stop_thread = Stop_Thread
    def run(self):
        i = 0
        # if config.MODE.Develop == False:
        #     self.robot = urx.Robot(config.IP_ADRESS, use_rt=True)
        # else:
        #     self.robot = None

        print("Force Process starts")
        # print("robot Force", self.robot)

        if config.MODE.Develop == False:
            # robot = urx.Robot(config.IP_ADRESS, use_rt=True)
            s = socket.socket(
                socket.AF_INET, socket.SOCK_STREAM)

            s.connect((config.IP_ADRESS, 63351))
            # logger_robot.info('Socket is connected')
        else:
            # logger_robot.info('Ethernet is not connected')
            s = None
            # os._exit(0)

        F = 0
        time_start = time.time()
        while True:
            if self.stop_thread.threads_stopper == False:


                # if i == 150:
                #    Fref = -6
                #
                # if i == 300:
                #    Fref = -3

                # if i == 1000:
                #     self.stop_movement = True


                # tstart = time.time()
                # response = s.recv(4096)
                response = s.recv(136)#1024 #512
                print(response)
                # print("socket response",response)
                # response = bytearray(response)
                try:
                    val = response.decode("ascii").split('(', 1)[1].split(')')
                    print(len(val))
                    val = val[0]
                except:
                    print("Not full package.")


                array = [float(x) for x in val[0:-1].split(',')]

                print ('Fx:', array[0], 'Fy:', array[1],'Fz:', array[2],'Mx:', array[3], 'My:', array[4], 'Mz:', array[5])
                #print ('      ')
                self.F_full = array
                self.F_changed_order = [array[2], array[0], array[1],array[5],array[3], array[4]]
                self.Fz = array[2] # robot X axis, contr directed
                self.Fy = array[1] # robot Z axis
                self.Fx = array[0] # robot Y axis
                self.Mx_raw = array[3]
                self.My = array[4]
                self.Mz = array[5]
                #print array[3],array[4], array[5]
                # self.Fdelta = config.FORCE.Fref - math.fabs(self.Fz)

                if self.Fz > config.FORCE.Fmax:
                    self.Fz_sat = config.FORCE.Fmax
                elif self.Fz < -config.FORCE.Fmax:
                    self.Fz_sat = -config.FORCE.Fmax
                else:
                    self.Fz_sat = self.Fz
                # print("Fz_sat", self.Fz_sat)

                #SEND to other threads
                # self.q_Force.value = self.Fz_sat
                # self.q_Full_Force[:] = self.F_full
                # posa = self.robot.getl()
                # point_x = posa[0]
                # Rx_current = posa[3]
                #print "X", point_x
                ######## F - total force

                #F = ((self.Fx) ** 2 + (self.Fy) ** 2 + (self.Fz) ** 2) ** 1 / 2





                ################ velocity control #################

                #TODO: move it to Move_Thread
                # self.point_x_next = point_x + config.FORCE.K_delta * (math.fabs(self.Fz_sat) - config.FORCE.Fref)

                # print(self.velocity_x )
                #self.velocity_x = Kf * (F - Fref)
                #print 'F:', F

                ####################################################
                if self.Mx_raw < -0.8:
                    self.Mx = -0.8
                elif self.Mx_raw > 0.8:
                    self.Mx = 0.8
                else:
                    self.Mx = self.Mx_raw

                Krx = 0.03
                self.velocity_Rx = -Krx*(self.Mx_raw)


                # if self.Mx < -0.2:
                #     self.Rx_next = Rx_current - 0.1*(math.fabs(self.Mx) - 0.1)
                #     R1 = 0.1 * (math.fabs(self.Mx) - 0.1)
                #     #print('change: - ', R1)
                # elif self.Mx > 0.2:
                #     self.Rx_next = Rx_current + 0.1*(math.fabs(self.Mx) - 0.1)
                #     R2 = 0.1 * (math.fabs(self.Mx) - 0.1)
                #     #print('change: + ', R2)

                # else:
                #     self.Rx_next = Rx_current

                #print "X+", self.point_x_next
                # tstop = time.time()
                #print ((tstop - tstart))
                #TODO: init robot for force function
                if self.Fz > config.FORCE.Fcrit:
                    # self.robot.stop()
                    self.stop_thread.threads_stopper = 1 #stop threads
                    print('Value is more than F critical. Throw threads_stopper flag')

                i = i + 1
                # print(self.num)

                self.num = self.num +1

            else:
                print("Force thread is stopped")
                time_thread = time.time() - time_start
                fps_im = self.num / time_thread
                print("fps Force thread {}, average time per cycle {}".format(fps_im, 1 / fps_im))
                break

                    #print 'i:', i




if __name__ == '__main__':

    device,model = initialization()

    stop_threads = Stop_Thread()
    # stop_threads2 = Stop_Thread()
    get_image = Get_Image(stop_threads)
    fcn_thread = FCN_Thread(get_image, stop_threads,device,model)
    shadow = Shadow_Image(stop_threads,get_image)
    force = Force_Thread(stop_threads)

    stop_threads.start()
    get_image.start()

    force.start()

    # shadow.start()
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