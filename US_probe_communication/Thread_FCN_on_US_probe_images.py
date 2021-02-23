import json
import websocket
from PIL import  Image
import numpy as np
import cv2
import torch
import torchvision
import FCN.sp_utils as utils
import logging
import keyboard
import os
import sys
import time
from threading import Thread
from US_probe_communication.config_US_probe_communtication import config
from FCN.sp_utils.run_test_without_labels import run_FCN_streamed_image,save_video
#https://techtutorialsx.com/2018/11/08/python-websocket-client-sending-binary-content/

message = config.TEST.JSON_WS
json_mylist = json.dumps(message, separators=(',', ':'))

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output_original.avi', fourcc, 3.0, (1280, 480))  # for images of size 480*640

ws = websocket.WebSocket()
ws.connect("ws://localhost:4100")
ws.send(json_mylist)

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.info("STARTING TEST")



class FCN_Thread(Thread):

    def __init__(self):
        Thread.__init__(self)

        self.result_im = np.zeros(0)
        self.image = np.zeros(0)
        self.image_flag = False

    def run(self):
        global stop_threads
        while True:
            binAnswer = ws.recv_frame()
            if websocket.ABNF.OPCODE_MAP[binAnswer.opcode] == "binary":
                # print("bytes: ",bytearray(binAnswer.data).__len__())
                # we need to receive the data of length 307329
                # image_byte_array = bytearray(binAnswer.data)[129:]
                image_byte_array = (binAnswer.data)[129:]
        # Create a PIL Image from our pixel array.
                self.image = Image.frombuffer('L',(640, 480),image_byte_array)

                image_display = np.array(self.image)
                # cv2.imwrite("image.png",image_display)
                # cv2.imshow("image", image_display)

                # cv2.waitKey(1)

            else:
                print("No image received")

    def FCN_run(self):
        global stop_threads
        probability = np.zeros(0)
        X = np.zeros(0)
        Y = np.zeros(0)
        patient = "US_probe"
        time_inference = utils.AverageMeter()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if config.TEST.Windows == True:
            model = utils.model_pose_resnet.get_pose_net(config.TEST.Windows_MODEL_FILE, is_train=False)
            logger.info('=> loading model from {}'.format(config.TEST.Windows_MODEL_FILE))
            model.load_state_dict(
                torch.load(config.TEST.Windows_MODEL_FILE, map_location=torch.device('cpu'))['model_state_dict'])
        else:
            print("Model is not defined")
            model = []

        with torch.no_grad():
            while True:

                if self.image.size: #image we get from WebSocket connection
                    # print("image received")
                    self.image_flag = True
                    start_time = time.time()
                    inputs,pred,probability, X, Y, frame_probability = run_FCN_streamed_image(self.image,model, device, probability,X,Y,logger)

                    if config.TEST.VIDEO == True:
                        self.result_im = save_video(out, inputs, pred, frame_probability, patient, target=None, labels=None)

                    if keyboard.is_pressed('c') or stop_threads==True:
                        print("avg time for one cycle",time_inference.avg)
                        out.release()
                        ws.close()
                        os._exit(0)
                        break

                    time_one = time.time() - start_time
                    time_inference.update(time_one)
                else:
                    self.image_flag = False


    def stop_thread(self):
        global stop_threads  # to stop all threads at the same time
        while True:
            if keyboard.is_pressed('q'):
                stop_threads = True  #global value
                print("Killing threads")
                break

if __name__ == '__main__':
    try:
        #TODO: If using more than 2 threads, the execution slows down significantly.
        #TODO: It is better to use multiprocessing package if more than 2
        stop_threads = False
        thread_im = FCN_Thread()
        thread_im.start()
        x = Thread(target=thread_im.FCN_run)
        x.start()
        print("RUN")
        # thread_im.FCN_run()

    except KeyboardInterrupt:
        print('Hello user you have pressed ctrl-c button.')
        out.release()
        ws.close()