import json
import websocket
from PIL import  Image
import numpy as np
import cv2
import keyboard
import os
from threading import Thread
from multiprocessing import Process
import socket
import time
#https://techtutorialsx.com/2018/11/08/python-websocket-client-sending-binary-content/
message = {
    "Command": "Us_Config",
    "US_module": 2,  # "US_DEVICE_UVF = 1", "US_DEVICE_PALM = 2", "US_DEVICE_TERASON = 3"
    "Posture_module": 1,  # POSTURE_SENSOR_UVF = 1 ,POSTURE_SENSOR_TRAKSTAR =2,POSTURE_SENSOR_REALSENSE =3
    "US_module_config": "",
    "Posture_module_config": ".\\test.uvf"
}

json_mylist = json.dumps(message, separators=(',', ':'))

file = open('LOG.txt', 'w')

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('Phantom_scan_8.avi', fourcc, 5.0, (640, 480))  # for images of size 480*640

ws = websocket.WebSocket()

#### TCP/IP to the remote PC
HOST = '192.168.0.101'
PORT = 6666
buffer_size = 307329

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_address = (HOST, PORT)
sock.connect(server_address)


class Get_Image_Class(Process):
    '''# To set a flag to stop the treads. By initiating the separate thread we make sure that the keyboard
    pressed value is received immediately'''

    def __init__(self):
        Process.__init__(self)
        self.threads_stopper = False # to stop all threads at the same time
        self.num = 0
    def run(self):


            ws.connect("ws://localhost:4100")
            ws.send(json_mylist)

            image_byte_array = []
            binAnswer = []
            while True:
                binAnswer = ws.recv_frame()

                if websocket.ABNF.OPCODE_MAP[binAnswer.opcode] == "text":
                    print("text received")
                    print(len(binAnswer.data))

                # print(websocket.ABNF.OPCODE_MAP[binAnswer.opcode])
                if websocket.ABNF.OPCODE_MAP[binAnswer.opcode] == "binary":
                    # print("bytes: ",bytearray(binAnswer.data).__len__())
                    # we need to receive the data of length 307329
                    image_byte_array = bytearray(binAnswer.data)[129:]
                    # print(len(image_byte_array))

                    sock.sendall(image_byte_array)

                # Create a PIL Image from our pixel array.
                #         pil_image = Image.frombuffer('L',(640, 480),image_byte_array)
                # image = np.array(pil_image)
                #
                # cv2.imshow("image",image)
                #
                # # Don't try to write out gray frames, only BGR, otherwise the output will be empty
                # # out.write(cv2.cvtColor(image,cv2.COLOR_GRAY2BGR))
                # cv2.waitKey(1)

                if keyboard.is_pressed('c'):
                    # print("avg time for one cycle", time_inference.avg)
                    out.release()
                    ws.close()
                    sock.close()
                    os._exit(0)
                    break



def Client_US_frames(ws,out):


    ws.connect("ws://localhost:4100")
    ws.send(json_mylist)

    image_byte_array = []
    binAnswer = []
    while True:
        binAnswer = ws.recv_frame()

        if websocket.ABNF.OPCODE_MAP[binAnswer.opcode] == "text":
            print("text received")
            print(len(binAnswer.data))

        # print(websocket.ABNF.OPCODE_MAP[binAnswer.opcode])
        if websocket.ABNF.OPCODE_MAP[binAnswer.opcode] == "binary":
            # print("bytes: ",bytearray(binAnswer.data).__len__())
            # we need to receive the data of length 307329
            image_byte_array = bytearray(binAnswer.data)[129:]
            # print(len(image_byte_array))

            sock.sendall(image_byte_array)



    # Create a PIL Image from our pixel array.
    #         pil_image = Image.frombuffer('L',(640, 480),image_byte_array)
            # image = np.array(pil_image)
            #
            # cv2.imshow("image",image)
            #
            # # Don't try to write out gray frames, only BGR, otherwise the output will be empty
            # # out.write(cv2.cvtColor(image,cv2.COLOR_GRAY2BGR))
            # cv2.waitKey(1)

        if keyboard.is_pressed('c'):
            # print("avg time for one cycle", time_inference.avg)
            out.release()
            ws.close()
            sock.close()
            os._exit(0)
            break

class Stop_Thread(Process):
    '''# To set a flag to stop the treads. By initiating the separate thread we make sure that the keyboard
    pressed value is received immediately'''

    def __init__(self):
        Process.__init__(self)
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
    try:
        p = Process(target = Client_US_frames(ws,out),args=())
        # p = Get_Image_Class()
        p2 = Stop_Thread()

        p.daemon = True
        p2.daemon = True
        p2.start()
        p.start()
        p.join()
        p2.join()
        # Client_US_frames(ws,out)

    except KeyboardInterrupt:
        print('Hello user you have pressed ctrl-c button.')
        out.release()
        ws.close()
        # sock.close()