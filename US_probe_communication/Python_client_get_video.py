import json
import websocket
from PIL import  Image
import numpy as np
import cv2
import keyboard
import os
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


def Client_US_frames(ws,out):

    ws.connect("ws://localhost:4100")
    ws.send(json_mylist)
    # ws.send_binary([100, 220, 130])
    image_byte_array = []
    binAnswer = []
    while True:
        binAnswer = ws.recv_frame()

        # print(websocket.ABNF.OPCODE_MAP[binAnswer.opcode])
        if websocket.ABNF.OPCODE_MAP[binAnswer.opcode] == "binary":
            # print("bytes: ",bytearray(binAnswer.data).__len__())
            # we need to receive the data of length 307329
            image_byte_array = bytearray(binAnswer.data)[129:]

    # Create a PIL Image from our pixel array.
            pil_image = Image.frombuffer('L',(640, 480),image_byte_array)
            image = np.array(pil_image)

            cv2.imshow("image",image)
            print(image.shape)

            # Don't try to write out gray frames, only BGR, otherwise the output will be empty
            out.write(cv2.cvtColor(image,cv2.COLOR_GRAY2BGR))
            cv2.waitKey(1)

        if keyboard.is_pressed('c'):
            # print("avg time for one cycle", time_inference.avg)
            out.release()
            ws.close()
            os._exit(0)
if __name__ == '__main__':
    try:
        Client_US_frames(ws,out)
        out.release()
        ws.close()
    except KeyboardInterrupt:
        print('Hello user you have pressed ctrl-c button.')
        out.release()
        ws.close()