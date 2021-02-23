import cv2
import numpy as np
from scipy import ndimage
from PIL import ImageGrab

#cap = cv2.VideoCapture("phantomBwater.wmv")
#cap = cv2.VideoCapture("test_ZiHao.mp4")
fheight =1080
fwidth = 1920
#cap = cv2.VideoCapture("video_US_robot.mp4")
kernel = np.ones((5,5),np.uint8)
kernel_opening = np.ones((10,10),np.uint8)
kernel_closing = np.ones((10,10),np.uint8)

#ret, frame = cap.read()
FOV = 0.045 # field of view of the probe 45mm in meters
#######(720L, 1280L) - robot image#########
#########______water____image shape rows, columns, channels (370L, 392L, 3L)_______#########
#h,w,_ = frame.shape
#print(frame.shape)
i = 1
alpha = 0.5
hnew = h = 1080
wnew = w = 1920
centroid_X = wnew/2
X = []
centroid_Y = hnew/2
Y = []

def find_centroid(c):
    M = cv2.moments(c)
    # calculate x,y coordinate of center
    if M["m00"] != 0:
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
    else:
        cX, cY = 0, 0
    return cX,cY


#mask = mask[30 : crop_img.shape[0], 30 : crop_img.shape[1]]
while True:

    frame = ImageGrab.grab(
        bbox=(0, 0, fwidth, fheight))
    frame = np.array(frame)
    h_im = 320
    w_im = 320
    t = 95
    hnew = 2 * h_im
    wnew = 2 * w_im
    frame = frame[((h / 2)) - h_im:(((h / 2)) + h_im),(w / 2 + t - w_im):(w / 2 + t + w_im)]

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #________Water_____#
    #frame = frame[50:h-50, 30:w-30]

    ###########add contrast and brightness########
    contrast = 1.6
    brightness = 0
    contr = cv2.addWeighted(frame, contrast, frame, 0, brightness)
    #contr = frame
    f = ndimage.median_filter(frame, 10)

   # f = cv2.addWeighted(f, contrast, frame, 0, brightness)


#######______for each frame draw centroid____-#######
    # lineThickness = 2; x1 = wnew / 2; y1 = hnew
    # position_relative = x1 - cX
    # cv2.circle(th2, (cX, cY), 20, (0, 100, 25), -1)
    # cv2.putText(th2, "centroid_X: %s"%position_relative, (cX - 25, cY - 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (170, 25, 120), 3)
    # cv2.line(th2, (x1, 0),(x1,y1), (25, 125, 0), lineThickness)

##################______________IMSHOW_______###########
    frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
    #result_img = cv2.resize(result_img, (0, 0), fx=0.5, fy=0.5)
    contr = cv2.resize(contr, (0, 0), fx=0.5, fy=0.5)
    f = cv2.resize(f, (0, 0), fx=0.5, fy=0.5)
    # edges = cv2.resize(edges, (0, 0), fx=0.5, fy=0.5)
    #cv2.imshow('PS', PS_f)

    #th2 = cv2.resize(th2, (0, 0), fx=0.5, fy=0.5)
    #frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
    #cv2.imshow("Result", np.hstack([frame, result_img]))
    #cv2.imshow('erosion',erosion)
    #cv2.imshow('dilation',dilation)
    cv2.imshow('original',frame)
    #cv2.imshow('result',result_img)
    cv2.imshow('contrast', contr)
    #cv2.imshow('result', filtered_contours)


    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break


    i = i + 1

cv2.waitKey(0)
cap.release()
cv2.destroyAllWindows()

