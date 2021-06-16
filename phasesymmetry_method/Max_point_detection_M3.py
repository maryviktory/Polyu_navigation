import cv2
import numpy as np
from scipy import ndimage
import phasesymmetry_method.phasesym as phasesym

import FCN.sp_utils as utils
import os

# from operator import itemgetter

#cap = cv2.VideoCapture("phantomBwater.wmv")
video_path = "D:\spine navigation Polyu 2021\DATASET_polyu\PWH_sweeps\Phantom_scan\Phantom_scan_4.avi" #phantom test scan
# video_path = "test_ZiHao.mp4"
image_path = "D:\spine navigation Polyu 2021\DATASET_polyu\PWH_sweeps\Subjects dataset\phantom_scans\phantom_sweep_4\Images"
Labels_path = "D:\spine navigation Polyu 2021\DATASET_polyu\PWH_sweeps\Subjects dataset\phantom_scans\phantom_sweep_4\Labels_heatmaps"
label_list = [os.path.join(Labels_path , item) for item in os.listdir(Labels_path )]
image_list = [os.path.join(image_path , item) for item in os.listdir(image_path )]




cap = cv2.VideoCapture(video_path) # "test_ZiHao.mp4"

#cap = cv2.VideoCapture("video_US_robot.mp4")
kernel = np.ones((5,5),np.uint8)
kernel_opening = np.ones((10,10),np.uint8)
kernel_closing = np.ones((10,10),np.uint8)

ret, frame = cap.read()
FOV = 0.045 # field of view of the probe 45mm in meters
#######(720L, 1280L) - robot image#########
#########______water____image shape rows, columns, channels (370L, 392L, 3L)_______#########
h,w,_ = frame.shape
print(frame.shape)
i = 1
alpha = 0.5
hnew = h
wnew = w
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

acc = utils.AverageMeter()
dist_error = utils.AverageMeter()
#mask = mask[30 : crop_img.shape[0], 30 : crop_img.shape[1]]
# while True:
for image,label in zip(image_list,label_list):
    frame = cv2.imread(image)
    # ret, frame = cap.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    label = cv2.imread(label)
    label = cv2.cvtColor(label, cv2.COLOR_BGR2GRAY)
    # print(frame)
    #________Water_____#
    frame = frame[90:h-90, 30:w-30]
    label = label[90:h - 90, 30:w - 30]
    # frame = frame[150:(h - 150), 350:(w - 350)]
    # frame = frame
    hnew = h -90-90
    wnew = w-30 - 30
    print()
    empty_image = np.zeros((frame.shape[0], frame.shape[1]), np.uint8)
    centroid_X = wnew / 2
    X = []
    centroid_Y = hnew / 2
    Y = []
    ###########add contrast and brightness########
    contrast = 1.2
    brightness = 1.5
    frame = cv2.addWeighted(frame, contrast, frame, 0, brightness)
    contr = frame
    f = ndimage.median_filter(frame, 10)
    contrast = 1.6
    brightness = 1.5
   # f = cv2.addWeighted(f, contrast, frame, 0, brightness)

    PS_f, orientation, _, T = phasesym.phasesym(frame, nscale=2, norient=1, minWaveLength=25, mult=1.6,
                                              sigmaOnf=0.25, k=1.5, polarity=1, noiseMethod=-1)
    #print(result.shape, type(result), result.dtype)

    PS = (PS_f*255).astype(np.uint8)

    ret, th2 = cv2.threshold(PS, 30, 255, cv2.THRESH_BINARY)

    ##########   erosion   ###########
    kernel = np.ones((5, 5), np.uint8)
    erosion = cv2.erode(th2, kernel, iterations=2)
    th2 = erosion

    ##########__________small CONTOURS SELECTION____________##########
    cnts, hierarchy = cv2.findContours(th2, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    contours = th2
    contours = cv2.cvtColor(contours, cv2.COLOR_GRAY2RGB)
    for c in cnts:
        cv2.drawContours(contours, c, -1, (155, 0, 20), 2)


    mask = np.ones(th2.shape[:2], dtype="uint8") * 255
    minArea = 100

    filteredContours = []
    fitcontours = []
    for c in cnts:
        area = cv2.contourArea(c)
        if area < minArea:
            filteredContours.append(c)
            cv2.drawContours(mask, [c], -1, 0, -1)
        else:
            fitcontours.append(c)

    small_contours_deleted = cv2.bitwise_and(th2, th2, mask=mask)
    th2 = small_contours_deleted
############____Find min of biggest contour ##########
    cm = max(cnts, key=cv2.contourArea)
    bottommost = tuple(cm[cm[:, :, 1].argmax()][0])
    min_X, min_y = bottommost

    mask_with_small_contours_deleted = np.ones(small_contours_deleted.shape[:2], dtype="uint8") * 255
    elim_contours_above = []
    # kot = []
####____eliminate_the highest countours___#####
    cY_prev = 0

    for c in fitcontours:
        cX, cY = find_centroid(c)
        if (cY < min_y and cY < hnew/5 or cY < hnew/5 ):
            elim_contours_above.append(c)
            cv2.drawContours(mask_with_small_contours_deleted, [c], -1, 0, -1)


    filtered_contours = cv2.bitwise_and(small_contours_deleted, small_contours_deleted,
                                        mask=mask_with_small_contours_deleted)


 ################################
    dilation = cv2.dilate(filtered_contours, kernel, iterations=10)
    th2 = dilation


################_________Contours_______################
    cnts, hierarchy = cv2.findContours(th2, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    #if cnts != None:
    try:
        c = max(cnts, key=cv2.contourArea)
        cv2.drawContours(th2, c, -1, (155, 255, 255), 5)
    except:
        print("no contour")
##########________Extreme_points_______#############
    tops = []
    X_list = []
    Y_list = []
    top_positions = []
    important_contours = []

    for c in cnts:
        cX, cY = find_centroid(c)
        if (cY < 8 * hnew / 10):

        # extBot = tuple(c[c[:, :, 1].argmax()][0])
        # extLeft = tuple(c[c[:, :, 0].argmin()][0])
        # extRight = tuple(c[c[:, :, 0].argmax()][0])
            extTop = tuple(c[c[:, :, 1].argmin()][0])
            max_X, max_y = extTop
            tops.append(extTop)
            X_list.append(max_X)

            Y_list.append(max_y)
            pos = max_X, max_y
            top_positions.append(pos)
    top_positions = np.array(top_positions, dtype = np.int32)

    if len(Y_list)>0:
        maxY = min(Y_list)
    else:
        maxY = 0

    if len(X_list) != 0:
        meanX = sum(X_list) / len(X_list)
    else:
        meanX = wnew/2

    #maxTop = meanX, maxY

    if len(tops)>0:
        maxTop = min(tops, key=lambda item: item[1])
    #print(maxTop)

    result_img = cv2.cvtColor(th2, cv2.COLOR_GRAY2RGB)
    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
###############____Centroid_approximation____#########
    alpha = 0.9
    if i ==0:
        centroid_X,centroid_Y = meanX, maxY
    if i > 0:
        #centroid_X = int(round((1 - alpha)*centroid_X +alpha*maxTop[0]))
        centroid_X = int(round((1 - alpha) * centroid_X + alpha * meanX))
        #centroid_Y = int(round((1 - alpha) * centroid_Y + alpha * maxY))
    #centroid_X, centroid_Y = meanX, maxY
    centroid_Y = maxY
    #centroid_Y = 200


#################_________Relative_Coordinate__X image - Z robot______############
    lineThickness = 2;
    # coordinate of image center
    x_image_center = wnew / 2;
    y_image_center = hnew
    position_relative = centroid_X - x_image_center # minus for left point, plus for right point

# proportion between FOV of probe and Pixels in image: FOV/wnew
    K_image_t_dist = FOV/wnew
    Z_robot_cord_f_image = K_image_t_dist*position_relative


    print(Z_robot_cord_f_image)

############________HULL__________##########

    top_contour = [top_positions]
    # for cnt in top_contour:
    #     hull = cv2.convexHull(cnt)
       # cv2.drawContours(result_img, [hull], -1, (0,255,255), 2)

    #     cv2.drawContours(frame, hull, i,color, 1, 8)
    #cv2.circle(result_img , extLeft, 8, (0, 0, 255), -1)
    #cv2.circle(result_img , extRight, 8, (0, 255, 0), -1)
   # cv2.circle(result_img , extTop, 8, (255, 0, 0), -1)


    for points in tops:
        cv2.circle(result_img, points, 15, (255, 0, 0), -1)

    cv2.circle(result_img, (centroid_X,centroid_Y), 15, (0, 255, 0), -1)
    cv2.circle(frame, (centroid_X,centroid_Y), 15, (0, 255, 0), -1)

    #cv2.circle(result_img , extBot, 8, (255, 255, 0), -1)
    #cv2.drawContours(result_img,hull, -1, (155, 255, 255), 5)
    #cv2.line(result_img, (x1, 0), (x1, y1), (25, 125, 0), lineThickness)
    #cv2.circle(result_img, (cX, cY), 20, (20, 125, 120), -1)
    #cv2.circle(result_img, (centroid_X, centroid_Y), 20, (200, 25, 0), -1)
    # cv2.putText(result_img, 'number of frame:%s'%i,(20,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
    # (170, 25, 120), 2)

    cv2.putText(result_img, "centroid_X: %s" % np.round(Z_robot_cord_f_image,4), (centroid_X - 25, centroid_Y - 25),
                cv2.FONT_HERSHEY_SIMPLEX, 1,
                 (170, 25, 120), 3)


#######______for each frame draw centroid____-#######
    # lineThickness = 2; x1 = wnew / 2; y1 = hnew
    # position_relative = x1 - cX
    # cv2.circle(th2, (cX, cY), 20, (0, 100, 25), -1)
    # cv2.putText(th2, "centroid_X: %s"%position_relative, (cX - 25, cY - 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (170, 25, 120), 3)
    # cv2.line(th2, (x1, 0),(x1,y1), (25, 125, 0), lineThickness)

##################______________IMSHOW_______###########
    # frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
    # result_img = cv2.resize(result_img, (0, 0), fx=0.5, fy=0.5)
    # contr = cv2.resize(contr, (0, 0), fx=0.5, fy=0.5)
    # f = cv2.resize(f, (0, 0), fx=0.5, fy=0.5)
    # edges = cv2.resize(edges, (0, 0), fx=0.5, fy=0.5)


    #cv2.imshow('PS', PS_f)

    #th2 = cv2.resize(th2, (0, 0), fx=0.5, fy=0.5)
    #frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
    #cv2.imshow("Result", np.hstack([frame, result_img]))
    #cv2.imshow('erosion',erosion)
    #cv2.imshow('dilation',dilation)

    ##########____Calculate accuracy with labels___############
    ksize = (101, 101)
    sigma = 20
    heatmap = np.zeros([label.shape[0], label.shape[1]])
    x, y = np.arange(label.shape[0]), np.arange(label.shape[1])

    gx = np.exp(-(x - centroid_X) ** 2 / (2 * sigma ** 2))
    gy = np.exp(-(y - centroid_Y) ** 2 / (2 * sigma ** 2))
    g = np.outer(gx, gy)
    # g /= np.sum(g)   # normalize, if you want that
    heatmap = heatmap + g * 255


    cv2.circle(empty_image, (centroid_X, centroid_Y), 15, 255, -1)

    empty_image2=empty_image

    label = np.expand_dims(label, axis=0)
    label = np.expand_dims(label, axis=0)


    empty_image = np.expand_dims(empty_image, axis=0)
    empty_image = np.expand_dims(empty_image, axis=0)


    acc_functional, avg_acc, cnt, pred, target, dists = utils.accuracy(empty_image,
                                                                       label,
                                                                       thr=0.5)
    acc.update(avg_acc, cnt)

    # real_distance = utils.real_distance_error_calculation(dists, config)
    if dists != -1:
    #
        dist_error.update(dists)



    cv2.imshow('empty im_', empty_image2)
    cv2.imshow('original',frame)
    cv2.imshow('result',result_img)
    cv2.imshow('resulted heatmap', heatmap)
    cv2.imshow("label",np.squeeze(label))
    cv2.imshow('empty im', np.squeeze(empty_image))
    #cv2.imshow('contrast', contr)
    #cv2.imshow('result', filtered_contours)


    k = cv2.waitKey(30) & 0xff
    if k == 27:
        print("Accuracy mean {}".format(acc.avg))
        print("Distance mean {}".format(dist_error.avg))
        break


    i = i + 1

print("Accuracy mean {}".format(acc.avg))
print("Distance mean {}".format(dist_error.avg))

cv2.waitKey(0)
cap.release()
cv2.destroyAllWindows()

