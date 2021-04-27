import matplotlib.pyplot as plt
import numpy as np
import os
import cv2


'''This code helps to generate labels from different subjects and record all in one folder'''
patients_dataset_file = "C:\\Users\maryv\PycharmProjects\Polyu_navigation\FCN_multiclass\dataset_patients_PWH_all.txt"
data_dir = "D:\spine navigation Polyu 2021\DATASET_polyu\PWH_sweeps\Subjects dataset"


# data_dir = "/media/maryviktory/My Passport/IROS 2020 TUM/DATASETs/Dataset/DATA_toNas_for CNN_IPCAI/data set patients images"
# data_dir = "/media/maryviktory/My Passport/IROS 2020 TUM/DATASETs/Dataset/Force_integration_DB/New_Patients"
# label_dir = "/media/maryviktory/My Passport/IROS 2020 TUM/Spine navigation vertebrae tracking/FCN_spine_point_regression/Heatmaps_sweeps/"


def open_file(name_file):
    train_file = open(name_file, "r")
    lines_train = train_file.read().splitlines()
    print('patient names:',lines_train)
    return lines_train


# patient_name = open_file(patients_dataset_file)

images_exist_flag = False

patient_name = "sweep018","sweep3013","sweep5005","sweep9001","sweep20001","sweep18001"

for patient in patient_name:
    input_dir = os.path.join(data_dir, ("%s/Labels_sacrum" % patient))


    label_dir = "D:\spine navigation Polyu 2021\DATASET_polyu\PWH_sweeps\Subjects dataset\%s\Labels_sacrum_heatmaps"%(patient)

    print("input {} to {}",input_dir, label_dir)

    print("process patient {}".format(patient))
    path = os.path.join(label_dir, patient)

    if images_exist_flag == True:
        image_dir = os.path.join(data_dir,("%s/Images"%patient))

        if not os.path.exists(os.path.join(label_dir, patient, 'Images')):
            os.mkdir(os.path.join(path, 'Images'))



    coords = []

    if not os.path.exists(input_dir):
        print("Does not exist {} for {}".format(input_dir,patient))
        input_dir = os.path.join(data_dir, ("%s/Labels_heatmaps" % patient))
        # continue


    label_list = [os.path.join(input_dir, item) for item in os.listdir(input_dir)]


    ksize = (101, 101)
    sigma = 20

    image_list = []

    # if not os.path.exists(os.path.join(label_dir, patient, 'Labels')):
    #
    #     os.mkdir(os.path.join(label_dir, patient))
    #     os.mkdir(os.path.join(path, 'Labels'))



    for label_path in label_list:
        label = cv2.imread(label_path,0)

        if images_exist_flag == True:
            image_path = os.path.join(image_dir, os.path.split(label_path)[-1])
            # print(image_path)

            image = cv2.imread(image_path,0)

        heatmap = np.zeros([label.shape[0], label.shape[1]])
        # print(heatmap.shape)

        if np.sum(label) != 0:

            # i - row of centroid, j - column,
            # calculate moments of binary image
            M = cv2.moments(label)

            # calculate x,y coordinate of center
            x0_0 = int(M["m10"] / M["m00"])
            y0_0 = int(M["m01"] / M["m00"])

            x0 = y0_0
            y0 = x0_0



            x, y = np.arange(label.shape[0]), np.arange(label.shape[1])

            gx = np.exp(-(x - x0) ** 2 / (2 * sigma ** 2))
            gy = np.exp(-(y - y0) ** 2 / (2 * sigma ** 2))
            g = np.outer(gx, gy)
            #g /= np.sum(g)   # normalize, if you want that
            heatmap = heatmap + g* 255


        # image_name = os.path.split(image_path)[-1]
        image_name = os.path.split(label_path)[-1]
        # print(image_name)

        # print(os.path.join(label_dir,patient,'Labels', image_name))
        # cv2.imwrite(os.path.join(label_dir,patient,'Labels', image_name), heatmap)
        cv2.imwrite(os.path.join(label_dir,image_name), heatmap)
        if images_exist_flag == True:
            cv2.imwrite(os.path.join(label_dir, patient,'Images', image_name), image)

        # print("Copy")
        # put text and highlight the center
        # cv2.circle(image, (x0_0, y0_0), 5, 0, -1)
        # plt.subplot(1, 2, 1)
        # plt.imshow(image)
        #
        # plt.subplot(1, 2, 2)
        # plt.imshow(heatmap)

        # plt.show()



