import matplotlib.pyplot as plt
import numpy as np
import os
import cv2

patients_dataset_file = "C:\\Users\maryv\PycharmProjects\Polyu_navigation\FCN\dataset_patients.txt"
data_dir = "D:\spine navigation Polyu 2021\DATASET_polyu\PWH_sweeps\Subjects dataset\phantom_scans"
label_dir = data_dir

# data_dir = "/media/maryviktory/My Passport/IROS 2020 TUM/DATASETs/Dataset/DATA_toNas_for CNN_IPCAI/data set patients images"
# data_dir = "/media/maryviktory/My Passport/IROS 2020 TUM/DATASETs/Dataset/Force_integration_DB/New_Patients"
# label_dir = "/media/maryviktory/My Passport/IROS 2020 TUM/Spine navigation vertebrae tracking/FCN_spine_point_regression/Heatmaps_sweeps/"


def open_file(name_file):
    train_file = open(name_file, "r")
    lines_train = train_file.read().splitlines()
    print('patient names:',lines_train)
    return lines_train


patient_name = open_file(patients_dataset_file)
patient_name = "phantom_sweep_4","sweep"

for patient in patient_name:


    image_dir = os.path.join(data_dir,("%s/Images"%patient))
    input_dir =  os.path.join(data_dir,("%s/Labels"%patient))
    coords = []
    if not os.path.exists(image_dir):
        print("Does not exist {} for {}".format(image_dir,patient))
        continue


    label_list = [os.path.join(input_dir, item) for item in os.listdir(input_dir)]


    ksize = (101, 101)
    sigma = 20

    image_list = []
    path = os.path.join(label_dir, patient)
    if not os.path.exists(os.path.join(label_dir, patient, 'Labels_heatmaps')):
        # os.mkdir(os.path.join(label_dir, patient))
        os.mkdir(os.path.join(path,'Labels_heatmaps'))
    # if not os.path.exists(os.path.join(label_dir, patient, 'Images')):
    #     os.mkdir(os.path.join(path, 'Images'))



    for label_path in label_list:
        label = cv2.imread(label_path,0)


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


        image_name = os.path.split(image_path)[-1]


        # print(os.path.join(label_dir,patient,'Labels', image_name))
        cv2.imwrite(os.path.join(label_dir,patient,'Labels_heatmaps', image_name), heatmap)
        # cv2.imwrite(os.path.join(label_dir, patient,'Images', image_name), image)

        # put text and highlight the center
        # cv2.circle(image, (x0_0, y0_0), 5, 0, -1)
        # plt.subplot(1, 2, 1)
        # plt.imshow(image)
        #
        # plt.subplot(1, 2, 2)
        # plt.imshow(heatmap)

        # plt.show()



