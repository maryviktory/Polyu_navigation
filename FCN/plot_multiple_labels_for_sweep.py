import matplotlib.pyplot as plt
import numpy as np
import os
import FCN.sp_utils as utils
from FCN.sp_utils.config import config
import cv2
import logging


def labels_extract(datalist):
    probability_label = np.zeros(0)
    x_array = np.zeros(0)
    y_array = np.zeros(0)
    patient_array = np.zeros((3,1))
    # print(patient_array)
    for i, labels in enumerate(datalist):
        # print("frame num {}", i)
        label_np = cv2.imread(labels,0)

        # label_np = cv2.imread(labels, 0)
        if np.sum(label_np) == 0:
            probability_label = np.append(probability_label, 0)
            x_array = np.append(x_array, 0)
            y_array = np.append(y_array, 0)
            # i - row of centroid, j - column,
            # calculate moments of binary image
        else:
            probability_label = np.append(probability_label, 1)
            # print(np.sum(label_np))

            M = cv2.moments(label_np)

            # calculate x,y coordinate of center
            x = int(M["m10"] / M["m00"])
            y = int(M["m01"] / M["m00"])
            x_array = np.append(x_array, x)
            y_array = np.append(y_array, y)

    return probability_label,x_array,y_array


def main():
    test_dir = "D:\spine navigation Polyu 2021\DATASET_polyu\To label by group\images data"
    color_ext = ["r",'b',"g"]
    fig = plt.figure()
    # fig2 = plt.figure()
    ax = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)

    for sweeps in ["sweep20002"]:
        for i,patient in enumerate(["Kelly","Maria"]):  #,"Timothy"

            print("process {} patient".format(patient))
            patient_dir = os.path.join(test_dir, patient,sweeps)

            labels_dir = os.path.join(patient_dir, "labels")
            labels_list = [os.path.join(labels_dir, item) for item in os.listdir(labels_dir)]

            probability_label, x_array, y_array = labels_extract(labels_list)


            ax.set(xlim=(0, 640), ylim=(0, len(probability_label)))
            ax2.set(xlim=(0, 640), ylim=(0, len(probability_label)))

            plot_path(ax,ax2,probability_label, x_array, y_array,color_ext[i],patient)

        ax.set_xlabel(' X coordinate')
        ax.set_ylabel(' step')
        # ax.legend()
        ax2.legend()
        ax.set_title(sweeps)
    plt.show()

def plot_path(ax,ax2,probability, X, Y,color_ext,title) :
    print("Plotting path")
    path = np.zeros(0)
    index = np.zeros(0)
    zs = 0
    scatter = 0
    for i in range(0, len(X)):
        xs = X[i]
        # ys = Y[i]

        if probability[i] > 0 and X[i] != 0:  # Spinous
            color = color_ext
            marker = "o"
            path= np.append(path,xs)
            index = np.append(index,i)
        else:  # Gap
            color = "#8c564b"
            marker = "x"

        ax.scatter(xs, zs, c=color, marker=marker)

        zs = zs + 1


    path_smoothed = utils.smooth(path)
    ax2.plot(path_smoothed, index, color = color_ext, label=title)
    # ax2.plot(path, index, label=title)

    # t = np.linspace(0, zs, zs)
    # ax1.plot(probability,t_p)

    # plt.show()


if __name__ == '__main__':
    main()
