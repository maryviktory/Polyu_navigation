import os
import logging
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from datetime import datetime
from scipy.interpolate import interp1d
import pandas as pd

def variance(data):
    # Number of observations
    n = len(data)
    # Mean of the data
    mean = sum(data) / n
    # Square deviations
    deviations = [(x - mean) ** 2 for x in data]
    # Variance
    variance = sum(deviations) / n
    return variance

def plot_traj(X,X_label,probability,name):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set(xlim=(0, 224), ylim=(0, len(probability)))

    path = np.zeros(0)
    index = np.zeros(0)
    label = np.zeros(0)
    index_point_array = np.zeros((0,3))
    path_target = np.zeros(0)
    index_target = np.zeros(0)
    zs = 0
    for i in range(0, len(X)):
        xs = X[i]
        x_lab = X_label[i]
        # ys = Y[i]

        if probability[i] > 0.6 and X[i] != 0:  # Spinous
            # print("append")
            color = "r"
            marker = "o"
            path = np.append(path, xs)
            index = np.append(index, i)
            label = np.append(label, x_lab)

            index_point_array = np.append(index_point_array,[[i,xs,x_lab]],axis=0 )

        else:  # Gap
            xs = 0
            color = "#8c564b"
            marker = "x"

        ax.scatter(xs, zs, c=color, marker=marker)

        # if probability_label[i] > 0.7 and X_label[i] != 0:  # Spinous
        #     color = "b"
        #     marker = "o"
        #     path_target = np.append(path, X_label[i])
        #     index_target = np.append(index, i)
        # else:  # Gap
        #     color = "#8c564b"
        #     marker = "x"
        # ax.scatter(X_label[i], zs, c=color, marker=marker)

        zs = zs + 1
    ax.plot(path, index,color='r', label="Predicted")
    ax.plot(path_target, index_target, color='b',label="Target")

    # ax1.plot(probability,t_p)
    ax.set_xlabel('coordinate {}'.format(name))
    ax.set_ylabel(' step')
    return path, index, index_point_array

    # plot_path(probability, X, Y,'r')
    # plot_path(probability_label, X_label, Y_label, "b")
def data_extraction():
    traj_path = "D:\IROS 2020 TUM\Spine navigation vertebrae tracking\FCN_spine_point_regression\Dataset_Heatmaps_all_subjects\Maria_T\Maria_T.csv"
    data_path_heatmap = "D:\IROS 2020 TUM\Spine navigation vertebrae tracking\FCN_spine_point_regression\Spinous_positions_sweeps\Maria_T.npz"
    csv_dframe = pd.read_csv(traj_path, delimiter=";")

    try:
        y = np.squeeze(csv_dframe["y"])
        x = np.squeeze(csv_dframe["x"])
        # print(len(y))

    except:
        print("FAILED")

    p_i, p_e = 0, len(y)


    data_array = np.load(data_path_heatmap)
    X = data_array["X"]
    Y = data_array["Y"]
    probability = data_array["probability"]
    X_label = data_array["X_label"]
    Y_label = data_array["Y_label"]


    print(len(X), len(X_label))
    # print(probability.size)
    # print(data_array["X"], len(data_array["X"]))
    path,index,index_point_array = plot_traj(X,X_label,probability,'X')
    # print(path)
    # print(index.shape,index)
    ###___INTERPOLATE___####
    f = interp1d(index,path)
    frames = np.linspace(0,index[-1],int(index[-1]))

    path_interp = f(frames)
    # print(path_interp)
    plt.plot(path_interp,frames)
    plt.title('path_inter,frames', fontweight='bold')
    # plt.plot(x[p_i:p_e],y[p_i:p_e])

    # plt.show()
    return path,path_interp, index_point_array, X_label


def main():
    plt.rcParams['figure.figsize'] = (10, 8)

    # intial parameters
    path,path_inter, index_point_array,X_label= data_extraction()
    # n_iter = len(path_inter)
    n_iter = len(index_point_array[:,1])
    sz = (n_iter,) # size of array
    # x = -0.37727 # truth value (typo in example at top of p. 13 calls this z)
    # z = np.random.normal(x,0.1,size=sz) # observations (normal about x, sigma=0.1)

    # z = path_inter
    # t = np.linspace(0,path_inter[-1],len(path_inter))
    z = index_point_array[:,1]
    t = index_point_array[:,0]
    label_x = index_point_array[:,2]
    # print(z, len(z))
    Q = 1e-5 # process variance #1e-5

    # allocate space for arrays
    xhat=np.zeros(sz)      # a posteri estimate of x
    P=np.zeros(sz)         # a posteri error estimate
    xhatminus=np.zeros(sz) # a priori estimate of x
    Pminus=np.zeros(sz)    # a priori error estimate
    K=np.zeros(sz)         # gain or blending factor

    R = 0.001 # estimate of measurement variance, change to see effect
    # print("variance",R)
    # intial guesses
    xhat[0] = 224/2
    P[0] = 0.0

    for k in range(1,n_iter):
        # R = variance(z[:k])

        # print("variance", R)
        # time update
        xhatminus[k] = xhat[k-1]
        Pminus[k] = P[k-1]+Q

        # measurement update
        K[k] = Pminus[k]/( Pminus[k]+R )
        xhat[k] = xhatminus[k]+K[k]*(z[k]-xhatminus[k])
        P[k] = (1-K[k])*Pminus[k]
        # print(z[k])

    t2 = np.linspace(0,len(X_label),len(X_label))
    plt.figure()
    plt.plot(z,t,'k+',label='noisy measurements')
    plt.plot(xhat,t,'b-',label='a posteri estimate')
    # plt.plot(label_x,t,'ro',label = "labels")
    plt.plot(X_label,t2,'rx',label = "labels")
    # plt.axhline(x,color='g',label='truth value')
    plt.legend()
    plt.title('Estimate vs. iteration step', fontweight='bold')
    plt.xlabel('X coordinate')
    plt.ylabel('Iteration')
    plt.show()



if __name__ == '__main__':
    main()



