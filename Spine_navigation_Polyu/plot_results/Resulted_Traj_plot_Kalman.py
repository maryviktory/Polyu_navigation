import os
import logging
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from datetime import datetime
from scipy.interpolate import interp1d
import pandas as pd
# from pykalman import KalmanFilter
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise
from Spine_navigation_Polyu.utils.functions import Kalman_filter_x_im
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
    Q = 90 # process variance #1e-5

    # allocate space for arrays
    xhat=np.zeros(sz)      # a posteri estimate of x
    P=np.zeros(sz)         # a posteri error estimate
    xhatminus=np.zeros(sz) # a priori estimate of x
    Pminus=np.zeros(sz)    # a priori error estimate
    K=np.zeros(sz)         # gain or blending factor

    R = 500 # estimate of measurement variance, change to see effect
    # print("variance",R)
    # intial guesses
    xhat[0] = 224/2

    P[0] = 0
    nth_list = [x for x in range(1,len(z))][0::5]
    # t = [x for x in range(1,len(z))]
    print(nth_list)
    B = 50
    # for k in range(1,n_iter):
    # filt = Kalman_filter_x_im()

    for k in range(1,n_iter):
        # xhatminus = np.append(xhatminus, xhat[-1])  # +B*0.01
        # Pminus = np.append(Pminus, P[-1] + Q)
        # filt.predict_stage()

        # R = variance(z[:k])

        # print("variance", R)
        # time update
        xhatminus[k] = xhat[k-1]#+B*0.01
        Pminus[k] = P[k-1]+Q
        # print(P[k-1],P[-1])
        # print("xhatminus",xhatminus[k])
        if k in nth_list:
            # K = np.append(K, Pminus[-1] / (Pminus[-1] + R))
            # # print("K[k]",K[k])
            # xhat = np.append(xhat, (xhatminus[-1] + K[-1] * (z[k] - xhatminus[-1])))
            # P = np.append(P, (1 - K[-1]) * Pminus[-1])

            # print(k)
            # measurement update
            K[k] = Pminus[k]/( Pminus[k]+R )
            # print("K[k]",K[k])
            xhat[k] = xhatminus[k]+K[k]*(z[k]-xhatminus[k])
            P[k] = (1-K[k])*Pminus[k]
            # print("xhat",xhat[k])
            # filt.update_with_measurement(mes)
        else:
            # xhat = np.append(xhat, xhatminus[-1])

            # filt.xhat = np.append(filt.xhat,filt.xhatminus[-1])
            xhat[k] = xhatminus[k]
    print(xhat[:100])
    print(len(xhat))
        # print(xhat)
    t2 = np.linspace(0,len(X_label),len(X_label))
    plt.figure()
    plt.plot(z,t,'k+',label='noisy measurements')
    plt.plot(xhat,t,'b-',label='a posteri estimate')
    # plt.plot(label_x,t,'ro',label = "labels")
    # plt.plot(X_label,t2,'rx',label = "labels")
    # plt.axhline(x,color='g',label='truth value')
    plt.legend()
    plt.title('Estimate vs. iteration step', fontweight='bold')
    plt.xlabel('X coordinate')
    plt.ylabel('Iteration')
    plt.show()




def make_ca_filter(dt, std):
    cafilter = KalmanFilter(dim_x=3, dim_z=1)
    cafilter.x = np.array([224/2, 0., 0.])
    cafilter.P *= 3
    cafilter.R *= std
    cafilter.Q = Q_discrete_white_noise(dim=3, dt=dt, var=0.02)
    cafilter.F = np.array([[1, dt, 0.5*dt*dt],
                           [0, 1,         dt],
                           [0, 0,          1]])
    cafilter.H = np.array([[1., 0, 0]])
    return cafilter

def initialize_const_accel(f):
    f.x = np.array([0., 0., 0.])
    f.P = np.eye(3) * 3

def make_cv_filter(dt, std):
    cvfilter = KalmanFilter(dim_x = 2, dim_z=1)
    cvfilter.x = np.array([0., 0.])
    cvfilter.P *= 3
    cvfilter.R *= std**2
    cvfilter.F = np.array([[1, dt],
                           [0,  1]], dtype=float)
    cvfilter.H = np.array([[1, 0]], dtype=float)
    cvfilter.Q = Q_discrete_white_noise(dim=2, dt=dt, var=0.02)
    return cvfilter

def initialize_filter(kf, std_R=None):
    """ helper function - we will be reinitialing the filter
    many times.
    """
    kf.x.fill(0.)
    kf.P = np.eye(kf.dim_x) * .1
    if std_R is not None:
        kf.R = np.eye(kf.dim_z) * std_R


if __name__ == '__main__':
    main()



