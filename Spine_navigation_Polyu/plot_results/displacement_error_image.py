import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

def smooth(signal, N=30):
    N = 2  # Filter order
    Wn = 0.05  # Cutoff frequency
    B, A = butter(N, Wn, output='ba')
    # Second, apply the filter
    filt = filtfilt(B, A, signal)
    return filt

def plot_path(probability, X, Y,color_ext,labels = None) :
    print("Plotting path")
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set(xlim=(0, 640), ylim=(0, len(probability)))

    path = np.zeros(0)
    index = np.zeros(0)
    zs = 0
    for i in range(0, len(X)):
        xs = X[i]
        ys = Y[i]

        if probability[i] > 0.5 and X[i] != 0:  # Spinous
            color = color_ext
            marker = "o"
            path= np.append(path,xs)
            index = np.append(index,i)
        else:  # Gap
            color = "#8c564b"
            marker = "x"

        ax.scatter(xs, zs, c=color, marker=marker)

        zs = zs + 1
    ax.plot(smooth(path), index, label="coordinate continious")
    t = np.linspace(0, zs, zs)


    # ax1.plot(probability,t_p)
    ax.set_xlabel(' X coordinate')
    ax.set_ylabel(' step')
    plt.show()
    return path

folder = "2"
data_path = "D:\spine navigation Polyu 2021\\robot_trials_output\phantom experiments\%s\Move_thread_output0.csv"%folder

frame = pd.read_csv(data_path)

print(frame["X"])
X_mid = 640/2
X_im = frame["X_im"]
Y_im = frame["Y_im"]
timestamp = frame["timestamp"] - frame["timestamp"][0]
Y_robot = frame["Y_tcp"]
X_robot = frame ["X_tcp"]
X_force = frame["Fx"]
Y_force = frame["Fy"]
Z_force = frame["Fz"]
frame_probability = frame["Frame_Probability"]

delta_X = abs(X_im - X_mid)



fig = plt.figure(1)
# plt.plot( timestamp,delta_X)
# plt.plot(X_im,Y_robot)
# plt.plot(X_robot, Y_robot)

t = np.linspace(0, len(X_robot), len(X_robot))
plt.plot(-smooth(X_robot),t)
# fig2 = plt.figure(2)

# plt.plot(timestamp,smooth(Y_force))
# plt.title(" Y force")

# fig3 = plt.figure(3)

# plt.plot(timestamp,smooth(X_force))
# plt.title(" X force")

path = plot_path(frame_probability,X_im,Y_im,"r")

# fig4 = plt.figure(4)

# plt.plot(path,timestamp)
# plt.title(" X_im")

plt.show()