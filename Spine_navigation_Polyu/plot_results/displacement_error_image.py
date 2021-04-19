import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

def smooth(signal, N=2):
    # N = 2  # Filter order
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
    # fig2 = plt.figure()
    # ax2 = plt.gca()
    # ax2.set_aspect((320 / zs) * 5)
    # ax2.set_xlim(0,640)
    # ax2.plot(smooth(path), index,label = "X image coordinate")
    # ax2.set_xlabel('X image smoothed')
    # ax2.set_ylabel('step')
    # ax.plot(320,0, 320,zs, 'g--', label='line 1', linewidth=2)
    t = np.linspace(0, zs, zs)

    fig4 = plt.figure()
    ax4 = plt.gca()
    ax4.set_aspect((320 / len(probability)) * 5)
    ax4.set_xlim(0, 640)
    ax4.plot(smooth(path), index,label = "X image coordinate")
    x = [320,320]
    y = [0,len(probability)]
    ax4.plot(x,y,'g--',label = "middle line")
    ax4.set_xlabel('X image smoothed')
    ax4.set_ylabel('step')

    fig3 = plt.figure()
    ax1 = plt.gca()  # you first need to get the axis handle
    # ax1.set_aspect((320/zs)*5)
    ax1.plot(index,smooth(abs(320-path)), label = "Delta X image")
    ax1.set_ylim(0, 640/2)
    ax1.set_ylabel('delta Xim')
    ax1.set_xlabel('step')

    # ax1.plot(probability,t_p)
    ax.set_xlabel(' X coordinate')
    ax.set_ylabel(' step')
    ax.legend()
    ax1.legend()
    ax4.legend()
    # ax2.legend()
    plt.show()
    return path

folder = "phantom experiments\scanning\median_filter_k21"
data_path = "D:\spine navigation Polyu 2021\\robot_trials_output\%s\Move_thread_output0.csv"%folder

frame = pd.read_csv(data_path)

# print(frame["X"])
X_mid = 640/2
X_im = frame["X_im"]
Y_im = frame["Y_im"]
timestamp = frame["timestamp"] - frame["timestamp"][0]
Y_robot = frame["Y"]
X_robot = frame ["X"]
Z_robot = frame["Z"]
X_force = frame["Fx"]
Y_force = frame["Fy"]
Z_force = frame["Fz"]
frame_probability = frame["Frame_Probability"]
X_tcp = frame["X_tcp"]
Y_tcp = frame["Y_tcp"]
Z_tcp = frame["Z_tcp"]
delta_X = abs(X_im - X_mid)

t = np.linspace(0, len(X_robot), len(X_robot))

plt.figure()
# plt.plot(timestamp,smooth(delta_X))
ax = plt.gca() #you first need to get the axis handle
ax.set_aspect(5*0.08/len(X_robot)) #sets the height to width ratio to 1.5.
# plt.plot(smooth(-Y_robot),Z_robot)
ax.plot(smooth(-X_tcp),t)
ax.set_xlabel('X_tcp')
ax.set_ylabel('step')
ax.set_xlim(-0.08,0.08)
#

# plt.plot(X_robot, Y_robot)

# plt.plot(-smooth(X_robot),t)

plt.figure()

plt.plot(t,smooth(Y_force))
plt.title(" Y force")

plt.figure()

plt.plot(t,smooth(X_force))
plt.title(" X force")

plt.figure()
plt.plot(t,smooth(Z_force))
plt.title(" Z force")

plt.figure()
plt.plot(t,smooth(frame_probability))
plt.title(" Probability")

# path = plot_path(frame_probability,X_im,Y_im,"r")



# plt.plot(path,timestamp)
# plt.title(" X_im")




plt.show()