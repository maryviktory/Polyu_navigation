import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
import cv2
import os


def smooth(signal, N=1):
    # N = 2  # Filter order
    Wn = 0.05  # Cutoff frequency
    B, A = butter(N, Wn, output='ba')
    # Second, apply the filter
    filt = filtfilt(B, A, signal)
    return filt

def undrift(signal,N=1):
    # N = 2  # Filter order
    Wn = 0.01  # Cutoff frequency
    B, A = butter(N, Wn, output='ba')
    # Second, apply the filter
    avg = filtfilt(B, A, signal)
    undrifted = signal - avg

    return undrifted

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
    ax.plot(path, index, label="coordinate continious")
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
    # plt.show()
    return path, index

folder = "\\trial high fps\\2_v004_robot_fps_20\\normal_fps"
data_path = "D:\spine navigation Polyu 2021\\robot_trials_output\%s\Move_thread_output0.csv"%folder

try:
    image = cv2.imread(os.path.join(os.path.split(data_path)[0],"outputP0.bmp"))
    image_height, width = image.shape[:2]
except:
    print("no image")


frame = pd.read_csv(data_path)
frame_IFL = pd.read_csv("D:\IROS 2020 TUM\DATASETs\Dataset\Force_integration_DB\complete_df_from Maria\Ardit_F15.csv")
# print(frame["X"])

timestamp = frame["timestamp"] - frame["timestamp"][0]
Y_robot = frame["Y"]
X_robot = frame ["X"]
Z_robot = frame["Z"]
X_force = frame["Fx"]
Y_force = frame["Fy"]
Z_force = frame["Fz"]


t = np.linspace(0, len(Z_force), len(Z_force))


plt.figure()
plt.plot(t,Z_force)

plt.title(" Z force")


print("Z_force")
print("MEAN:",np.mean(Z_force[:400]))
print("STD",np.std(Z_force[:400]))
print("Z_force")
print("MEAN:",np.mean(Z_force[400:]))
print("STD",np.std(Z_force[400:]))


# plt.show()




X_mid = 640/2
X_im = frame["X_im"]
Y_im = frame["Y_im"]
frame_probability = frame["Frame_Probability"]
# frame_probability_sacrum = frame["Frame_Probability_sacrum"]
X_tcp = frame["X_tcp"]
Y_tcp = frame["Y_tcp"]
Z_tcp = frame["Z_tcp"]


# force_X_IFL = np.array(frame_IFL["Force X (N)"].str.replace(',', '.'))
# force_Y_IFL = frame_IFL["Force Y (N)"].str.replace(',', '.')
# force_Z_IFL = np.array(frame_IFL["Force Z (N)"])
force_X_IFL = frame_IFL["x_force"][240:]
force_Y_IFL = frame_IFL["y_force"][240:]
force_Z_IFL = frame_IFL["z_force"][400:]

print("mean IFL Z force", np.mean(force_Z_IFL))
print("std IFL Z force", np.std(force_Z_IFL))

# for index, row in frame_IFL.iterrows():
#     # print(row["class_num"])
#
#     val_1 = float(row["Force X (N)"].replace(',', '.'))
#     val_2 = float(row["Force Y (N)"].replace(',', '.'))
#     val_3 = float(row["Force Z (N)"].replace(',', '.'))
#     # array = [float(x) for x in val[0:-1].split(',')]
#     force_X_IFL = np.append(force_X_IFL, val_1)
#     force_Y_IFL = np.append(force_Y_IFL, val_2)
#     force_Z_IFL = np.append(force_Z_IFL, val_3)
#
# print(force_Y_IFL)
# class_num = frame["class_num"]

# plt.figure()
# plt.plot(force_X_IFL)
# plt.title(" X IFL force")
#
# plt.figure()
# plt.plot(force_Y_IFL)
# plt.title(" Y IFL force")
# plt.figure()
# plt.plot(force_Z_IFL)
# plt.title(" Z IFL force")


classes = False
if classes ==True:
    class_1 = np.zeros(0)
    class_2 = np.zeros(0)
    class_3 = np.zeros(0)

    for index, row in frame.iterrows():
        # print(row["class_num"])

        val = row["class_num"].split('[', 1)[1].split(']')[0]
        array = [float(x) for x in val[0:-1].split(',')]
        class_1 = np.append(class_1,array[0])
        class_2 = np.append(class_2, array[1])
        class_3 = np.append(class_3, array[2])


# path,index = plot_path(frame_probability,X_im,Y_im,"r")

t = np.linspace(0, len(X_robot), len(X_robot))


# plt.figure()
# plt.plot(t,Z_force)
#
# plt.title(" Z force")
#
#
# print("Z_force")
# print("MEAN:",np.mean(Z_force[:400]))
# print("STD",np.std(Z_force[:400]))
# print("Z_force")
# print("MEAN:",np.mean(Z_force[400:]))
# print("STD",np.std(Z_force[400:]))


# # print('z_FORCE',np.array(Z_force))
# Z_force_array = np.array(Z_force[:21])
# # print("Z_force_array",Z_force_array)
# Z_force_short = np.array(Z_force[21:])
# # print("Z_force_short",Z_force_short)
# last_average = np.zeros(0)
#
# for i in range(len(Z_force)-21):
#     move_average_online = np.convolve(Z_force_array, np.ones(20) / 20, mode='valid')
#     # print(move_average_online)
#      # print("Z_force_short[i]",Z_force_short[i])
#     Z_force_array = np.append(Z_force_array, Z_force_short[i])
#     last_average = np.append(last_average, move_average_online[-1])
#
# # move_average = np.convolve(Z_force, np.ones(20)/20, mode='valid')
# #
# # plt.figure()
# # plt.plot(t[:-19],move_average)
#
# plt.figure()
# plt.plot(t[:-20],move_average_online)
#
# plt.figure()
# plt.plot(t[:-21],last_average)

try:

    # stiffness = frame["stiffness"]
    # plt.figure()
    # plt.plot(t, stiffness)
    # plt.title(" Stiffness")

    # sum = 0
    # num =0
    # for s in stiffness:
    #     if s!=0:
    #         sum = sum+s
    #         num = num +1
    # print("stiffness mean: ", sum/num)

    # print("MEAN:", np.mean(stiffness))
    # print("STD", np.std(stiffness))

    X_filt = frame["x_filt"]
    Z_Force_filt = frame["force_curr_filt"]



    plt.figure()
    plt.plot(t, Z_Force_filt)
    plt.title(" Z force kalman filt")
    plt.ylim(min(Z_Force_filt),max(Z_Force_filt))

    velocity = frame["velocity_force"]
    plt.figure()
    plt.plot(t,velocity)
    plt.title("Velocity Force")

    velocity = frame["velocity_force"]
    plt.figure()
    plt.plot(t, velocity)
    plt.title("Velocity Force")

    ax = plt.gca()
    ax.set_aspect((320 / len(X_filt)) * 5)
    ax.plot(t, index)
    ax.set_xlabel('X_im_unfilt')
    ax.set_xlim(0, 640)

except:
    print("no X_filt")


delta_X = abs(X_im - X_mid)

# plt.figure()
# plt.plot(t,frame_probability)
# plt.title(" Probability")


if classes == True:
    plt.figure()
    ax1 = plt.subplot(6, 1, 1)
    # ax1.set_title("CNN probabilities")
    plt.xlabel('Frames', fontsize=8)
    plt.ylabel("Sacrum prob.", fontsize=8)
    ax1.plot(class_3)

    ax2 = plt.subplot(6, 1, 2)
    # ax2.set_title("Labels")
    plt.ylabel("Z_force", fontsize=8)
    plt.xlabel('Frames', fontsize=8)
    ax2.plot(Z_force)

    ax3 = plt.subplot(6, 1, 3)
    # ax2.set_title("Labels")
    plt.ylabel("Lumbar prob.", fontsize=8)
    plt.xlabel('Frames', fontsize=8)
    ax3.plot(class_2)

    ax4 = plt.subplot(6, 1, 4)
    # ax2.set_title("Labels")
    plt.ylabel("Thoracic prob.", fontsize=8)
    plt.xlabel('Frames', fontsize=8)
    ax4.plot(class_1)

    ax5 = plt.subplot(6, 1, 5)
    # ax2.set_title("Labels")
    plt.ylabel("Probability heatmap sacrum", fontsize=8)
    plt.xlabel('Frames', fontsize=8)
    ax5.plot(frame_probability_sacrum)

    ax6 = plt.subplot(6, 1, 6)
    # ax2.set_title("Labels")
    plt.ylabel("Probabilities heatmap", fontsize=8)
    plt.xlabel('Frames', fontsize=8)
    ax6.plot(frame_probability)








# plt.show()




plt.figure()
# plt.plot(timestamp,smooth(delta_X))
ax = plt.gca() #you first need to get the axis handle
ax.set_aspect((image_height/480)*0.08/len(X_robot)) #sets the height to width ratio to 1.5.
# plt.plot(smooth(-Y_robot),Z_robot)
ax.plot(smooth(-X_tcp),t)
ax.set_xlabel('X_tcp')
ax.set_ylabel('step')
ax.set_xlim(-0.08,0.08)

plt.figure()
# plt.plot(timestamp,smooth(delta_X))
ax = plt.gca() #you first need to get the axis handle
ax.set_aspect(3*0.08/len(X_robot)) #sets the height to width ratio to 1.5.
# plt.plot(smooth(-Y_robot),Z_robot)
ax.plot(smooth(-X_tcp),t)
ax.set_xlabel('X_tcp')
ax.set_ylabel('step')
ax.set_xlim(-0.08,0.08)




plt.figure()
# plt.plot(timestamp,smooth(delta_X))
ax = plt.gca() #you first need to get the axis handle
ax.set_aspect((image_height/480)*0.05/len(Z_tcp)) #sets the height to width ratio to 1.5.
# plt.plot(smooth(-Y_robot),Z_robot)
ax.plot(smooth(Z_tcp),t)
ax.set_xlabel('Z_tcp')
ax.set_ylabel('step')
ax.set_xlim(0,0.1)

plt.figure()
# plt.plot(timestamp,smooth(delta_X))
ax = plt.gca() #you first need to get the axis handle
ax.set_aspect(3*0.05/len(Z_tcp)) #sets the height to width ratio to 1.5.
# plt.plot(smooth(-Y_robot),Z_robot)
ax.plot(smooth(Z_tcp),t)
ax.set_xlabel('Z_tcp')
ax.set_ylabel('step')
ax.set_xlim(0,0.1)

#
# plt.figure()
# # plt.plot(timestamp,smooth(delta_X))
# ax = plt.gca() #you first need to get the axis handle
# ax.set_aspect((320 / len(X_filt)) * 5) #sets the height to width ratio to 1.5.
# # plt.plot(smooth(-Y_robot),Z_robot)
# ax.plot(smooth(X_filt),t)
# ax.set_xlabel('X_filt')
# ax.set_ylabel('step')
# ax.set_xlim(0,640)

# plt.plot(X_robot, Y_robot)

# plt.plot(-smooth(X_robot),t)

# plt.plot(t,smooth(undrift(Y_force)))
# plt.title(" Y force")
#
# plt.figure()
#
# plt.plot(t,X_force)
# plt.title(" X force")
#









plt.show()