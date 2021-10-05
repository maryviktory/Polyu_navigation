import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, medfilt
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

folder = "human experiments\Ho YIN\\1\CNN"
data_path ="E:\spine navigation Polyu 2021\\robot_trials_output\human experiments\Long Yin Lam\\3\Long_yin_lam_model_20.csv"
frame = pd.read_csv(data_path)

# 'GAP Prob',"Sacrum Prob", "Lumbar Prob","Thoracic Prob"
class_1 = frame["GAP Prob"]
class_2 = frame["Sacrum Prob"]
class_3 = frame["Lumbar Prob"]
class_4 = frame["Thoracic Prob"]


ax1 = plt.subplot(5, 1, 1)
# ax1.set_title("CNN probabilities")
plt.xlabel('Frames', fontsize=8)
plt.ylabel("Sacrum prob.", fontsize=8)
ax1.plot(class_2)

ax2 = plt.subplot(5, 1, 2)
# ax2.set_title("Labels")
plt.ylabel("Gap prob", fontsize=8)
plt.xlabel('Frames', fontsize=8)
ax2.plot(class_1)

ax3 = plt.subplot(5, 1, 3)
# ax2.set_title("Labels")
plt.ylabel("Lumbar prob.", fontsize=8)
plt.xlabel('Frames', fontsize=8)
ax3.plot(class_3)

# ax4 = plt.subplot(5, 1, 4)
# # ax2.set_title("Labels")
# plt.ylabel("Thoracic prob.", fontsize=8)
# plt.xlabel('Frames', fontsize=8)
# ax4.plot(class_4)

mult = class_2-class_1
N=21
move_average_online = np.convolve(mult, np.ones(N) / N, mode='valid')
# log = np.log(class_2)

ax5 = plt.subplot(5, 1, 5)
# ax2.set_title("Labels")
plt.ylabel("Thoracic prob.", fontsize=8)
plt.xlabel('Frames', fontsize=8)
ax5.plot(move_average_online)

resulted_array = np.zeros(len(move_average_online))
for i,element in enumerate(move_average_online):
    n = 100
    current_data = move_average_online[0:i]

    if i>n and i>200 and np.mean(current_data[-n:]) < 0.3:
        resulted_array[i] = 0.07
    else:
        resulted_array[i] = 0.04

ax4 = plt.subplot(5, 1, 4)
# ax2.set_title("Labels")
plt.ylabel("resulted_array.", fontsize=8)
plt.xlabel('Frames', fontsize=8)
ax4.plot(resulted_array)


plt.show()