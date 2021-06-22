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

folder = "human experiments\Ho YIN\\1\FCN"
data_path = "D:\spine navigation Polyu 2021\\robot_trials_output\%s\HO_YIN_long_model_20.csv"%folder
frame = pd.read_csv(data_path)

# 'GAP Prob',"Sacrum Prob", "Lumbar Prob","Thoracic Prob"
class_1 = np.array(frame["Heatmap Prob"])
class_2 = frame["Sacrum Prob"]
class_3 = frame["Lumbar Prob"]
class_4 = frame["Thoracic Prob"]

# print(class_1)

ax1 = plt.subplot(4, 1, 1)
# ax1.set_title("CNN probabilities")
plt.xlabel('Frames', fontsize=8)
plt.ylabel("Sacrum prob.", fontsize=8)
ax1.plot(class_2)

ax2 = plt.subplot(4, 1, 2)
# ax2.set_title("Labels")
plt.ylabel("Gap prob", fontsize=8)
plt.xlabel('Frames', fontsize=8)
ax2.plot(class_1)

ax3 = plt.subplot(4, 1, 3)
# ax2.set_title("Labels")
plt.ylabel("Lumbar prob.", fontsize=8)
plt.xlabel('Frames', fontsize=8)
ax3.plot(class_3)

ax4 = plt.subplot(4, 1, 4)
# ax2.set_title("Labels")
plt.ylabel("Thoracic prob.", fontsize=8)
plt.xlabel('Frames', fontsize=8)
ax4.plot(class_4)

plt.show()