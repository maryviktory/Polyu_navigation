import os
import logging
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from datetime import datetime
from scipy.interpolate import interp1d
import pandas as pd

def interpolation(y,z):
    eq_spaced_y = np.linspace(y[0], y[-1], len(y))
    f = interp1d(y, z)
    z_interp = f(eq_spaced_y)
    return z_interp

# TODO: add documentation for these function
def find_valid_indexes(y):
    smoothed_vel = np.squeeze(smooth(np.diff(y), 10))

    max_vel_pos = np.argmax(smoothed_vel)
    max_vel = smoothed_vel[max_vel_pos]
    first_half = np.flip(smoothed_vel[0:max_vel_pos])

    try:
        p1 = max_vel_pos - np.where(first_half<(0.2*max_vel))[0][0]

        second_half = smoothed_vel[max_vel_pos::]
        p2 = max_vel_pos + np.where(second_half < (0.2 * max_vel))[0][0]
    except:
        return 0, len(smoothed_vel) - 1, smoothed_vel

    return p1, p2, smoothed_vel


def smooth(signal, N=30):
    N = 2  # Filter order
    Wn = 0.05  # Cutoff frequency
    B, A = butter(N, Wn, output='ba')
    # Second, apply the filter
    filt = filtfilt(B, A, signal)
    return filt


def undrift(signal):
    N = 2  # Filter order
    Wn = 0.01  # Cutoff frequency
    B, A = butter(N, Wn, output='ba')
    # Second, apply the filter
    avg = filtfilt(B, A, signal)
    undrifted = signal - avg

    return undrifted

def normalize(z_interp):
    z_normalized = (z_interp - min(z_interp)) / (max(z_interp) - min(z_interp))

    return z_normalized

