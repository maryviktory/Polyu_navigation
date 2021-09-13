import pandas as pd
# from scipy.spatial.transform import Rotation as R
import numpy as np
import cv2
import math3d as m3d

#Code to find the transform between setups
# B=M^T*A*M


csys = m3d.Transform()
def get_points_trans(position):
    """
    get #current# transform from base to to tcp
    """
    trans = csys.inverse * m3d.Transform(position)
    return trans


pd_frame = pd.DataFrame(columns=["timestamp","X_im","Y_im","Frame_Probability","X_tcp","Y_tcp","Z_tcp","X", "Y", "Z","Rx","Ry","Rz","Fx","Fy","Fz","Mx","My","Mz","x_filt","velocity_im","velocity_force","force_curr_filt",
                                         "stiffness", "class_num","Frame_Probability_sacrum","X_sac","Y_sac"])


save_frame = "E:\spine navigation Polyu 2021\\robot_trials_output\output_high_fps_rotated.csv"
data_path = "E:\spine navigation Polyu 2021\\robot_trials_output\output_high_fps.csv"
frame = pd.read_csv(data_path)


A_setup_pose = [0.785589932,0.03341791,	-0.254397932,	1.237330832,	1.159732788,	1.191409569]

# B setup pose
X = frame["X"]
Y = frame["Y"]
Z = frame["Z"]
Rx = frame["Rx"]
Ry = frame["Ry"]
Rz = frame["Rz"]




m4_A = np.zeros((4,4))
m4_B = np.zeros((4,4))
m4_rotate = np.zeros((4,4)) #we want to find it
# m4_rotate[:3,:3] = [ -1.0,0.0,0.0],[ 0.0,-1.0,0.0],[ 0.0,0.0,1]
# m4_rotate[:3,3] = 0,0,0
# m4_rotate[3,3] = 1

tr = get_points_trans(A_setup_pose)
R = tr.orient.array
t = tr.pos.array
m4_A[:3,:3] = R
m4_A[:3,3] = t
m4_A[3,3] = 1



rotVec = np.zeros((1, 3), np.float32)
for p in range(0,len(X)-1):

    position = [X[p], Y[p], Z[p], Rx[p], Ry[p], Rz[p]]
    tr = get_points_trans(position)
    R = tr.orient.array
    t = tr.pos.array
    m4_B[:3,:3] = R
    m4_B[:3,3] = t
    m4_B[3,3] = 1

    # result_m4 = np.dot(m4_rotate.transpose(),m4)
    # result_m4 = np.dot(result_m4,m4_rotate)
    # # print(result_m4)
    # rotVec, _ = cv2.Rodrigues(result_m4[:3,:3],rotVec)
    # k = np.append(result_m4[:3,3],rotVec.T)

#     pd_frame = pd_frame.append(
#         {'timestamp': 0, "X_im": 0, "Y_im": 0, "Frame_Probability": 0, "X_tcp": 0,
#          "Y_tcp": 0,
#          "Z_tcp": 0, 'X': k[0], 'Y': k[1], 'Z': k[2],
#          'Rx': k[3],
#          'Ry': k[4], 'Rz': k[5], "Fx": 0, "Fy": 0,
#          "Fz": 0,
#          "Mx": 0, "My": 0, "Mz": 0, "x_filt": 0, "velocity_im": 0,
#          "velocity_force": 0, "force_curr_filt": 0, "stiffness": 0,
#          "class_num": [0, 0, 0], "Frame_Probability_sacrum": 0,
#          "X_sac": 0, "Y_sac": 0}, ignore_index=True)
#
# pd_frame.to_csv(save_frame, header=False)







