import math
import GUI.Scolioscan_robotics_GUI_qt5 as GUI
import sys
import time
import torch.multiprocessing as mp
from multiprocessing import Process
from multiprocessing.queues import SimpleQueue
import multiprocessing
from PIL import ImageGrab
import random
from threading import Thread
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import QObject,pyqtSignal
from PyQt5.QtGui import QImage
import socket
import urx
from utils.config_robot_GUI import config
import utils as utils
import websocket
import json
import os
import logging
import pandas as pd
from multiprocessing.managers import BaseManager
from Spine_navigation_Polyu.Multiprocess_FCN import FCN_Thread,get_image_fun
from Spine_navigation_Polyu.utils.functions import save_video
# import mss
# import mss.tools

#################################_____PARAMETERS to adjust_____##########################
logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.info("STARTING SCOLIOSCAN ROBOTIC SOFT")

# fheight =1080
# fwidth = 1920
#
# print(fwidth, fheight)
# fourcc = cv2.VideoWriter_fourcc(*'XVID')
# out = cv2.VideoWriter('output.avi', fourcc, 20.0, (fwidth, fheight))

# out = cv2.VideoWriter('output_original.avi', fourcc, 3.0, (1280, 480))  # for images of size 480*640


# w = 400; hg = 360

'''transformation matrix to easily swap the coordinate system. 
Originally the axis in the code are the following in the base system: z - to the right hand side (image control), x - to the skin surface (force control),
y - up. To swap it to the other direction, we need to multiply it with transformation matrix
 [V_robx, V_roby, V_robz]= [-1 0 0; 0 -1 0; 0 0 1] * [v_force, v_up, v_im] The transformation matrix should be
 picked to match the axis of the robot base coordinate system
 V_robot = mp.matmul(T_marix, V_control)
 
 tool coordinate system (preferably to use): z - to the surface, x - left hand side if face the robot, y - down
 [V_robx, V_roby, V_robz]= [0 0 1; 0 -1 0; 1 0 0] * [v_force, v_up, v_im]
 '''


# T_force = -1
# T_up = -1
# T_im = 1
if config.MODE.BASE_csys == True:
    T_matrix = [[1,0,0],[0,0,1],[0,1,0]]
    control_force = [1,0,0] #force is first one in [v_force, v_up, v_im]
    force_s = np.matmul(T_matrix,control_force)
    force_slot = int(np.argwhere(np.matmul(T_matrix,control_force)!=0))
    # print("force axis",force_slot)
    T_force = int(force_s[force_slot])
    # print(force_s,force_slot, T_force)
    V_control = np.zeros(3)
    V_robot = np.zeros(3)
    axis_up_base = 1 #y
    axis_force_base = 0 #X
    axis_image_base = 2 #Z

else:
    T_matrix = [[0,0,1],[0,-1,0],[1,0,0]]
    control_force = [1,0,0]#force is first one in [v_force, v_up, v_im]
    force_s = np.matmul(T_matrix, control_force)
    force_slot = int(np.argwhere(np.matmul(T_matrix, control_force) != 0))

    print("force axis",force_slot)
    T_force = int(force_s[force_slot])

    V_control = np.zeros(3)
    axis_up = 0 #X
    axis_force = force_slot #X
    axis_image = 1 #Z

#TODO: write everything from one experiment to csv file

csv_file_name = ('results/%sPosition_plot_Fref_%s_Kf_%s_Kz_%s_thr_%s.csv'%(config.Trajectory_n,config.FORCE.Fref,config.FORCE.Kf,config.FORCE.Kz,config.FORCE.thr))

i = 0
j = 0
k = 0
h = 0
temp = []
position = []
full_position = []
corrected_position = []

stopper = None

#########################################################


# class for GUI
class Window(QtWidgets.QDialog, GUI.Ui_Dialog):

    def __init__(self):
        QtWidgets.QDialog.__init__(self)
        self.setupUi(self)
        self.center()
        ###########_____Butttons___________#############
        self.Button_freedrive.clicked.connect(self.robot_freedrive_able)
        self.Button_start.clicked.connect(self.set_start_point)
        self.Button_stop.clicked.connect(self.set_last_point)
        self.toolButton.clicked.connect(self.FileBrowse)
        self.Button_move.clicked.connect(self.start_on_click)
        self.Button_stop_move.clicked.connect(self.stop_robot_move)
        self.Radio_button_plot.toggled.connect(self.Plot_abled)

        ###########__________INIT_________##############
        if config.MODE.Develop == False:
            self.robot = urx.Robot(config.IP_ADRESS, use_rt=True)
            utils.reset_FT300(self.robot)
        else:
            self.robot = None

        print(self.robot)
        #TODO: Don't use Queue if you need to get the last value. use Value or Array
        self.last_record_point = None
        self.first_record_point = mp.Array("f",range(6))
        self.first_record_point_var = None
        self.file_exists = False
        self.move_end_distance = mp.Value('f', 0)
        self.threads_stopper = mp.Value('i', False)  # False
        self.robot_stopper = mp.Value('i', False)
        self.q_Force = mp.Value("f",0)
        self.q_Full_Force = mp.Array("f",range(6))




# alignment of Window
    def center(self):
        frameGm = self.frameGeometry()
        screen = QtWidgets.QApplication.desktop().screenNumber(QtWidgets.QApplication.desktop().cursor().pos())
        centerPoint = QtWidgets.QApplication.desktop().screenGeometry(screen).center()
        frameGm.moveCenter(centerPoint)
        self.move(frameGm.topLeft())

    def robot_freedrive_able(self):
        self.robot.set_freedrive(True, timeout=60)

    def robot_freedrive_disable(self):
        self.robot.set_freedrive(False, timeout=60)

    # call of functions while clicking the main button "start"


    def stop_robot_move(self):

        self.threads_stopper.value = 1
        time.sleep(0.2)
        self.Force_Thread.terminate()
        self.Move_Thread.terminate()
        time.sleep(1)
        if not self.Force_Thread.is_alive():
            print("[MAIN]: WORKER is a goner")
            self.Force_Thread.join(timeout=0.1)
            self.Move_Thread.join(timeout=0.1)
            os._exit(0)

    def set_start_point(self):
        # self.Record_thread.start()
        self.first_record_point_var = self.robot.getl()
    def set_last_point(self):
        # self.Record_thread.stopper = True
        self.last_record_point = self.robot.getl()

    def FileBrowse(self):
        self.file_exists = True
        filePath = QtWidgets.QFileDialog.getOpenFileName(self,
                                                     'Single File',
                                                     "~/Desktop/PyRevolution/PyQt5",
                                                     '*.txt')[0]

        self.label_trajectory_name.setText(filePath[-19:])

        fileHandle = open(filePath, 'r')
        #lines = fileHandle.readlines()
        self.lines = [line.rstrip('\n') for line in fileHandle]
        print(self.lines)

    def Plot_abled(self):
        self.Plot_Thread.radio_button_enabled = True

    def Initialize_file(self):

        m = 0
        position_from_file = np.zeros(0)
        while m < len(self.lines) - 1:
            val = self.lines[m].split('[', 1)[1].split(']')[0]
            val2 = ([float(x) for x in val[0:-1].split(',')])  # list comprehension
            np.append(position_from_file,val2)

            m += 1
        first_record_point = position_from_file[0]
        last_record_point = position_from_file[-1]
        return first_record_point, last_record_point

    def start_on_click(self):


###########____INIT first, last points and distance to move_______###########
        current_pose = self.robot.getl()
        if self.file_exists == True:
            self.first_record_point_var, self.last_record_point = self.Initialize_file()
            self.first_record_point = self.first_record_point_var
            # self.last_record_point = last_record_point

        if self.first_record_point_var is None:
            logger.info("No first point specified. Accept current pose as init")
            print("No first point specified. Accept current pose as init")
            self.first_record_point = current_pose
            self.first_record_point_var = current_pose


        if self.move_end_distance.value == 0:
            if self.last_record_point is None:
                self.move_end_distance.value = config.default_distance
            else:
                self.move_end_distance.value = self.last_record_point[axis_up] - self.first_record_point_var[axis_up] #axis_up = Y
        else:
            self.move_end_distance.value = float(self.lineEdit.text())

        if self.move_end_distance.value > config.maximum_distance:  # in meters
            self.move_end_distance.value = config.maximum_distance


        print("move_end_distance", self.move_end_distance.value)
        print("first_record_point",self.first_record_point[:])

        # print(self.first_record_point)

###################_______INIT_PROCESSES____##############################

        # utils.reset_FT300(self.robot)
        self.robot.close() #Can't pass robot to Process, so I have to init it once again inside the process
        time.sleep(0.2)


        self.Move_Thread = Move_Thread(self.q_Full_Force, self.q_Force, self.move_end_distance,
                                   self.first_record_point, self.threads_stopper)
        self.Move_Thread.daemon = True
        self.Move_Thread.start()

        self.Force_Thread = Force_Thread(self.q_Full_Force ,self.q_Force,self.threads_stopper)
        self.Force_Thread.daemon = True
        self.Force_Thread.start()

        #TODO: Add FCN
        # if config.IMAGE.FCN == True:
            # self.FCN_thread.start()
            # self.FCN_thread.FCN_run()
        # self.Image_thread.start()
        # self.Image_thread.PIX.connect(self.setImage)


    def runn(self):
        print("New function")

    def setImage(self, image):
        self.label_3.setPixmap(QtGui.QPixmap.fromImage(image))

# even when red cross is pressed - two windows are closing
    def closeEvent(self, QCloseEvent):
        QCloseEvent.accept()

#############_main loop_#############


class Force_Thread(Process):
    '''Force sensor has different coordinate system, it reads measurements in tool coordintate system,
        so that when it touches the surface perpendicular to the tool, it reads the Fz

        To get exact forces in the robot coordinate system, it is better to multiply Forces from the Force sensor with
        the transformation matrix from the robot, Universal robot can provide it at each moment of time.

        To symplify the task just the proportion between the Fz and Fset is used to generate the velocity
        for the robot control. ?Velocity should be set also in the tool frame? Otherwise the axis of the robot
        and the force sensor should be aligned during robot manipulation

        Now is (check sensor and base alignment):
        x robot - z force sensor
        y robot - x force sensor
        z robot - y force sensor

        #TODO: To change it the transformation matrix should be introduced.
        '''
    def __init__(self,q_Full_Force ,q_Force,threads_stopper):
        Process.__init__(self)
        self.Fz = 0
        self.Fdelta = 0
        self.Rx_next = 0
        self.Mx_raw = 0
        # self.point_x_next = 0
        # self.force_tread_stopper = False
        # self.stop_movement = False
        self.num = 0
        self.q_Force = q_Force
        self.q_Full_Force = q_Full_Force
        self.threads_stopper = threads_stopper

    def run(self):
        i = 0
        # if config.MODE.Develop == False:
        #     self.robot = urx.Robot(config.IP_ADRESS, use_rt=True)
        # else:
        #     self.robot = None

        print("Force Process starts")
        # print("robot Force", self.robot)
        if config.MODE.Develop == False:
            # robot = urx.Robot(config.IP_ADRESS, use_rt=True)
            s = socket.socket(
                socket.AF_INET, socket.SOCK_STREAM)

            s.connect((config.IP_ADRESS, 63351))
            logger.info('Socket is connected')
        else:
            logger.info('Ethernet is not connected')
            s = None
            # os._exit(0)

        F = 0
        time_start = time.time()
        while bool(self.threads_stopper.value) == False:

            # if i == 150:
            #    Fref = -6
            #
            # if i == 300:
            #    Fref = -3

            # if i == 1000:
            #     self.stop_movement = True


            tstart = time.time()
            response = s.recv(4096)
            # print("socket response",response)
            # response = bytearray(response)
            val = response.decode("ascii").split('(', 1)[1].split(')')[0]
            array = [float(x) for x in val[1:-1].split(',')]

            # print ('Fx:', array[0], 'Fy:', array[1],'Fz:', array[2],'Mx:', array[3], 'My:', array[4], 'Mz:', array[5])
            #print ('      ')
            self.F_full = array
            self.F_changed_order = [array[2], array[0], array[1],array[5],array[3], array[4]]
            self.Fz = array[2] # robot X axis, contr directed
            self.Fy = array[1] # robot Z axis
            self.Fx = array[0] # robot Y axis
            self.Mx_raw = array[3]
            self.My = array[4]
            self.Mz = array[5]
            #print array[3],array[4], array[5]
            self.Fdelta = config.FORCE.Fref - math.fabs(self.Fz)

            if self.Fz > config.FORCE.Fmax:
                self.Fz_sat = config.FORCE.Fmax
            elif self.Fz < -config.FORCE.Fmax:
                self.Fz_sat = -config.FORCE.Fmax
            else:
                self.Fz_sat = self.Fz
            # print("Fz_sat", self.Fz_sat)

            #SEND to other threads
            self.q_Force.value = self.Fz_sat
            self.q_Full_Force[:] = self.F_full
            # posa = self.robot.getl()
            # point_x = posa[0]
            # Rx_current = posa[3]
            #print "X", point_x
            ######## F - total force

            #F = ((self.Fx) ** 2 + (self.Fy) ** 2 + (self.Fz) ** 2) ** 1 / 2





            ################ velocity control #################

            #TODO: move it to Move_Thread
            # self.point_x_next = point_x + config.FORCE.K_delta * (math.fabs(self.Fz_sat) - config.FORCE.Fref)

            # print(self.velocity_x )
            #self.velocity_x = Kf * (F - Fref)
            #print 'F:', F

            ####################################################
            if self.Mx_raw < -0.8:
                self.Mx = -0.8
            elif self.Mx_raw > 0.8:
                self.Mx = 0.8
            else:
                self.Mx = self.Mx_raw

            Krx = 0.03
            self.velocity_Rx = -Krx*(self.Mx_raw)


            # if self.Mx < -0.2:
            #     self.Rx_next = Rx_current - 0.1*(math.fabs(self.Mx) - 0.1)
            #     R1 = 0.1 * (math.fabs(self.Mx) - 0.1)
            #     #print('change: - ', R1)
            # elif self.Mx > 0.2:
            #     self.Rx_next = Rx_current + 0.1*(math.fabs(self.Mx) - 0.1)
            #     R2 = 0.1 * (math.fabs(self.Mx) - 0.1)
            #     #print('change: + ', R2)

            # else:
            #     self.Rx_next = Rx_current

            #print "X+", self.point_x_next
            tstop = time.time()
            #print ((tstop - tstart))
            #TODO: init robot for force function
            if self.Fz > config.FORCE.Fcrit:
                # self.robot.stop()
                self.threads_stopper.value = 1 #stop threads
                print('Value is more than F critical. Throw threads_stopper flag')

            i = i + 1
            # print(self.num)

            self.num = self.num +1

        else:
            print("Force thread is stopped")
            time_thread = time.time() - time_start
            fps_im = self.num / time_thread
            print("fps Force thread {}, average time per cycle {}".format(fps_im, 1 / fps_im))
            #print 'i:', i


class Move_Thread(Process):
    def __init__(self, q_num,q_Force,move_end_distance,first_record_point,threads_stopper):
        Process.__init__(self)

        self.num = 0
        self.q_Force = q_Force
        self.threads_stopper = threads_stopper

        self.move_end_distance = move_end_distance
        self.first_record_point = first_record_point
        self.q_Full_Force = q_num
        self.patient = "phantom"
    def run(self):

        print("Move Process starts")
        print("move end distance", self.move_end_distance.value)
        if config.MODE.Develop == False:
            self.robot = urx.Robot(config.IP_ADRESS, use_rt=True)
            # self.robot.set_tcp((0, 0, 0.1, 0, 0, 0))
            # self.robot.set_payload(2, (0, 0, 0.1))
            utils.reset_FT300(self.robot)
        else:
            self.robot = None

        time.sleep(0.2) #to wait until robot gets reset_FT300 comand


        print("go to first point")
        self.move_first_point()

        Pose = self.robot.getl()
        # robot_pose_array = Pose
        start_movement_Y = Pose[1]
        num = 0
        T_b_init = self.robot.get_pose()
        time_start = time.time()

        if config.MODE.FCN == True:
            pd_frame = pd.DataFrame(columns=["timestamp","X_im","Y_im","X_tcp","Y_tcp","Z_tcp","X", "Y", "Z","Rx","Ry","Rz","Fx","Fy","Fz","Mx","My","Mz"])
        else:
            pd_frame = pd.DataFrame(
                columns=["timestamp", "X_tcp", "Y_tcp", "Z_tcp", "X", "Y", "Z", "Rx", "Ry", "Rz", "Fx", "Fy", "Fz",
                         "Mx", "My", "Mz"])


        while bool(self.threads_stopper.value) == False:

            T_point = self.robot.get_pose()
            pos_curr = T_point.pose_vector.tolist()

            force_curr = self.q_Force.value
            force_curr_full = self.q_Full_Force[:]
            pos_y_curr = pos_curr[1]

            #Transformation from initial TCP position to the following points to get the coordinates in initial TCP ccordinate system
            # T_point_inbase = T_tcp_pose_init_inbase*T_point_intcp
            #T_point_intcp = T_tcp_pose_init_inbase^(-1) * T_point_inbase
            T_points_intcp = np.around(np.linalg.inv(T_b_init.matrix) * T_point.matrix, 4)
            # TODO: record information (time,robot pose, force, image, image coordinates) to csv file
            # TODO: how to record positions, in base coordinates? or somehow find tool coordinates

            if config.MODE.FCN == True:
                pd_frame = pd_frame.append({'timestamp': time.time(), "X_tcp":T_points_intcp[0,3],"Y_tcp":T_points_intcp[1,3],"Z_tcp":T_points_intcp[2,3],'X': pos_curr[0],'Y': pos_curr[1],'Z': pos_curr[2],'Rx': pos_curr[3],
                                        'Ry': pos_curr[4],'Rz': pos_curr[5], "Fx": force_curr_full[0],"Fy": force_curr_full[1],"Fz": force_curr_full[2],
                                        "Mx": force_curr_full[3],"My": force_curr_full[4],"Mz": force_curr_full[5]}, ignore_index=True)


            else:
                pd_frame = pd_frame.append(
                    {'timestamp': time.time(), "X_tcp": T_points_intcp[0, 3], "Y_tcp": T_points_intcp[1, 3],
                     "Z_tcp": T_points_intcp[2, 3], 'X': pos_curr[0], 'Y': pos_curr[1], 'Z': pos_curr[2],
                     'Rx': pos_curr[3],
                     'Ry': pos_curr[4], 'Rz': pos_curr[5], "Fx": force_curr_full[0], "Fy": force_curr_full[1],
                     "Fz": force_curr_full[2],
                     "Mx": force_curr_full[3], "My": force_curr_full[4], "Mz": force_curr_full[5]}, ignore_index=True)

            vel_up = config.VELOCITY_up
            vel_im = 0.0

            if config.MODE.FORCE == True:
                self.velocity_x = -config.FORCE.Kf * (-force_curr - config.FORCE.Fref)

                vel_force = self.velocity_x
            else:
                vel_force = 0.0

            V_control[0] = vel_force
            V_control[1] = vel_up
            V_control[2] = vel_im
            # if math.fabs(pos_y_curr - y_position_target) < 0.2:
            #     vel_up = -Kz * (pos_y_curr - y_position_target)
            #     V_control[1] = vel_up
            # if math.fabs(pos_z_curr - z_position_target) < 0.2:
            #     vel_im = -Kz * (pos_z_curr - z_position_target)
            #     V_control[2] = vel_im
            #vel_Rx = self.force.velocity_Rx
            #print(vel_z, vel_y)


            V_robot = np.matmul(T_matrix,V_control)
            # print(V_robot)
            # print(self.q_Force.value)
            # print("GO GO GO")

            # vel_x = vel_x*T_force
            # vel_y = vel_y*T_up
            # vel_z = vel_z*T_im
            if config.MODE.BASE_csys == True:
                program = 'speedl([%s,%s,%s,0,0,0],0.05,0.5)' % (V_robot[0], V_robot[1], V_robot[2])
            #print program
                self.robot.send_program(program)
            else:
                velocities = [V_robot[0], V_robot[1],V_robot[2], 0.0, 0.0, 0.0]
                min_time = 0.5
                self.robot.speedl_tool(velocities, config.FORCE.a, min_time)

            # if self.force.stop_movement:
            #     self.plotme = True
            #     break

            self.num = self.num+1
            # print("move end distance",self.move_end_distance.value)
            if math.fabs(pos_y_curr - start_movement_Y)> self.move_end_distance.value:
                # self.stop_video = True
                # self.plotme = True
                self.threads_stopper.value = 1 #Stop other threads
                print('Robot finished {} m'.format(self.move_end_distance))
                time_thread = time.time() - time_start
                if time_thread !=0:
                    fps_im = self.num / time_thread
                    print("fps Move thread {}, average time per cycle {}".format(fps_im, 1 / fps_im))

                while os.path.exists("Move_thread_output%s.csv" % num):
                    num = num + 1
                # print(robot_pose_array[:, 0])
                pd_frame.to_csv(os.path.join(config.IMAGE.SAVE_PATH, "Move_thread_output%s.csv" % num))

                self.robot.close()
                # robot.close()
                # s.close()
                break


            # if self.target_pose.stop_flag_for_robot_move == True:
            #     self.plotme = True
            #     print('Robot finished')
            #     time_thread = time.time() - time_start
            #     fps_im = self.num / time_thread
            #     print("fps Move thread {}, average time per cycle {}".format(fps_im, 1 / fps_im))
            #     self.robot.close()
            #     # s.close()
            #     break
        else:
            print('robot forced stop')
            # self.plotme = True
            time_thread = time.time() - time_start
            if time_thread and self.num != 0:
                fps_im = self.num / time_thread
                print("fps Move thread {}, average time per cycle {}".format(fps_im, 1 / fps_im))

            while os.path.exists("Move_thread_output%s.csv" % num):
                num = num + 1
            # print(robot_pose_array[:, 0])
            pd_frame.to_csv(os.path.join(config.IMAGE.SAVE_PATH, "Move_thread_output%s.csv" % num))

            # self.robot.stopj()
            # self.robot.close()

            #robot.stopj()

    def move_first_point(self):

        # print(self.first_record_point.get())
        first_pose = self.first_record_point[:]
        print(first_pose)

        first_pose[0] += 0.002


        # robot.movep(self.position[0], acc=a, vel=v, wait=True, relative=False, threshold=None)
        self.robot.movep(first_pose, acc=config.FORCE.a, vel=config.FORCE.v, wait=True, relative=False, threshold=None)
        print('robot in 0 position')
        move_first = True
        while move_first ==True:
            # print("GO")
            if bool(self.threads_stopper.value) == False:
                Fz_1 = self.q_Force.value
                # print("force: ",self.q_num[:])

                # print("Fz from first move",Fz_1)

                if math.fabs(Fz_1) < config.FORCE.Fref_first_move:
                    # self.position[0][0] += -0.0002
                    # robot.movep(self.position[0], acc=a, vel=v, wait=False, relative=False, threshold=None)
                    #print "move to first position"

                    self.robot.z_t += 0.002*T_force
                    # self.robot.movep(first_pose, acc=config.FORCE.a, vel=config.FORCE.v, wait=False, relative=False, threshold=None)
                    if config.MODE.BASE_csys == True:
                        first_pose[force_slot] += 0.002*T_force
                        self.robot.movel_tool(first_pose, acc=config.FORCE.a, vel=config.FORCE.v, wait=False, relative=False,
                                     threshold=None)
                    else:
                        self.robot.z_t += 0.002 * T_force

                else:
                    print('Force value reached')
                    move_first = False
                    # self.start_plotting = True
                    # self.start_recording = True
                    # self.Force_value_reached = True
                    # St = False
                    # self.robot.stopj()
                    break
            else:
                logger.info("Robot move stopped from function: move_first_point")
                move_first = False
# class Image_thread(QtCore.QThread):
#     changePixmap = pyqtSignal(QImage)
#     PIX = QtCore.pyqtSignal(QtGui.QImage)
#
#     def __init__(self, Pose_thread, Force_thread, Move_thread):
#         QtCore.QThread.__init__(self)
#
#         self.move_thread = Move_thread
#         self.target_pose = Pose_thread
#
#         self.force = Force_thread
#         self.num = 0
#
#
#     def run(self):
#         i = 0
#         fourcc = cv2.VideoWriter_fourcc(*'XVID')
#         out = cv2.VideoWriter('output.avi', fourcc, 20.0, (fwidth, fheight))
#         time_start = time.time()
#         while True:
#             if self.move_thread.start_recording == True:
#
#         # # ****************************************
#         #         while (True):
#                 img = ImageGrab.grab(
#                     bbox=(0, 0, fwidth, fheight))  # bbox specifies specific region (bbox= x,y,width,height)
# #40,75
#                 frame = np.array(img)
#                 # frame = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)
#                 position_video = robot.getl()
#                 Force = self.force.F_changed_order
#                 f3.write(repr(position_video + Force) + '\n')
#
#
#                 cv2.waitKey(100)
#                 rgbImage = frame
#                 #rgbImage = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#                 convertToQtFormat = QImage(rgbImage.data, rgbImage.shape[1], rgbImage.shape[0], QImage.Format_RGB888)
#                 #p = convertToQtFormat.scaled(401, 351)
#                 p = convertToQtFormat.scaled(w, hg)
#
#                 self.PIX.emit(p)
#
#
#                 #cv2.imshow("test", rgbImage)
#                 out.write(rgbImage)  # write output file of changed frames
#
#                 # make conditions when the program is switched off
#                 # if cv2.waitKey(1) & 0xFF == ord('q'):
#                 #     break
#                 #key = cv2.waitKey(1)
#                 self.num = self.num + 1
#
#             if self.target_pose.stop_flag_for_robot_move == True:
#                  print('video stopped')
#                  print("Image thread is stopped")
#                  time_thread = time.time() - time_start
#                  fps_im = self.num / time_thread
#                  print("fps Image thread {}, average time per cycle {}".format(fps_im, 1 / fps_im))
#                  f3.close()
#                  out.release()
#                  break
#             i += 1
#
#
#             #         if key == 27:
#         # break
#
#         # Release everything if job is finished
#         #
#         cv2.destroyAllWindows()
#         sys.exit()
#

def main():
    try:
        app = QtWidgets.QApplication(sys.argv)
        form = Window()
        form.show()
        app.exec_()
    except KeyboardInterrupt:
        print("Killed")
        # f3.close()
        # out.release()


# launch GUI function
if __name__ == '__main__':
    main()

