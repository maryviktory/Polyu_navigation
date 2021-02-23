import math
from Spine_navigation_Polyu.GUI.Scolioscan_robotics_big_GUI_Q4 import Ui_Dialog as GUI
import sys
import time
from PIL import ImageGrab
import random
from threading import Thread
import numpy as np
import matplotlib.pyplot as plt
from PyQt4 import QtCore, QtGui
from PyQt4.QtCore import pyqtSignal
from PyQt4.QtGui import QImage
import socket
import urx
import cv2
from scipy import ndimage
from Spine_navigation_Polyu.utils.phasesym import phasesym
from pydispatch import dispatcher
from Spine_navigation_Polyu.utils.config import config
import Spine_navigation_Polyu.utils as utils


#################################_____PARAMETERS to adjust_____##########################

fheight =1080
fwidth = 1920
#
#
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output_original.avi', fourcc, 20.0, (640,640))
out_stack = cv2.VideoWriter('output_stack.avi', fourcc, 20.0, (640, 320))
################____Imaging parameters____################
kernel = np.ones((5,5),np.uint8)
kernel_opening = np.ones((10,10),np.uint8)
kernel_closing = np.ones((10,10),np.uint8)

# FOV = 0.045 # field of view of the probe 45mm in meters

# alpha = 0.5

Fref = config.FORCE.Fref
Kf = config.FORCE.Kf
Kz = config.FORCE.Kz
thr = config.FORCE.thr
Trajectory_n = config.Trajectory_n
###############################

file_name1 = ('results/%sPosition_plot_Fref_%s_Kf_%s_Kz_%s_thr_%s.txt'%(Trajectory_n,Fref,Kf,Kz,thr))

file_name2 = ('results/%sForce_plot_Fref_%s_Kf_%s_Kz_%s_thr_%s.txt'%(Trajectory_n,Fref,Kf,Kz,thr))
#fl = open(file_name1, 'w')
#fl2 = open(file_name2, 'w')
file_current_pose = open('results/current_pose_XYZ.txt', 'w')
file_current_force = open('results/current_force_X_direction.txt', 'w')
file_name3 = ('trajectories/Position_for_video.txt')
f3 = open(file_name3, 'w')
figure_name = 'results/%s.figure_Fref_%s_Kf_%s_Kz_%s_thr_%s.png'%(Trajectory_n,Fref,Kf,Kz,thr)
figure_name_Y ='results/%s.Y_figure_Fref_%s_Kf_%s_Kz_%s_thr_%s.png'%(Trajectory_n,Fref,Kf,Kz,thr)
################################################################################

i = 0
j = 0
k = 0
h = 0
temp = []
position = []
full_position = []
corrected_position = []
##################____________Default_file_of positions________#######################

file = open('data/basic path.txt', 'w')
file2 = open('data/2th_coordinates.txt', 'w')
file3 = open('data/corrected_position.txt', 'w')


lines = [line.rstrip('\n') for line in open('data/position.txt')]

#Read positions from file
while i < len(lines) - 1:
    val = lines[i].split('(', 1)[1].split(')')[0]
    val2 = ([float(x) for x in val[0:-1].split(',')])  # list comprehension
    temp.append(val2)
    i += 1

# Filter coordinates and write into file
while k < len(temp) - 1:
    if math.fabs(temp[k][0]) - math.fabs(temp[k + 1][0]) >= 0.001 \
            or math.fabs(temp[k][1]) - math.fabs(temp[k + 1][1]) >= 0.001:
        full_position.append(temp[k])
        file.write(repr(temp[k]) + '\n')

    k += 1
file.close()

print(len(full_position))

# Write again coordinates in the other file without the last one
while j < len(full_position) - 1:
    position.append(full_position[j])
    file2.write(repr(full_position[j]) + '\n')

    j += 1
file2.close()

##########################################################
#a) Connect to robot

if config.Mode_Develop == False:
    robot = urx.Robot(config.IP_ADRESS, use_rt=True)
    s = socket.socket(
            socket.AF_INET, socket.SOCK_STREAM)

    s.connect((config.IP_ADRESS, 63351))
    print('connected')

stopper = None
############### connection establishement_thread_force reading ##############

#########################################################


# class for GUI
class Window(QtGui.QDialog, GUI.Ui_Dialog):

    def __init__(self):
        QtGui.QDialog.__init__(self)
        self.setupUi(self)
        self.center()
        ###########_____Butttons___________#############
        self.Button_freedrive.clicked.connect(self.robot_freedrive_able)
        self.Button_start.clicked.connect(self.button_start_record)
        self.Button_stop.clicked.connect(self.stop_record_button)
        self.toolButton.clicked.connect(self.FileBrowse)
        self.Button_move.clicked.connect(self.start_on_click)
        self.Button_stop_move.clicked.connect(self.stop_robot_move)
        self.Radio_button_plot.toggled.connect(self.Plot_abled)

        ###########__________Threads_________##############
        self.movie = QtGui.QMovie(self)
        self.Record_thread = Record_path()
        self.Force_Thread = Force_Thread()
        self.Position_set = Position_set()

        self.Move_Thread = Move_Thread(self.Position_set,self.Force_Thread)
        self.Plot_Thread = Plot_Thread(self.Position_set,self.Force_Thread,self.Move_Thread)
        self.Image_thread = Image_thread(self.Position_set,self.Force_Thread, self.Move_Thread)

        self.position_from_file = position #sample positions from file

# alignment of Window
    def center(self):
        frameGm = self.frameGeometry()
        screen = QtGui.QApplication.desktop().screenNumber(QtGui.QApplication.desktop().cursor().pos())
        centerPoint = QtGui.QApplication.desktop().screenGeometry(screen).center()
        frameGm.moveCenter(centerPoint)
        self.move(frameGm.topLeft())

    def robot_freedrive_able(self):
        robot.set_freedrive(True, timeout=60)

    def robot_freedrive_disable(self):
        robot.set_freedrive(False, timeout=60)

    # call of functions while clicking the main button "start"

    def move_robot(self):
        self.thread_move.stopper_r = False
        self.thread_move.start()

    def stop_robot_move(self):
        self.Move_Thread.stop_flag = True
        self.Move_Thread.stop_video = True
        #self.Image_thread.stop_button = True

    def button_start_record(self):
        self.Record_thread.start()
        #self.Record_thread.join()

    def stop_record_button(self):
        self.Record_thread.stopper = True

    def FileBrowse(self):
        filePath = QtGui.QFileDialog.getOpenFileName(self,
                                                     'Single File',
                                                     "~/Desktop/PyRevolution/PyQt4",
                                                     '*.txt')

        self.label_trajectory_name.setText(filePath[-19:])

        fileHandle = open(filePath, 'r')
        #lines = fileHandle.readlines()
        self.lines = [line.rstrip('\n') for line in fileHandle]
        print(self.lines)

    def Plot_abled(self):
        self.Plot_Thread.radio_button_enabled = True

    def Initialize_file(self):

        m = 0
        while m < len(self.lines) - 1:
            val = self.lines[m].split('[', 1)[1].split(']')[0]
            val2 = ([float(x) for x in val[0:-1].split(',')])  # list comprehension
            self.Position_set.position_from_file.append(val2)

            m += 1

    def start_on_click(self):

        utils.reset_FT300(robot)
        time.sleep(5)
        self.Image_thread.start()

        self.connect(self.Image_thread, QtCore.SIGNAL('PIX'), self.setImage)
        self.Force_Thread.start()
        self.Position_set.start()
        self.Initialize_file()
        self.Move_Thread.start()
        self.Plot_Thread.start()
        # self.Force_Thread.join()
        # self.Position_set.join()
        # self.Move_Thread.join()
        # self.Plot_Thread.join()

    def setImage(self, image):
        self.label_3.setPixmap(QtGui.QPixmap.fromImage(image))

# even when red cross is pressed - two windows are closing
    def closeEvent(self, QCloseEvent):
        QCloseEvent.accept()

#############_main loop_#############

# Classes for Threads, which obtain the value of force and position from
# another py file, which connects to robot

class Record_path(QtCore.QThread):
    def __init__(self):
        super(Record_path, self).__init__()
        self.stopper = False
        self.f_p = 0
        self.f_f = 0

        #self.robot = Robot_connect()

    def run(self):
        b = random.randint(1,10)
        self.f_p = open(('position_%s.txt')%b, 'w')
        #self.f_f = open('force.txt', 'w')

        j = 0
        k = 0
        temp = []
        position = []
        full_position = []

        while 1:
            if self.stopper == False:

                posa = robot.getl()
                array_p = np.around(posa, decimals=4)  # rounded position

                position = (
                    array_p[0], array_p[1], array_p[2], array_p[3], array_p[4], array_p[5])

                temp.append(posa)
                #self.f_p.write(repr(position) + '\n')

                time.sleep(.1)
            else:
                print(len(temp))
                # print(temp)
                # print(temp[1][0])

                while k < len(temp) - 1:
                    if math.fabs(math.fabs(temp[k][0]) - math.fabs(temp[k + 1][0])) >= 0.001 \
                            or math.fabs(math.fabs(temp[k][1]) - math.fabs(temp[k + 1][1])) >= 0.001:
                        full_position.append(temp[k])

                    k += 1

                print(len(full_position))
                print(full_position)
                while j < len(full_position) - 1:
                    #position.append(full_position[j])
                    self.f_p.write(repr(full_position[j]) + '\n')

                    j += 1

                self.f_p.close()
                #self.f_f.close()

                break

class Force_Thread(Thread):
    def __init__(self):
        Thread.__init__(self)
        self.Fz = 0
        self.Fdelta = 0
        self.Rx_next = 0
        self.Mx_raw = 0
        self.point_x_next = 0
        self.force_tread_stopper = False
        self.stop_movement = False
    def run(self):
        i = 0

        F = 0
        while self.force_tread_stopper == False:

            # if i == 150:
            #    Fref = -6
            #
            # if i == 300:
            #    Fref = -3

            # if i == 1000:
            #     self.stop_movement = True


            tstart = time.time()
            response = s.recv(4096)
            val = response.split('(', 1)[1].split(')')[0]
            array = [float(x) for x in val[1:-1].split(',')]

            #print ('Fx:', array[0], 'Fy:', array[1],'Fz:', array[2],'Mx:', array[3], 'My:', array[4], 'Mz:', array[5])
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
            self.Fdelta = Fref - math.fabs(self.Fz)

            if self.Fz > config.FORCE.Fmax:
                self.Fz_sat = config.FORCE.Fmax
            elif self.Fz < -config.FORCE.Fmax:
                self.Fz_sat = -config.FORCE.Fmax
            else:
                self.Fz_sat = self.Fz
            #print "Fz_sat", Fz_sat

            posa = robot.getl()
            point_x = posa[0]
            Rx_current = posa[3]
            #print "X", point_x
            ######## F - total force

            #F = ((self.Fx) ** 2 + (self.Fy) ** 2 + (self.Fz) ** 2) ** 1 / 2




            self.point_x_next = point_x + config.FORCE.K_delta * (math.fabs(self.Fz_sat) - Fref)

            ################ velocity control #################


            self.velocity_x = Kf*(-Fref - self.Fz_sat)
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
            if self.Fz > config.FORCE.Fcrit:
                robot.stop()
                print('Value is more than F critical')
            i = i + 1
            #print 'i:', i
class Position_set(Thread):

    def __init__(self,):
        Thread.__init__(self)
        self.position_from_file = []
        self.target = position[1]
        self.stop_flag_for_robot_move = False
        self.flag = False

    def run(self):
        threshold = None
        timeout = 5
        start_dist = self._get_lin_dist_c()
        #print(start_dist)
        if threshold is None:
            threshold = thr

            if threshold < 0.001:  # roboten precision is limited
                threshold = 0.001
        count = 0
        i = 2 # start with third point
        self.target = self.position_from_file[1]
        while self.stop_flag_for_robot_move == False:
            while i <len(self.position_from_file)-1:

                dist = self._get_lin_dist_c()
                #print(dist)
                if dist < threshold:
                    print('change coordinate here')

                    #print dist

                    self.target = self.position_from_file[i]
                    print(i,'point number:',self.target)
                    #pass_pose = Move_Thread(self.target)
                    i+=1
                if count > timeout * 10:
                    print('timeout')
                else:
                    count = 0

                count += 1
            if i == len(self.position_from_file)-1:

                self.stop_flag_for_robot_move = True
                print('job is done')
                break

    def _get_lin_dist_c(self):
        # FIXME: we have an issue here, it seems sometimes the axis angle received from robot
        pose = robot.getl()
        dist = 0

        for i in range(1,3):
            dist += (self.target[i] - pose[i]) ** 2

        # for i in range(3, 6):
        #     dist += ((target[i] - pose[i]) / 5) ** 2  # arbitraty length like
        return dist ** 0.5


class Move_Thread(Thread):
    def __init__(self, Pose_thread, Force_thread):
        Thread.__init__(self)
        self.target_pose = Pose_thread
        self.stop_flag = False
        self.force = Force_thread
        self.Zim = 0
        self.stop_video = False
        self.position = position
        self.start_plotting = False
        self.start_recording = False
        self.plotme = False
        self.Force_value_reached = False
        dispatcher.connect(self.dispatch_receive, signal="Z coordinate", sender=Image_thread)

    def run(self):
        self.move_first_point()
        vel_y = 0
        vel_z = 0
        Pose = robot.getl()
        start_movement_Y = Pose[1]
        while True:
            if self.stop_flag == False:

                # self.start_plotting = True
                # self.start_recording = True
                target_pose_from_list =self.target_pose.target
                pos_curr = robot.getl()
########################____X____#################
                vel_x = self.force.velocity_x

########################____Y____#################

                y_position_target = target_pose_from_list[1]
                pos_y_curr = pos_curr[1]

                if math.fabs(pos_y_curr - y_position_target) < 0.2:
                    vel_y = -Kz * (pos_y_curr - y_position_target)

                if self.Force_value_reached == True:
                    vel_y = -0.005
########################____Z____#################
                z_position_target = target_pose_from_list[2]
                pos_z_curr = pos_curr[2]
                K_zim = 0.3

                if self.Zim == 0:
                    if math.fabs(pos_z_curr - z_position_target) < 0.2:
                        vel_z = -Kz * (pos_z_curr - z_position_target)

                else:
                    if math.fabs(self.Zim) < 0.05 and math.fabs(self.Zim) > 0.003: ### 0.045 - sire of probe, 0.005 - robot precision
                        vel_z = K_zim * (self.Zim)
                    else:
                        vel_z = 0
########################____Rotations____#################
                #target_pose_from_list[4] = -1.20949955982  # RAD. 0 degree about Ry - roll axis
                # target_pose_from_list[5] = 1.20949955982 # correct Rz

                # def dispatch_receive(self,message):
                #     Zim = message

                #Zim = dispatcher.connect(dispatch_receive, signal = "Z coordinate", sender = Image_thread)


                # if math.fabs(Zim) < 0.2:
                #     vel_z = -Kz * (Zim)
                print ('From Move thread: ',self.Zim, vel_x, vel_y, vel_z)
                #vel_Rx = self.force.velocity_Rx
                #print(vel_z, vel_y)

########################____FEED to ROBOT____#################
                program = 'speedl([%s,%s,%s,0,0,0],0.05,0.07)' % (vel_x, vel_y, vel_z)
                #print program
                robot.send_program(program)

                # if self.force.stop_movement:
                #     self.plotme = True
                #     break

                if math.fabs(pos_y_curr - start_movement_Y)> 0.3:
                    self.stop_video = True
                    self.plotme = True
                    print('Robot finished 0.2 m')
                    # robot.close()
                    # s.close()
                    break

                if self.target_pose.stop_flag_for_robot_move == True:
                    self.plotme = True
                    print('Robot finished')
                    # robot.close()
                    # s.close()
                    break
            else:
                print('robot forced stop')
                break
                #robot.stopj()

    def dispatch_receive(self, message):
        self.Zim = message

    def move_first_point(self):
        first_pose = self.target_pose.position_from_file
        first_pose[0][0] += 0.002

        St = True
        #robot.movep(self.position[0], acc=a, vel=v, wait=True, relative=False, threshold=None)
        robot.movep(first_pose[0], acc=config.FORCE.a, vel=config.FORCE.v, wait=True, relative=False, threshold=None)
        print('robot in 0 position')
        while St == True:
            Fz_1 = self.force.Fz

            #print "Fz for first position is", Fz_1
            if math.fabs(Fz_1) < config.FORCE.Fref_first_move:
                # self.position[0][0] += -0.0002
                # robot.movep(self.position[0], acc=a, vel=v, wait=False, relative=False, threshold=None)
                #print "move to first position"
                first_pose[0][0] += -0.0002
                robot.movep(first_pose[0], acc=config.FORCE.a, vel=config.FORCE.v, wait=False, relative=False, threshold=None)

            else:
                print('Force value reached')
                self.start_plotting = True
                self.start_recording = True
                self.Force_value_reached = True
                St = False
                robot.stopj()
                break

class Plot_Thread(Thread):
    def __init__(self, Pose_thread, Force_thread, Move_thread):
        Thread.__init__(self)
        self.move_thread = Move_thread
        self.target_pose = Pose_thread
        self.stop_flag = True
        self.force = Force_thread
        self.position = position
        self.radio_button_enabled = False

    def run(self):
        while True:
            if self.radio_button_enabled == True:
                if self.move_thread.start_plotting == True:
                    start_time = time.time()
                    current_pose = []
                    current_force = []
                    Fy = []
                    Fz = []
                    Fx = []
                    My = []
                    Mz = []
                    X_pose = []
                    Y_pose = []
                    Z_pose = []
                    X_pose_raw = []
                    Y_pose_raw = []
                    Z_pose_raw = []
                    current_Mx = []

                    while True:
                        cp = robot.getl()
                        current_pose.append(cp)
                        Fx.append(self.force.Fx)
                        Fy.append(self.force.Fy)
                        Fz.append(self.force.Fz)
                        My.append(self.force.My)
                        Mz.append(self.force.Mz)
                        current_force.append(self.force.Fz)
                        current_Mx.append(self.force.Mx_raw)
                        #print self.force.Fz
                        #fl2.write(repr(self.force.Fz) + '\n')

                        #if self.target_pose.stop_flag_for_robot_move == True:
                        if self.move_thread.plotme == True:
                            end_time = time.time()
                            exec_time = end_time - start_time
                            n = exec_time / (len(current_pose))
                            t = np.arange(start_time, end_time, n)
                            np.set_printoptions(precision=4)

                            x1 = np.linspace(start_time, end_time, len(self.target_pose.position_from_file), endpoint=True)

                            j = 0
                            while j < len(current_pose):
                                file_current_pose.write(repr(current_pose[j]) + '\n')
                                j += 1

                            j = 0
                            while j < len(current_force):
                                file_current_force.write(repr(current_force[j]) + '\n')
                                j += 1


                           ################# X Y Z ####################
                            j = 0
                            while j < len(current_pose):
                                X_pose.append(current_pose[j][0])
                                #fl.write(repr(current_pose[j][0]) + '\n')
                                j += 1

                            j = 0
                            while j < len(current_pose):
                                Y_pose.append(current_pose[j][1])
                                # fl.write(repr(current_pose[j][0]) + '\n')
                                j += 1

                            j = 0
                            while j < len(current_pose):
                                Z_pose.append(current_pose[j][2])
                                # fl.write(repr(current_pose[j][0]) + '\n')
                                j += 1

                            ################## RAW ################
                            j = 0
                            while j < len(self.target_pose.position_from_file):
                                X_pose_raw.append(self.target_pose.position_from_file[j][0])
                                j += 1

                            j = 0
                            while j < len(self.target_pose.position_from_file):
                                Y_pose_raw.append(self.target_pose.position_from_file[j][1])
                                j += 1

                            j = 0
                            while j < len(self.target_pose.position_from_file):
                                Z_pose_raw.append(self.target_pose.position_from_file[j][2])
                                j += 1



                            # print len(X_pose_raw)
                            # print X_pose_raw
                            # print x1


                            ################## X - direction #################
                            plt.figure(figsize=(10,7))
                            plt.suptitle(figure_name, fontsize=10, fontweight='bold')
                            plt.subplot(221)
                            plt.title('feedback position X')
                            plt.plot(t, X_pose)
                            plt.xlabel('time (s)')
                            plt.ylabel('X, m')
                            plt.grid(True)

                            #plt.figure(figsize=(10, 7))
                            plt.subplot(222)
                            plt.plot(t,current_force)
                            plt.title('feedback force F')
                            plt.xlabel('time (s)')
                            plt.ylabel('Force, N')
                            plt.axhline(y=-Fref, color='r', linestyle='-')
                            plt.grid(True)


                            plt.subplot(223)
                            plt.title('initial recorded position X')
                            plt.plot(x1, X_pose_raw)
                            plt.ylabel('X, m')
                            plt.xlabel('time (s)')
                            plt.grid(True)

                            plt.subplot(224)
                            plt.title('Mx')
                            plt.plot(t, current_Mx)
                            plt.xlabel('time (s)')
                            plt.ylabel('Mx, N*m')
                            plt.grid(True)

                            plt.subplots_adjust(left=0.2, wspace=0.5, top=0.9, hspace=0.5, bottom=0.1)
                            #plt.savefig(figure_name)
                            plt.show()


                            ########## Y Z direction ###############
                            plt.figure(figsize=(10, 7))
                            plt.suptitle(figure_name_Y, fontsize=10, fontweight='bold')
                            plt.subplot(221)
                            plt.title('feedback position Y')
                            plt.plot(t, Y_pose)
                            plt.xlabel('time (s)')
                            plt.ylabel('Y, m')
                            plt.grid(True)

                            # # plt.figure(figsize=(10, 7))
                            # plt.subplot(222)
                            # plt.plot(t, current_force)
                            # plt.title('feedback force F')
                            # plt.xlabel('time (s)')
                            # plt.ylabel('Force, N')
                            # plt.axhline(y=-Fref, color='r', linestyle='-')
                            # plt.grid(True)

                            plt.subplot(223)
                            plt.title('initial recorded position Y')
                            plt.plot(x1, Y_pose_raw)
                            plt.ylabel('Y, m')
                            plt.xlabel('time (s)')
                            plt.grid(True)

                            plt.subplot(222)
                            plt.title('feedback position Z')
                            plt.plot(t, Z_pose)
                            plt.xlabel('time (s)')
                            plt.ylabel('Z, m')
                            plt.grid(True)

                            plt.subplot(224)
                            plt.title('initial recorded position Z')
                            plt.plot(x1, Z_pose_raw)
                            plt.ylabel('Z, m')
                            plt.xlabel('time (s)')
                            plt.grid(True)

                            plt.subplots_adjust(left=0.2, wspace=0.5, top=0.9, hspace=0.5, bottom=0.1)
                            # plt.savefig(figure_name_Y)
                            plt.show()


                            #finish threads here
                            break


class Image_thread(QtCore.QThread):
    changePixmap = pyqtSignal(QImage)
    def __init__(self, Pose_thread, Force_thread, Move_thread):
        QtCore.QThread.__init__(self)

        self.move_thread = Move_thread
        self.target_pose = Pose_thread

        self.force = Force_thread
        self.stop_button = False



    def run(self):
        i = 0
        fourcc = cv2.VideoWriter_fourcc(*'XVID')

        h_im = 320
        w_im = 320
        t = 95
        hnew = 2 * h_im
        wnew = 2 * w_im

        out = cv2.VideoWriter('results/output_original.avi', fourcc, 20.0, (640, 640))

############____OUT for a STack images___############
        out_stack = cv2.VideoWriter('results/output_stack.avi', fourcc, 20.0, (640, 320))
        img = ImageGrab.grab(bbox=(0, 0, fwidth, fheight))
        frame = np.array(img)
        h,w,_ = frame.shape

        while True:
            if self.move_thread.start_recording == True:

        # # ****************************************
        #         while (True):
                img = ImageGrab.grab(
                    bbox=(0, 0, fwidth, fheight))  # bbox specifies specific region (bbox= x,y,width,height)
#40,75
                frame = np.array(img)
                cv2.waitKey(100)



                #rgbImage = frame
                #rgbImage = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                #convertToQtFormat = QImage(rgbImage.data, rgbImage.shape[1], rgbImage.shape[0], QImage.Format_RGB888)
                #p = convertToQtFormat.scaled(401, 351)
                #p = convertToQtFormat.scaled(w, hg)
                #self.emit(QtCore.SIGNAL('PIX'), p)





                #####___h/2+t____middle of the screen with information
                ####____for US image depth 40 mm___####

                # h_im = 320
                # w_im = 445
                # t = 95
                # hnew = 2 * h_im
                # wnew = 2 * w_im

                ############_____for US Image depth 55 mm ########

                h_im = 320
                w_im = 320
                t = 95
                hnew = 2 * h_im
                wnew = 2 * w_im
                ###################################
                frame = frame[((h / 2)) - h_im:(((h / 2)) + h_im), (w / 2 + t - w_im):(w / 2 + t + w_im)]

                original_frame = frame
                #convertToQtFormat = QImage(new_frame.data, new_frame.shape[1], new_frame.shape[0], QImage.Format_RGB888)

                ############____VIDEO_____
                print("shape of frame:", frame.shape)
                #out.write(frame)  # write output file of changed frames
                centroid_X = wnew / 2
                X = []
                centroid_Y = hnew / 2
                Y = []
                #cv2.imshow('new size',frame)

                #
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                # ###########add contrast and brightness########
                contrast = 1.6
                brightness = 0
                frame = cv2.addWeighted(frame, contrast, frame, 0, brightness)

                f = ndimage.median_filter(frame, 10)
                # f = cv2.addWeighted(f, contrast, frame, 0, brightness)

                PS_f, orientation, _, T = phasesym(frame, nscale=2, norient=1, minWaveLength=25, mult=1.6,
                                                            sigmaOnf=0.25, k=1.5, polarity=1, noiseMethod=-1)
                # print(result.shape, type(result), result.dtype)

                PS = (PS_f * 255).astype(np.uint8)

                ret, th2 = cv2.threshold(PS, 30, 255, cv2.THRESH_BINARY)

                ##########   erosion   ###########
                kernel = np.ones((5, 5), np.uint8)
                erosion = cv2.erode(th2, kernel, iterations=2)
                th2 = erosion



                ##########__________small CONTOURS SELECTION____________##########
                cnts, hierarchy = cv2.findContours(th2, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
                contours = th2
                contours = cv2.cvtColor(contours, cv2.COLOR_GRAY2RGB)
                for c in cnts:
                    cv2.drawContours(contours, c, -1, (155, 0, 20), 2)

                mask = np.ones(th2.shape[:2], dtype="uint8") * 255
                minArea = 100

                filteredContours = []
                fitcontours = []
                for c in cnts:
                    area = cv2.contourArea(c)
                    if area < minArea:
                        filteredContours.append(c)
                        cv2.drawContours(mask, [c], -1, 0, -1)
                    else:
                        fitcontours.append(c)

                small_contours_deleted = cv2.bitwise_and(th2, th2, mask=mask)
                th2 = small_contours_deleted
                ############____Find min of biggest contour ##########
                cm = max(cnts, key=cv2.contourArea)
                bottommost = tuple(cm[cm[:, :, 1].argmax()][0])
                min_X, min_y = bottommost

                mask_with_small_contours_deleted = np.ones(small_contours_deleted.shape[:2], dtype="uint8") * 255
                elim_contours_above = []
                important_contours = []

                # kot = []
                ####____eliminate_the highest countours___#####
                cY_prev = 0

                for c in fitcontours:
                    cX, cY = utils.find_centroid(c)
                    if (cY < min_y and cY < hnew /6.5 or cY < hnew /6.5):
                        elim_contours_above.append(c)
                        cv2.drawContours(mask_with_small_contours_deleted, [c], -1, 0, -1)


                filtered_contours = cv2.bitwise_and(small_contours_deleted, small_contours_deleted,
                                                    mask=mask_with_small_contours_deleted)

                ################################
                dilation = cv2.dilate(filtered_contours, kernel, iterations=10)
                th2 = dilation

                ################_________Contours_______################
                cnts, hierarchy = cv2.findContours(th2, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

                # c = max(cnts, key=cv2.contourArea)
                # cv2.drawContours(th2, c, -1, (155, 255, 255), 5)
                #

                ##########________Extreme_points_______#############
                tops = []
                X_list = []
                Y_list = []
                top_positions = []
                for c in cnts:
                    cX, cY = utils.find_centroid(c)
                    if (cY < 7 * hnew / 10):
                        # extBot = tuple(c[c[:, :, 1].argmax()][0])
                        # extLeft = tuple(c[c[:, :, 0].argmin()][0])
                        # extRight = tuple(c[c[:, :, 0].argmax()][0])
                        extTop = tuple(c[c[:, :, 1].argmin()][0])
                        max_X, max_y = extTop
                        tops.append(extTop)
                        X_list.append(max_X)

                        Y_list.append(max_y)
                        pos = max_X, max_y
                        top_positions.append(pos)
                top_positions = np.array(top_positions, dtype=np.int32)

                if len(Y_list) > 0:
                    maxY = min(Y_list)
                else:
                    maxY = 0

                if len(X_list) != 0:
                    meanX = sum(X_list) / len(X_list)
                else:
                    meanX = wnew / 2

                # maxTop = meanX, maxY

                if len(tops) > 0:
                    maxTop_one = min(tops, key=lambda item: item[1])
                else:
                    maxTop_one = [hnew/2,0]


                result_img = cv2.cvtColor(th2, cv2.COLOR_GRAY2RGB)
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
                original_frame = frame
                ###############____Centroid_approximation____#########
                alpha = 0.6
                if i == 0:
                    centroid_X, centroid_Y = meanX, maxY

                if i > 0:
                    #centroid_X = int(round((1 - alpha) * centroid_X + alpha * meanX))
                    #centroid_X = int(round((1 - alpha) * centroid_X + alpha * maxTop_one[0]))
                    centroid_X = int(round((1 - config.alpha) * centroid_X + config.alpha * meanX))
                    # centroid_Y = int(round((1 - alpha) * centroid_Y + alpha * maxY))
                # centroid_X, centroid_Y = meanX, maxY
                centroid_Y = maxY
                # centroid_Y = 200

                #################_________Relative_Coordinate__X image - Z robot______############
                lineThickness = 2;
                # coordinate of image center
                x_image_center = wnew / 2;
                y_image_center = hnew
                position_relative = centroid_X - x_image_center  # minus for left point, plus for right point

                # proportion between FOV of probe and Pixels in image: FOV/wnew
                K_image_t_dist = config.FOV / wnew
                Z_robot_cord_f_image = K_image_t_dist * position_relative




                self.Z_robot_image_cent = Z_robot_cord_f_image
                print('From Image thread: ',self.Z_robot_image_cent)
                dispatcher.send(message = Z_robot_cord_f_image,signal = "Z coordinate", sender = Image_thread)




                for points in tops:
                    cv2.circle(result_img, points, 15, (255, 0, 0), -1)

                cv2.circle(result_img, (centroid_X, centroid_Y), 15, (0, 255, 0), -1)
                #cv2.circle(frame, (centroid_X, centroid_Y), 15, (0, 255, 0), -1)

                # cv2.imshow('original', frame)
                # cv2.imshow('result', result_img)

                frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
                #new_frame = cv2.resize(new_frame, (0, 0), fx=0.5, fy=0.5)
                result_img = cv2.resize(result_img, (0, 0), fx=0.5, fy=0.5)
                #cv2.imshow('res',result_img)
                cv2.imshow("Result", np.hstack([frame, result_img]))
                #result = np.hstack([frame, result_img])
                #print(result.shape)

 ##################_________RECORD_VIDEO and positions_____###########

                position_video = robot.getl()
                Force = self.force.F_changed_order
                f3.write(repr(position_video + Force) + '\n')

                out_stack.write(np.hstack([frame, result_img]))

                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                out.write(original_frame)

            # if self.stop_button == True:
            #     print('video stopped because "stop button"')
            #     f3.close()
            #     out.release()
            #     out_stack.release()
            #     break

            if self.move_thread.stop_video == True:
                print('video stopped because robot finished or "stop button"')
                f3.close()
                out.release()
                out_stack.release()
                break

            if self.target_pose.stop_flag_for_robot_move == True:
                print('video stopped because all positions finished')
                f3.close()
                out.release()
                out_stack.release()
                break
            i += 1

        cv2.destroyAllWindows()
        sys.exit()


def main():
    try:
        app = QtGui.QApplication(sys.argv)
        form = Window()
        form.show()
        app.exec_()
    except KeyboardInterrupt:
        f3.close()
        out.release()
        out_stack.release()


# launch GUI function
if __name__ == '__main__':
    main()

