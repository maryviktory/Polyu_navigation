import math
import GUI.Scolioscan_robotics_big_GUI_Qt5 as GUI
import sys
import time
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
from utils.config import config
import utils as utils


# import mss
# import mss.tools

#################################_____PARAMETERS to adjust_____##########################

fheight =1080
fwidth = 1920

print(fwidth, fheight)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, 20.0, (fwidth, fheight))

w = 400; hg = 360

'''transformation matrix to easily swap the coordinate system. 
Originally the axis in the code are the following: y - down, x-opposite to the skin surface (force control),
z - to the right hand side (image control). To swap it to the other direction, we need to multiply it with transformation matrix
 [V_robx, V_roby, V_robz]= [-1 0 0; 0 -1 0; 0 0 1] * [v_force, v_up, v_im] The transformation matrix should be
 picked to match the axis of the robot base coordinate system
 V_robot = mp.matmul(T_marix, V_control)
 '''

# T_force = -1
T_up = -1
T_im = 1

T_matrix = [[-1,0,0],[0,-1,0],[0,0,1]]
control_force = [1,0,0]
force_s = np.matmul(T_matrix,control_force)
force_slot = int(np.argwhere(np.matmul(T_matrix,control_force)!=0))
T_force = int(force_s[force_slot])
# print(force_s,force_slot, T_force)

V_control = np.zeros(3)
V_robot = np.zeros(3)




Fref = config.FORCE.Fref
Kf = config.FORCE.Kf
Kz = config.FORCE.Kz
thr = config.FORCE.thr
Trajectory_n = config.Trajectory_n
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
while i < len(lines) - 1:
    val = lines[i].split('(', 1)[1].split(')')[0]
    val2 = ([float(x) for x in val[0:-1].split(',')])  # list comprehension
    temp.append(val2)
    i += 1

while k < len(temp) - 1:
    if math.fabs(temp[k][0]) - math.fabs(temp[k + 1][0]) >= 0.001 \
            or math.fabs(temp[k][1]) - math.fabs(temp[k + 1][1]) >= 0.001:
        full_position.append(temp[k])
        file.write(repr(temp[k]) + '\n')

    k += 1
file.close()

print(len(full_position))

while j < len(full_position) - 1:
    position.append(full_position[j])
    file2.write(repr(full_position[j]) + '\n')

    j += 1
file2.close()
##########################################################
#a) Connect to robot
if config.Mode_Develop == False:
    robot = urx.Robot("158.132.172.194", use_rt=True)
    s = socket.socket(
            socket.AF_INET, socket.SOCK_STREAM)

    s.connect(('158.132.172.194', 63351))
    print('connected')

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

        self.position_from_file = position

# alignment of Window
    def center(self):
        frameGm = self.frameGeometry()
        screen = QtWidgets.QApplication.desktop().screenNumber(QtWidgets.QApplication.desktop().cursor().pos())
        centerPoint = QtWidgets.QApplication.desktop().screenGeometry(screen).center()
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

    def button_start_record(self):
        self.Record_thread.start()
        #self.Record_thread.join()

    def stop_record_button(self):
        self.Record_thread.stopper = True

    def FileBrowse(self):
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
        while m < len(self.lines) - 1:
            val = self.lines[m].split('[', 1)[1].split(']')[0]
            val2 = ([float(x) for x in val[0:-1].split(',')])  # list comprehension
            self.Position_set.position_from_file.append(val2)

            m += 1

    def start_on_click(self):

        utils.reset_FT300(robot)
        # time.sleep(5)
        self.Image_thread.start()

        self.Image_thread.PIX.connect(self.setImage)
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
        self.f_p = open(('trajectories/position_%s.txt')%b, 'w')
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
    '''Force sensor has different coordinate system, it reads measurements in tool coordintate system,
        so that when it touches the surface perpendicular to the tool, it reads the Fz

        To get exact forces in the robot coordinate system, it is better to multiply Forces from the Force sensor with
        the transformation matrix from the robot, Universal robot can provide it at each moment of time.

        To symplify the task just the proportion between the Fz and Fset is used to generate the velocity
        for the robot control. ?Velocity should be set also in the tool frame? Otherwise the axis of the robot
        and the force sensor should be aligned during robot manipulation

        Now is:
        x robot - z force sensor
        y robot - x force sensor
        z robot - y force sensor

        #TODO: To change it the transformation matrix should be introduced.
        '''
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
            self.Fdelta = Fref - math.fabs(self.Fz)

            if self.Fz > config.FORCE.Fmax:
                self.Fz_sat = config.FORCE.Fmax
            elif self.Fz < -config.FORCE.Fmax:
                self.Fz_sat = -config.FORCE.Fmax
            else:
                self.Fz_sat = self.Fz
            print("Fz_sat", self.Fz_sat)

            posa = robot.getl()
            point_x = posa[0]
            Rx_current = posa[3]
            #print "X", point_x
            ######## F - total force

            #F = ((self.Fx) ** 2 + (self.Fy) ** 2 + (self.Fz) ** 2) ** 1 / 2




            self.point_x_next = point_x + config.FORCE.K_delta  * (math.fabs(self.Fz_sat) - Fref)

            ################ velocity control #################


            self.velocity_x = Kf*(-Fref - self.Fz_sat)
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
        self.position = position
        self.start_plotting = False
        self.start_recording = False
        self.plotme = False

    def run(self):
        self.move_first_point()
        print("go to first point")
        vel_y = 0
        vel_z = 0
        while True:
            if self.stop_flag == False:
                # self.start_plotting = True
                # self.start_recording = True
                target_pose_from_list =self.target_pose.target
                z_position_target = target_pose_from_list[2]
                y_position_target = target_pose_from_list[1]

                pos_curr = robot.getl()
                pos_z_curr = pos_curr[2]
                pos_y_curr = pos_curr[1]
                #print pos_curr

                #target_pose_from_list[4] = -1.20949955982  # RAD. 0 degree about Ry - roll axis
                # target_pose_from_list[5] = 1.20949955982 # correct Rz

                vel_force = self.force.velocity_x
                V_control[0] = vel_force
                if math.fabs(pos_y_curr - y_position_target) < 0.2:
                    vel_up = -Kz * (pos_y_curr - y_position_target)
                    V_control[1] = vel_up
                if math.fabs(pos_z_curr - z_position_target) < 0.2:
                    vel_im = -Kz * (pos_z_curr - z_position_target)
                    V_control[2] = vel_im
                #vel_Rx = self.force.velocity_Rx
                #print(vel_z, vel_y)


                V_robot = np.matmul(T_matrix,V_control)



                # vel_x = vel_x*T_force
                # vel_y = vel_y*T_up
                # vel_z = vel_z*T_im

                program = 'speedl([%s,%s,%s,0,0,0],0.05,0.5)' % (V_robot[0], V_robot[1], V_robot[2])
                #print program
                robot.send_program(program)

                # if self.force.stop_movement:
                #     self.plotme = True
                #     break
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

                first_pose[0][force_slot] += -0.0002*T_force
                robot.movep(first_pose[0], acc=config.FORCE.a, vel=config.FORCE.v, wait=False, relative=False, threshold=None)

            else:
                print('Force value reached')
                self.start_plotting = True
                self.start_recording = True
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
    PIX = QtCore.pyqtSignal(QtGui.QImage)

    def __init__(self, Pose_thread, Force_thread, Move_thread):
        QtCore.QThread.__init__(self)

        self.move_thread = Move_thread
        self.target_pose = Pose_thread

        self.force = Force_thread


    def run(self):
        i = 0
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter('output.avi', fourcc, 20.0, (fwidth, fheight))
        while True:
            if self.move_thread.start_recording == True:

        # # ****************************************
        #         while (True):
                img = ImageGrab.grab(
                    bbox=(0, 0, fwidth, fheight))  # bbox specifies specific region (bbox= x,y,width,height)
#40,75
                frame = np.array(img)
                # frame = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)
                position_video = robot.getl()
                Force = self.force.F_changed_order
                f3.write(repr(position_video + Force) + '\n')


                cv2.waitKey(100)
                rgbImage = frame
                #rgbImage = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                convertToQtFormat = QImage(rgbImage.data, rgbImage.shape[1], rgbImage.shape[0], QImage.Format_RGB888)
                #p = convertToQtFormat.scaled(401, 351)
                p = convertToQtFormat.scaled(w, hg)

                self.PIX.emit(p)


                #cv2.imshow("test", rgbImage)
                out.write(rgbImage)  # write output file of changed frames

                # make conditions when the program is switched off
                # if cv2.waitKey(1) & 0xFF == ord('q'):
                #     break
                #key = cv2.waitKey(1)

            if self.target_pose.stop_flag_for_robot_move == True:
                 print('video stopped')
                 f3.close()
                 out.release()
                 break
            i += 1
            #         if key == 27:
        # break

        # Release everything if job is finished
        #
        cv2.destroyAllWindows()
        sys.exit()


def main():
    try:
        app = QtWidgets.QApplication(sys.argv)
        form = Window()
        form.show()
        app.exec_()
    except KeyboardInterrupt:
        f3.close()
        out.release()


# launch GUI function
if __name__ == '__main__':
    main()

