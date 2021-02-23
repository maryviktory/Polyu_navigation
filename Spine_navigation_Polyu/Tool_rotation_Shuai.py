import urx
import time
from threading import Thread
import math3d as m3d
import pandas as pd
import os


IP_ADRESS = "158.132.172.194"
robot = urx.Robot(IP_ADRESS, use_rt=True)
record_pose = True

class Record(Thread):
    def __init__(self):
        Thread.__init__(self)

        self.record_pose = True

    def run(self):
        pd_frame = pd.DataFrame(columns=["time","X", "Y", "Z", "Rx", "Ry", "Rz"])
        start_time = time.time()
        while self.record_pose == True:
            pose = robot.getl()

            time_csv = time.time()-start_time
            pd_frame = pd_frame.append({'time':time_csv,'X':pose[0],'Y':pose[1],'Z':pose[2],'Rx':pose[3],'Ry':pose[4],'Rz':pose[5]},
                                       ignore_index=True)

        else:
            total_time = time.time() - start_time
            print("total time", total_time)
            pd_frame.to_csv( "robot_frame.csv")

            os._exit(0)

if __name__ == '__main__':

    robot.set_tcp((0, 0, 0.27, 0, 0, 0))  # 0.1 - 10 cm
    robot.set_payload(1.5)  # KG
    time.sleep(0.2)
    pose_thread = Record()
    pose_thread.start()
    start_time = time.time()
    print("start moving")
    robot.movej((0, 0, 0, 0, 0, -3.14), acc=0.1, vel=0.05, wait=True, relative=True, threshold=None)
    # robot.rx -= 0.1  # rotate tool around X axis
    pose_thread.record_pose = False
    print("finished moving {}".format(time.time() - start_time))

