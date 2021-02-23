def seam_tracking(self):
    global logi_pre, logi_post, rs_pre, rob, new_seal_start_pose, new_vel_vec, new_seal_end_pose

    rob = urx.Robot("192.168.0.2")
    tcp_torch = [-0.0002, -0.09216, 0.32202, 0, 0, 0]
    rob.set_tcp(tcp_torch)
    time.sleep(0.1)  # pause is essentail for tcp to take effect, min time is 0.1s

    rob.movel(new_seal_start_pose, acc=0.1, vel=0.1, wait=True)

    self.start = Point()# from ROS messaging
    self.start.x = 1
    self.start.y = 0
    self.start.z = 0
    self.pub_start_process.publish(self.start)

    vel_scale = 0.01

    translation_speed = vel_scale * 0.5

    pose_diff = np.linalg.norm(np.array(new_seal_end_pose[:3]) - np.array(new_seal_start_pose[:3]))
    total_time = pose_diff / translation_speed

    print "total time is (s)"
    print total_time

    vx = 0
    vy = translation_speed
    vz = 0

    vrx = 0
    vry = 0
    vrz = 0

    dx = 0
    dy = 0
    dz = 0

    drx = 0
    dry = 0
    drz = 0

    a = vel_scale * 5
    t = 0.1

    # usr_input = raw_input("please enter to continue(q to quit): ")
    print("start feedback control")
    usr_input = ''

    if not usr_input == 'q':

        # rob.set_digital_out(0,True)
        rob.set_digital_out(0, False)
        time.sleep(1)

        start = time.time()

        while True:

            if logi_pre:

                print "working"

                logi_y_diff = logi_pre[-1]
                angle_diff = logi_pre[0]
                y_scale = vel_scale * 0.5
                a_scale = vel_scale * 2

                if abs(logi_y_diff) > 2:
                    # print rs_y_diff
                    dx = (logi_y_diff) * y_scale * 0.8
                    # if logi_y_diff > 0:
                    #     dy = -(logi_y_diff-0.5)*y_scale
                    # else:
                    #     dy = -(logi_y_diff+0.5)*y_scale
                elif abs(logi_y_diff) > 1:
                    dx = (logi_y_diff) * y_scale * 0.3
                elif abs(logi_y_diff) > 0.5:
                    dx = (logi_y_diff) * y_scale * 0.1
                elif abs(logi_y_diff) > 0.2:
                    dx = (logi_y_diff) * y_scale * 0.05
                else:
                    dx = 0
                    print "all well"

                if abs(dx) > vel_scale * 0.5:
                    if dx > 0:
                        dx = vel_scale * 0.5
                    else:
                        dx = -vel_scale * 0.5

                if abs(angle_diff) > 5:
                    drz = (angle_diff) * a_scale
                elif abs(angle_diff) > 2:
                    drz = (angle_diff) * a_scale * 0.4
                elif abs(angle_diff) > 0.5:
                    drz = (angle_diff) * a_scale * 0.1

                if abs(drz) > a_scale * 4:
                    if drz > 0:
                        drz = a_scale * 4
                    else:
                        drz = -a_scale * 4

                print "moving at z = " + str(drz)

            vel_speed = [vx + dx, vy + dy, vz + dz, vrx + drx, vry + dry, vrz + drz]
            rob.speedl_tool(vel_speed, a, t)

            end = time.time()
            print int(end - start)

            residual = total_time * 0.9 - (end - start)

            print residual

            if residual <= 0:
                print("break")
                break
            elif 0 < residual <= 2:
                self.start = Point()
                self.start.x = -1
                self.start.y = 0
                self.start.z = 0
                self.pub_start_process.publish(self.start)
                print("publish end")
            elif 2 < residual <= 3.5:
                self.start = Point()
                self.start.x = 0
                self.start.y = 1
                self.start.z = 0
                self.pub_start_process.publish(self.start)

        self.start = Point()
        self.start.x = -1
        self.start.y = 0
        self.start.z = 0
        self.pub_start_process.publish(self.start)
        rob.stop()
        rob.set_digital_out(0, False)
        # raw_input("Press any to continue")
        rob.translate_tool((0, 0, -0.08), vel=0.1, acc=0.1, wait=True)
        rob.stop()

    if usr_input == 'q':
        rob.translate_tool((0, 0, -0.08), vel=0.1, acc=0.1, wait=True)