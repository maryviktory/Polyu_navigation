import socket
import time
s = socket.socket(
            socket.AF_INET, socket.SOCK_STREAM)

s.connect(('158.132.172.194', 63351))
print('connected')

while True:
    # if i == 150:
    #    Fref = -6
    #
    # if i == 300:
    #    Fref = -3

    # if i == 1000:
    #     self.stop_movement = True


    response = s.recv(4096)
    val = response.split('(', 1)[1].split(')')[0]
    array = [float(x) for x in val[1:-1].split(',')]
    time.sleep(2)
    print ('Fx:', array[0], 'Fy:', array[1], 'Fz:', array[2], 'Mx:', array[3], 'My:', array[4], 'Mz:', array[5])
    # print ('      ')