# Make sure to have the add-on "ZMQ remote API" running in
# CoppeliaSim and have following scene loaded:
#
# scenes/messaging/synchronousImageTransmissionViaRemoteApi.ttt
#
# Do not launch simulation, but run this script
#
# All CoppeliaSim commands will run in blocking mode (block
# until a reply from CoppeliaSim is received). For a non-
# blocking example, see simpleTest-nonBlocking.py

import time

import math
import numpy as np
#import cv2
import matplotlib.pyplot as plt
from numpy import savetxt
#from matplotlib.animation import FuncAnimation

from coppeliasim_zmqremoteapi_client import RemoteAPIClient


print('Program started')



client = RemoteAPIClient('localhost',23004)
sim = client.getObject('sim')

#ctrlPts = [x y z qx qy qz qw]

#pathHandle = sim.createPath(ctrlPts, options = 0, subdiv = 100, smoothness = 1.0, orientationMode = 0, upVector = [0, 0, 1])



visionSensorHandle = sim.getObject('/Vision_sensor')
fl_w = sim.getObject('/flw')
fr_w = sim.getObject('/frw')
rr_w = sim.getObject('/rrw')
rl_w = sim.getObject('/rlw')
IMU = sim.getObject('/Accelerometer_forceSensor')
#COM = sim.getObject('/Husky/ReferenceFrame')
COM = sim.getObject('/Husky/Accelerometer/Accelerometer_mass')
Husky_ref = sim.getObject('/Husky')
#InertialFrame = sim.getObject('/InertialFrame')

#Gyro = sim.getObject('/GyroSensor_reference')
#gyroCommunicationTube=sim.tubeOpen(0,'gyroData'..sim.getNameSuffix(nil),1)


x_acc = []
y_acc = []
z_acc = []
x_ang = []
y_ang = []
z_ang = []
x_pos = []
y_pos = []
lin_vel = []
ang_vel = []
lin_vel2 = []
lin_vel3 = []
ang_vel2 = []
sim_time = []
cmd_wheel_l = []
cmd_wheel_r = []
wheel_l = []
wheel_r = []
rlz_wheel_l = []
rlz_wheel_r = []
counter = []


# When simulation is not running, ZMQ message handling could be a bit
# slow, since the idle loop runs at 8 Hz by default. So let's make
# sure that the idle loop runs at full speed for this program:
defaultIdleFps = sim.getInt32Param(sim.intparam_idle_fps)
sim.setInt32Param(sim.intparam_idle_fps, 0)

#sim.setEngineFloatParam(int paramId, int objectHandle, float floatParam)

# Run a simulation in stepping mode:
client.setStepping(True)
sim.startSimulation()





while (t:= sim.getSimulationTime()) < 60:
    #print(t)
    
    # IMAGE PROCESSING CODE
    
    #img, resX, resY = sim.getVisionSensorCharImage(visionSensorHandle)
    #img = np.frombuffer(img, dtype=np.uint8).reshape(resY, resX, 3)

    # In CoppeliaSim images are left to right (x-axis), and bottom to top (y-axis)
    # (consistent with the axes of vision sensors, pointing Z outwards, Y up)
    # and color format is RGB triplets, whereas OpenCV uses BGR:
    
    #img = cv2.flip(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), 0)
    #cv2.imshow('', img)
    #cv2.waitKey(1)

    # INPUT
    # WHEEL CONTROL CODE 3.391 6.056
    Left_ = 4
    Right_ = 12
    cmd_wheel_l.append(Left_)
    cmd_wheel_r.append(Right_)



    '''
    # Convert to V and Omega
    A_mat = np.array([[0.0825,0.0825],[-0.2775,0.27755]])
    wheel_vels = np.array([[Left_],[Right_]])
    #print(wheel_vels)
    #velocity = np.matmul(A_mat,wheel_vels)
    velocity = np.array([[3 + 0.5*np.sin(0.3*t + 0)],[0.5*np.sin(0.3*t + 0)]])
    #print(velocity)
    A_inv = np.array([[6.1358,-2.9177],[6.1358,2.9177]])
    phi_dots = np.matmul(A_inv,velocity)
    #print(phi_dots)
    phi_dots = phi_dots.astype(float)
    Left = phi_dots[0].item()
    Right = phi_dots[1].item()
    '''


    sim.setJointTargetVelocity(fl_w, Left_)
    sim.setJointTargetVelocity(fr_w, Right_)
    sim.setJointTargetVelocity(rl_w, Left_)
    sim.setJointTargetVelocity(rr_w, Right_)
    wheel_l.append(Left_)
    wheel_r.append(Right_)
    
    wheel_calc = (0.165/2)*(Left_+Right_)


    #OUTPUT

    # Position for validation
    position = sim.getObjectPosition(COM,sim.handle_world)
    if position[0] == None:
        pass
    else:
        x_pos.append(position[0])
        y_pos.append(position[1])
        #print(position[0],position[1])

    # Linear Velocity for validation
    #linear_vel, angular_vel = sim.getObjectVelocity(COM | sim.handleflag_axis)
    linear_vel, angular_vel = sim.getVelocity(Husky_ref)
    
    
    #print(linear_vel)
    #print("global angular Vel:")
    #print(angular_vel)
    #linear_vel2, angular_vel2 = sim.getObjectVelocity(Husky_ref | sim.handleflag_axis)
    #linear_vel, angular_vel = sim.getVelocity(Husky_ref)
    #print(linear_vel)
    #print(angular_vel)

    # But these velocities are in spatial frame: need to convert them to body frame with adjoint transformation
    # https://dellaert.github.io/20S-8803MM/Readings/3D-Adjoints-note.pdf

    sRb = sim.getObjectMatrix(COM,sim.handle_world)
    #sRb = [ round(elem, 2) for elem in sRb ]

    Rot = np.array([[sRb[0],sRb[1],sRb[2]],[sRb[4],sRb[5],sRb[6]],[sRb[8],sRb[9],sRb[10]]])
    vel_body = np.matmul(np.transpose(Rot),np.array([[linear_vel[0]],[linear_vel[1]],[linear_vel[2]]]))
    lin_vel.append(vel_body[2].item())

    '''

    print("List of transform matrix")
    print(sRb)
    sRb2 = sim.getMatrixInverse(sRb)

    #sRb_i = sim.getObjectMatrix(Husky_ref,sim.handle_inverse)
    
    
    #print(sRb)
    Rot = np.array([[sRb[0],sRb[1],sRb[2]],[sRb[4],sRb[5],sRb[6]],[sRb[8],sRb[9],sRb[10]]])
    trans = np.array([[position[0]],[position[1]],[position[2]]])
    trans_ss = np.array([[0,-position[2],position[1]],[position[2],0,-position[0]],[-position[1],position[0],0]])
    space_rot_vel = np.array([[0],[0],[angular_vel[2]]])
    space_lin_vel = np.array([[linear_vel[0]],[linear_vel[1]],[linear_vel[2]]])

    temp_a = np.matmul(trans_ss, Rot)
    temp_b = np.matmul(temp_a,space_rot_vel)
    temp_c = np.matmul(Rot,space_lin_vel)

    vel_body = np.array([[temp_b[0]+temp_c[0]],[temp_b[1]+temp_c[1]],[temp_b[2]+temp_c[2]]])
    

    # Lynch and Park Notations:
    T = np.array([[sRb[0],sRb[1],sRb[2],sRb[3]],[sRb[4],sRb[5],sRb[6],sRb[7]],[sRb[8],sRb[9],sRb[10],sRb[11]],[0,0,0,1]])
    #T2d = np.array([[sRb[0],sRb[1],sRb[2],sRb[3]],[sRb[4],sRb[5],sRb[6],sRb[7]],[sRb[8],sRb[9],sRb[10],sRb[11]],[0,0,0,1]])
    T2 = np.array([[sRb2[0],sRb2[1],sRb2[2],sRb2[3]],[sRb2[4],sRb2[5],sRb2[6],sRb2[7]],[sRb2[8],sRb2[9],sRb2[10],sRb2[11]],[0,0,0,1]])
    #T_in = np.array([[sRb_i[0],sRb_i[1],sRb_i[2],sRb_i[3]],[sRb_i[4],sRb_i[5],sRb_i[6],sRb_i[7]],[sRb_i[8],sRb_i[9],sRb_i[10],sRb_i[11]],[0,0,0,1]])
    print("Transformantion Matrix")
    print(T)
    #V = np.array([[0],[0],[angular_vel[2]],[linear_vel[0]],[linear_vel[1]],[linear_vel[2]]])

    V = np.array([[0,-angular_vel[2],angular_vel[1],linear_vel[0]],[angular_vel[2],0,-angular_vel[0],linear_vel[1]],[-angular_vel[1],angular_vel[0],0,linear_vel[2]],[0,0,0,0]])
    V2 = np.array([[0,-angular_vel2[2],angular_vel2[1],linear_vel2[0]],[angular_vel2[2],0,-angular_vel2[0],linear_vel2[1]],[-angular_vel2[1],angular_vel2[0],0,linear_vel2[2]],[0,0,0,0]])
    #print(V)

    from numpy.linalg import inv

    #print(inv(T))
    #temp_a = np.cross(inv(T),V)

    
    sRb = self.sim.getObjectMatrix(self.COM,self.sim.handle_world)
    


    #V_ = T*V
    vel_body0 = T @ V @ inv(T)
    #vel_body = vel_body0
    vel_body = sRb2[0]*vel_body0[0,3] + sRb2[4]*vel_body0[1,3]
    #vel_body2 = T @ vel_body @ inv(T)

    print("vel_body")
    print(vel_body)
    #print(vel_body[0,3].item())



    #print(np.shape(trans))
    #vel_body = 

    #matrix: array(list object python) of 12 values [Vx0 Vy0 Vz0 P0 Vx1 Vy1 Vz1 P1 Vx2 Vy2 Vz2 P2]





    

    #lin_vel.append(vel_body0[0,3].item())
    #lin_vel.append(vel_body)
    lin_vel.append(vel_body[2].item())
    #ang_vel2.append(angular_vel[2])


    #lin_vel2.append(wheel_calc)
    #lin_vel3.append(linear_vel[0])

    #ang_vel2.append(vel_body[1,0].item())



    '''
    # IMU and Gyro readings
    IMU_X = sim.getFloatSignal("myIMUData_X")
    if IMU_X:
        x_acc.append(IMU_X)
        #print(IMU_X)
    IMU_Y = sim.getFloatSignal("myIMUData_Y")
    if IMU_Y:
        y_acc.append(IMU_Y)
        #print(IMU_Y)
    IMU_Z = sim.getFloatSignal("myIMUData_Z")
    if IMU_Z:
        z_acc.append(IMU_Z)
        #print(IMU_Z)

    Gyro_X = sim.getFloatSignal("myGyroData_angX")
    if IMU_X:
        x_ang.append(Gyro_X)
        #print(Gyro_X)
    Gyro_Y = sim.getFloatSignal("myGyroData_angY")
    if Gyro_Y:
        y_ang.append(Gyro_Y)
        #print(Gyro_Y)
    Gyro_Z = sim.getFloatSignal("myGyroData_angZ")
    if Gyro_Z:
        z_ang.append(Gyro_Z)
        print("Angular Vel")
        print(Gyro_Z)


    # Realized joint velocity
    rlz_l = sim.getJointVelocity(fl_w)
    rlz_r = sim.getJointVelocity(fr_w)
    rlz_wheel_l.append(rlz_l)
    rlz_wheel_r.append(rlz_r)


    sim_time.append(t)
    print("Time")
    #print(t)
    print("=================================================================================")

    client.step()  # triggers next simulation step



sim.stopSimulation()

# Restore the original idle loop frequency:
sim.setInt32Param(sim.intparam_idle_fps, defaultIdleFps)

#cv2.destroyAllWindows()

print('Program ended')

#savetxt('x_acc_cc.csv', x_acc, delimiter=',')
#savetxt('y_acc_cc.csv', y_acc, delimiter=',')
#savetxt('z_acc_cc.csv', z_acc, delimiter=',')
#savetxt('x_ang_cc.csv', x_ang, delimiter=',')
#savetxt('y_ang_cc.csv', y_ang, delimiter=',')
#savetxt('z_ang_06_05.csv', z_ang, delimiter=',')

#def every_tenth(data,no):
    #data2 = data[no - 1::no]
    #return [data[0],data2]

#print(len(lin_vel))
#lin_vel = every_tenth(lin_vel,10)
#print(len(lin_vel))



#savetxt('x_pos_0.csv', x_pos, delimiter=',')
#savetxt('y_pos_0.csv', y_pos, delimiter=',')
#savetxt('x_vel_0.csv', lin_vel, delimiter=',')
#savetxt('x_vel2.csv', lin_vel2, delimiter=',')
#savetxt('x_vel3.csv', lin_vel3, delimiter=',')
#savetxt('z_ang_0.csv', z_ang, delimiter=',')
#savetxt('z_ang2.csv', ang_vel2, delimiter=',') # Angular from conversion (Inertial to body)
#savetxt('wheel_l.csv', wheel_l, delimiter=',')
#savetxt('wheel_r.csv', wheel_r, delimiter=',')
#savetxt('rlz_wheel_l.csv', rlz_wheel_l, delimiter=',')
#savetxt('rlz_wheel_r.csv', rlz_wheel_r, delimiter=',')
#savetxt('cmd_wheel_l.csv', cmd_wheel_l, delimiter=',')
#savetxt('cmd_wheel_r.csv', cmd_wheel_r, delimiter=',')


print('Done saving files')

