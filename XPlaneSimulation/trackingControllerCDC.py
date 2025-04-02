#!/usr/bin/env python2.7

#################################################################################
#                               08/06/2023
#################################################################################

import sys
import time
import signal
import rospy
import Control_utlis as utlis      #Funciones para envio de datos por UDP
import socket
import numpy as np
import scipy as sp
import scipy.io as sio
import math 
from scipy import linalg as ln
from scipy.linalg import expm, inv
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy.linalg
import xplane_ros.msg as xplane_msgs
import rosplane_msgs.msg as rosplane_msgs
from nav_msgs.msg import Odometry
from std_msgs.msg import Float64MultiArray

QUADcon = None
UDP_PORT = 49005

# Open a Socket on UDP Port 49000
UDP_IP = "127.0.0.1"
sock = socket.socket(socket.AF_INET, # Internet
                     socket.SOCK_DGRAM) # UDP
sock.bind((UDP_IP, UDP_PORT))

class QUADController():
    def __init__(self):
        self.client = utlis.XPlaneConnect()

        teleport = False
        if teleport:
            self.z_fall= -212.0   #-213.78
            quat = [1.0,0.0,0.0,0.0]
            dref = "sim/flightmodel/position/local_y"  # Z
            drefv = "sim/flightmodel/position/local_vy"  # Z
            drefa = "sim/flightmodel/position/local_ay"  # Z

            init_psi = "sim/flightmodel/position/q"
            #self.client.sendDREF(dref, self.z_fall) 
            
            #self.client.sendDREF(drefv, 0.0) 
            #self.client.sendDREF(drefa, 0.0)
            #self.client.sendDREF(init_psi, 11) 


            

        self.time_zero = rospy.Time.now()
        

        self.Plotting = False
        self.saved_quat = []
        self.saved_w = []
        self.saved_tau = []
        self.saved_xyz = []
        self.saved_z = []
        self.saved_y = []
        self.saved_refs = []
        self.saved_Thrust = []
        self.saved_Rot = []
        self.saved_Desired_Rot = []
        self.saved_Desired_Rd = []
        self.saved_rot_error = []
        self.saved_Throttles = []
        self.saved_sc_Thrust = []
        self.saved_sc_Tau = []
        self.saved_angles = []
        self.saved_phi = []
        self.saved_desired_angles = []
        self.saved_desired_angles2 = []
        self.saved_sin_Rd = []
        self.saved_sin_Rdseno = []
        self.saved_omega_Rd = []
        self.saved_omega_Rdseno = [-1]
        self.saved_omega_Rdsenos = []
        self.posErrors = []
        self.zErrors = []
        self.savedz = []
        self.savedRefz = []
        self.count = 0
        self.xv = np.zeros([6])
        self.w = np.zeros([3])
        self.myQuat = np.array([1,0,0,0])
        self.data = [\
                [25, -998, -998, -998, -998, -998, -998, -998, -998],\
            ]
        self.poscurrent = [-998, -998, -998, -998, -998, -998, -998]   
        self.timeos_before_first_measure = 0

        # Scalar basis
        self.e0 = np.array([[1, 0, 0, 0],
                       [0, 1, 0, 0],
                       [0, 0, 1, 0],
                       [0, 0, 0, 1]])

        # Vector basis
        self.e1 = np.array([[0, 0, 1, 0],
                       [0, 0, 0, 1],
                       [1, 0, 0, 0],
                       [0, 1, 0, 0]])

        self.e2 = np.array([[0, 0, 0, -1],
                       [0, 0, 1, 0],
                       [0, 1, 0, 0],
                       [-1, 0, 0, 0]])

        self.e3 = np.array([[1, 0, 0, 0],
                       [0, 1, 0, 0],
                       [0, 0, -1, 0],
                       [0, 0, 0, -1]])

        # Bivector basis
        self.e1e2 = np.array([[0, 1, 0, 0],
                         [-1, 0, 0, 0],
                         [0, 0, 0, -1],
                         [0, 0, 1, 0]])

        self.e2e3 = np.array([[0, 0, 0, 1],
                         [0, 0, -1, 0],
                         [0, 1, 0, 0],
                         [-1, 0, 0, 0]])

        self.e3e1 = np.array([[0, 0, 1, 0],
                         [0, 0, 0, 1],
                         [-1, 0, 0, 0],
                         [0, -1, 0, 0]])

        # Trivector basis
        self.e1e2e3 = np.array([[0, 1, 0, 0],
                           [-1, 0, 0, 0],
                           [0, 0, 0, 1],
                           [0, 0, -1, 0]])

        self.bpant =  np.array([[1, 0, 0, 0],
                       [0, 1, 0, 0],
                       [0, 0, 1, 0],
                       [0, 0, 0, 1]])

        self.dt = 0.01                      # Debe coincidir con el Ros Rate =  100hz
        self.g = 9.81
        self.m = 0.84                       # Xplanes 1.86 lb = 0.84 kg                            
                                            # Is Inertia I=mr^2 ???
                                            # Radius of gyration
                                            # roll = pitch = 0.52 ft, yaw = 0.75
                                            #convertion factor = 0.06715
        self.Jx = self.m * 0.52**2 * 0.06715  
        self.Jy = self.m * 0.52**2 * 0.06715  
        self.Jz = self.m * 0.75**2 * 0.06715  
        self.JJ = np.diag([self.Jz, self.Jx, self.Jy])

        self.prev_xv = np.zeros(6)
        self.prev_Trust = np.zeros(1)

        self.daant = np.zeros(3)
        self.jdthdant = np.zeros((4, 4))
        self.jdthdant2 = np.zeros((4, 4))

        #LRQ Design
        Ar = np.block([
            [np.zeros((3, 3)), np.eye(3)],
            [np.zeros((3, 6))]])
        Br = np.block([
            [np.zeros((3, 3))],
            [np.eye(3)]])
        Mr = np.block([
            [Ar, Br],
            [np.zeros((3, 9))]])
        Mdr = expm(self.dt * Mr)
        Adr = Mdr[:6, :6]
        Bdr = Mdr[:6, 6:9]

        Ap = np.block([
            [np.zeros((3, 3)), np.eye(3)],
            [np.zeros((3, 6))]])
        Bp = np.block([
            [np.zeros((3, 3))],
            [np.eye(3)]])
        Mp = np.block([
            [Ap, Bp],
            [np.zeros((3, 9))]
        ])
        Mdp = expm(self.dt * Mp)
        Adp = Mdp[:6, :6]
        Bdp = Mdp[:6, 6:9]

        # Incrementar estado - Converge mas rapido, Disminuir entrada - Converge mas rapido
# ---------------------------------------------------------------------
        Qp = np.block([
            [np.diag([4500, 4500, 2500]), np.zeros((3, 3))],
            [np.zeros((3, 3)), np.diag([1, 1, 1400])]
        ]) * 1e-4                                           # Weighting matrix for state

        Rp = np.diag([1,1,4])*1e-1                          # Weighting matrix for input
#______________________________________________________________________________-
        Qr = np.block([    
            [np.diag([0.0001,100,100])*21 , np.zeros((3, 3))],
            [np.zeros((3, 3)), np.diag([1,0.1,0.1])*1e1 ]
        ])                                                  # Weighting matrix for state

        Rr = np.diag([1,0.3,0.3]) *1e1                      # Weighting matrix for input

        self.Lr = np.array(self.discretelqr(Adr, Bdr, Qr, Rr))
        self.Lr[0, 0] = 0
        self.Lr[0, 3] = 0 

        self.Lp = np.array(self.discretelqr(Adp, Bdp, Qp, Rp))

        self.Bir = np.matmul(ln.inv(np.matmul(Br.T,Br)),Br.T)     

        rospy.Subscriber("/xplane/flightmodel/odom", Odometry, self.odomcallback)
        self.timer = rospy.Timer(rospy.Duration(0.01), self.sendToXPlane)

    def discretelqr(self,A, B, Q, R):
        X = np.matrix(scipy.linalg.solve_discrete_are(A, B, Q, R))
        K = np.matrix(scipy.linalg.inv(B.T*X*B+R)*(B.T*X*A))
        eigVals, eigVecs = scipy.linalg.eig(A-B*K)
        return K

    def expBivector(self, BV):
        BV=self.kVectorPart(BV,2)    
        BV1,BV2,BV3 = self.multiVectorParts(BV)[4:7]
        bAb=np.sqrt(BV1**2+BV2**2+BV3**2)
        if bAb>1e-7:
            sa=np.sin(bAb)/bAb
        else:
            sa=1
        expA=np.cos(bAb)*self.e0 + BV1*sa*self.e1e2 + BV2*sa*self.e2e3 + BV3*sa*self.e3e1
        expA=expA/self.multivectorNorm(expA)
        return self.kVectorPart(expA,0)+self.kVectorPart(expA,2)

    # Decomposes the multivector M into its scalar, vector, bivector, and trivector parts
    def multiVectorParts(self, M):
        m1 = np.trace((np.matmul(M,self.e1) + np.matmul(self.e1, M)) / 2) / 4
        m2 = np.trace((np.matmul(M, self.e2) + np.matmul(self.e2, M)) / 2) / 4
        m3 = np.trace((np.matmul(M, self.e3) + np.matmul(self.e3, M)) / 2) / 4
        m4 = -np.trace((np.matmul(M, self.e1e2) + np.matmul(self.e1e2, M)) / 2) / 4
        m5 = -np.trace((np.matmul(M, self.e2e3) + np.matmul(self.e2e3, M)) / 2) / 4
        m6 = -np.trace((np.matmul(M, self.e3e1) + np.matmul(self.e3e1, M)) / 2) / 4
        m7 = -np.trace((np.matmul(M, self.e1e2e3) + np.matmul(self.e1e2e3, M)) / 2) / 4
        m0 = np.trace(M - m1 * self.e1 - m2 * self.e2 - m3 * self.e3 - m4 * self.e1e2 - m5 * self.e2e3 - m6 * self.e3e1 - m7 * self.e1e2e3) / 4
        return m0, m1, m2, m3, m4, m5, m6, m7

    # Norm of a multivector M
    def multivectorNorm(self, M):
        m0, m1, m2, m3, m4, m5, m6, m7 = self.multiVectorParts(M)
        return np.sqrt(m0 ** 2 + m1 ** 2 + m2 ** 2 + m3 ** 2 + m4 ** 2 + m5 ** 2 + m6 ** 2 + m7 ** 2)

    # Extracts the k-vector part of the multivector M
    def kVectorPart(self, M, k):
        if k == 0:
            return np.trace(M)/4*self.e0
        elif k == 1:
            m1 = np.trace((np.matmul(M,self.e1)+np.matmul(self.e1,M))/2)/4
            m2 = np.trace((np.matmul(M,self.e2)+np.matmul(self.e2,M))/2)/4
            m3 = np.trace((np.matmul(M,self.e3)+np.matmul(self.e3,M))/2)/4
            return m1*self.e1 + m2*self.e2 + m3*self.e3
        elif k == 2:
            m4 = -np.trace((np.matmul(M,self.e1e2)+np.matmul(self.e1e2,M))/2)/4
            m5 = -np.trace((np.matmul(M,self.e2e3)+np.matmul(self.e2e3,M))/2)/4
            m6 = -np.trace((np.matmul(M,self.e3e1)+np.matmul(self.e3e1,M))/2)/4
            return m4*self.e1e2 + m5*self.e2e3 + m6*self.e3e1
        elif k == 3:
            return m7 * self.e1e2e3
            m7 = -np.trace((np.matmul(M,self.e1e2e3)+np.matmul(self.e1e2e3,M))/2)/4
            return m7*self.e1e2e3

    # Calculates the rotational error between desired and actual bivectors

    def rotationalError2(self, R, Rd):
        Re = np.matmul(R.T,Rd)
        Be = self.kVectorPart(Re, 2)

        y = self.multivectorNorm(Be)
        x = self.multiVectorParts(self.kVectorPart(Re, 0))[0]
        the = np.arctan2(y, x)

        if self.multivectorNorm(Be) < 1e-7:
            je = self.kVectorPart(zeros(4, 4), 2)
        else:
            je = Be / self.multivectorNorm(Be)
        jethe = self.kVectorPart(-2 * je * the, 2)

        return jethe                   


    # Vector field for rotational dynamics
    def myf(self, x, JJ):
        f = np.zeros(6)
        f[3] = (x[4] * x[5] * (JJ[1] - JJ[2]))[0]
        f[4] = (x[5] * x[3] * (JJ[2] - JJ[0]))[0]
        f[5] = (x[3] * x[4] * (JJ[0] - JJ[1]))[0]
        return np.reshape(f, (6, 1))

    # Estimates the rotation matrix R from the vector field f3
    def estR(self, f3, e3):
        Rb = 1*self.e0 + np.matmul(f3, e3)
        Rb = self.kVectorPart(Rb, 0) + self.kVectorPart(Rb, 2)
        if self.multivectorNorm(Rb)>1e-5:
            Rb=Rb/self.multivectorNorm(Rb)
        else:
            Rb=1*self.e0
        return self.kVectorPart(Rb,0)+self.kVectorPart(Rb,2)

    def refsxyz(self, count,ref):
       

        if ref == 'ramps':
            t=count#*0.001
            f=0.05
            if t<750:
                x=0.0
                dx=0.0
                self.xlast = x

                y=0.0
                dy=0.0
                self.ylast = y

            if t>750 and t<3000:
                x= 0
                dx=0
                self.xlast = x

                y= 0
                dy=0
                self.ylast = y

            if t>8000 and t<10000:
                x=self.xlast
                dx=0

                y= 0
                dy=0
                self.ylast = y
                
            else:
                x=self.xlast
                dx=0

                y=self.ylast
                dy=0

            
            ddx=0
            dddx=0
            ddddx=0

        
            ddy=0
            dddy=0
            ddddy=0

            if t<3000:
                z=-t/300.0
                dz=-1/300.0

                self.zlast = z
            else:
                z = self.zlast
                dz=0
            
            ddz=0
            dddz=0
            ddddz=0
        elif ref == 'circlegood':
            # Circle
            t=count#*0.001
            f=0.05
            if t<750:
                x=0.0
                dx=0.0
                ddx=0
                dddx=0
                ddddx=0
                self.xlast = x

                y=0.0
                dy=0.0
                ddy=0
                dddy=0
                ddddy=0
                self.ylast = y

            if t>750 and t<5750:
                x= t*0.0039 - 2.9
                dx=0.0039
                ddx=0
                dddx=0
                ddddx=0
                self.xlast = x

                y= t*0.0039 - 2.9
                dy=0.0039
                ddy=0
                dddy=0
                ddddy=0
                self.ylast = y

            if t>5750 and t<20000:
                x=self.xlast
                dx=0
                ddx=0
                dddx=0
                ddddx=0

                x_center = 0
                y_center = 19.5
                x_start = 19.5
                y_start = 19.5

                # Calculate the angle corresponding to (x_start, y_start)
                angle_start = 0#np.arctan2(y_start - y_center, x_start - x_center)
               
                r = 19.5
                circleTime = (t - 5750) * 0.0006
                x = r * np.cos(circleTime-angle_start)


                y= 0#t/800.0 - 10
                dy=0#1/800.0
                ddy=0
                dddy=0
                ddddy=0


                y = r * np.sin(circleTime-angle_start) + y_center



                self.ylast = y
                
            else:
                x=self.xlast
                dx=0
                ddx=0
                dddx=0
                ddddx=0

                y=self.ylast
                dy=0
                ddy=0
                dddy=0
                ddddy=0

            if t<5750:
                z=-t*0.0052
                dz=-0.0052

                self.zlast = z
            elif t<12000:
                z = self.zlast
                dz=0
            else:
                z = self.zlast
                dz=0
            
            ddz=0
            dddz=0
            ddddz=0

        elif ref == 'circle':
            # Circle
            t=count#*0.001
            f=0.05
            if t<750:
                x=0.0
                dx=0.0
                ddx=0
                dddx=0
                ddddx=0
                self.xlast = x

                y=0.0
                dy=0.0
                ddy=0
                dddy=0
                ddddy=0
                self.ylast = y

            if t>750 and t<5750:
                x= t*0.0039 - 2.9
                dx=0.0039
                ddx=0
                dddx=0
                ddddx=0
                self.xlast = x

                y= t*0.0039 - 2.9
                dy=0.0039
                ddy=0
                dddy=0
                ddddy=0
                self.ylast = y

            if t>5750 and t<20000:
                x=self.xlast

                x_center = 0
                y_center = 19.5
                x_start = 19.5
                y_start = 19.5

                # Calculate the angle corresponding to (x_start, y_start)
                angle_start = 0#np.arctan2(y_start - y_center, x_start - x_center)
               
                r = 19.5
                circleTime = (t - 5750) * 0.0006                    
                x = r * np.cos(circleTime-angle_start)
                dx = -r * np.sin(circleTime-angle_start)* 0.0006
                ddx = 0 # -r * np.cos(circleTime-angle_start)
                dddx = 0 # r * np.sin(circleTime-angle_start)
                ddddx = 0 # r * np.cos(circleTime-angle_start)

                y = r * np.sin(circleTime-angle_start) + y_center
                dy = r * np.cos(circleTime-angle_start)* 0.0006 #1/800.0
                ddy = 0 # -r * np.sin(circleTime-angle_start)
                dddy = 0 # -r * np.cos(circleTime-angle_start)
                ddddy = 0 # r * np.sin(circleTime-angle_start)

                self.ylast = y
                
            else:
                x=self.xlast
                dx=0
                ddx=0
                dddx=0
                ddddx=0

                y=self.ylast
                dy=0
                ddy=0
                dddy=0
                ddddy=0

            if t<5750:
                z=-t*0.0052
                dz=-0.0052

                self.zlast = z
            elif t<12000:
                z = self.zlast
                dz=0
            else:
                z = self.zlast
                dz=0
            
            ddz=0
            dddz=0
            ddddz=0

        elif ref == 'rose':
            # 4-petal Rose Reference
            t = count  # *0.001
            a = 50
            n = 2
            d = 1
            f = 5e-5

            k = n / d

            if t < 750:  # Initial hold
                x = 0.0
                dx = 0.0
                ddx = 0
                dddx = 0
                ddddx = 0
                self.xlast = x

                y = 0.0
                dy = 0.0
                ddy = 0
                dddy = 0
                ddddy = 0
                self.ylast = y

            elif t > 750 and t < 5750:  # Diagonal Ascension
                x = t * 0.0039 - 2.9
                dx = 0.0039
                ddx = 0
                dddx = 0
                ddddx = 0
                self.xlast = x

                y = t * 0.0039 - 2.9
                dy = 0.0039
                ddy = 0
                dddy = 0
                ddddy = 0
                self.ylast = y

            elif t > 5750:  # 4-Petal Rose Motion
                x = a * np.cos(2 * f * np.pi * (t - 5750)) * np.cos(2 * f * k * np.pi * (t - 5750)) - 30.5
                dx = -2 * a * f * k * np.pi * np.sin(2 * f * k * np.pi * (t - 5750)) * np.cos(2 * f * np.pi * (t - 5750)) - \
                    2 * a * f * np.pi * np.sin(2 * f * np.pi * (t - 5750)) * np.cos(2 * f * k * np.pi * (t - 5750))
                ddx = 0
                dddx = 0
                ddddx = 0

                y = a * np.sin(2 * f * np.pi * (t - 5750)) * np.cos(2 * f * k * np.pi * (t - 5750)) + 20
                dy = -2 * a * f * k * np.pi * np.sin(2 * f * np.pi * (t - 5750)) * np.sin(2 * f * k * np.pi * (t - 5750)) + \
                    2 * a * f * np.pi * np.cos(2 * f * np.pi * (t - 5750)) * np.cos(2 * f * k * np.pi * (t - 5750))
                ddy = 0
                dddy = 0
                ddddy = 0

            else:
                x=self.xlast
                dx=0
                ddx=0
                dddx=0
                ddddx=0

                y=self.ylast
                dy=0
                ddy=0
                dddy=0
                ddddy=0

            if t < 5750:
                z = -t * 0.0052
                dz = -0.0052
                self.zlast = z
            elif t < 12000:
                z = self.zlast
                dz = 0
            else:
                z = self.zlast
                dz = 0

            ddz = 0
            dddz = 0
            ddddz = 0

        elif ref == 'fall':
            # free fall recovery
            t = count 

            x = -0.375
            dx = 0.0
            ddx = 0
            dddx = 0
            ddddx = 0

            y = -0.02
            dy = 0.0
            ddy = 0
            dddy = 0
            ddddy = 0

            # z = -213.78+20
            # dz = 0
            ddz = 0
            dddz = 0
            ddddz = 0    

            if t < 35000:
                z = -13
                dz = 0
                self.zlast = z
                # z = -t * 0.0052
                # dz = -0.0052
                # self.zlast = z
            elif t < 37000:
                z = self.zlast
                dz = 0
            else:
                z = self.zlast
                dz = 0

            


        else:
            print('No ref given')    

        return [x, y, z, dx, dy, dz, ddx, ddy, ddz, dddx, dddy, dddz, ddddx, ddddy, ddddz]

    def xv_dot(self, xv, T, bt):                 
        ve3 = np.array([0, 0, 1])

        bt = bt / self.multivectorNorm(bt)
        btaux = self.kVectorPart(bt,1)
        bt = self.multiVectorParts(btaux)[1:4]
        bt = np.array([bt])
        dxv = np.zeros(6)
        dxv[0:3] = xv[3:6]
        dxv[3:6] = self.g*ve3 - T/self.m * bt
        return dxv


    def funRfin2(self, count):


        if 4000 < count < 6000:
            #j1 = 0.1
            dj1 = 0
            ddj1 = 0

            f0 = 0.1   # Start frequency (Hz)
            f1 = 10  # End frequency (Hz)
            T = 40.0  # Duration (seconds)

            # Softening factor: reduce rapid frequency changes
            a = 1  # Controls how aggressively frequency increases
            f_t = f0 + (f1 - f0) * (count / T)**a

            # Compute phase
            phi_t = 2 * np.pi * np.cumsum(f_t * 0.01)  # Integrate frequency with time step 0.01

            j1 = 0.4 * np.sin(phi_t)

        elif 6000 < count < 8000:
            #j1 = 0.1
            dj1 = 0
            ddj1 = 0

            f0 = 10   # Start frequency (Hz)
            f1 = 30  # End frequency (Hz)
            T = 40.0  # Duration (seconds)

            # Softening factor: reduce rapid frequency changes
            a = 0.8  # Controls how aggressively frequency increases
            f_t = f0 + (f1 - f0) * (count / T)**a

            # Compute phase
            phi_t = 2 * np.pi * np.cumsum(f_t * 0.01)  # Integrate frequency with time step 0.01

            j1 = 0.4 * np.sin(phi_t)   
        
        elif 8000 < count < 10000:
            #j1 = 0.1
            dj1 = 0
            ddj1 = 0

            f0 = 30   # Start frequency (Hz)
            f1 = 1  # End frequency (Hz)
            T = 40.0  # Duration (seconds)

            # Softening factor: reduce rapid frequency changes
            a = 0.8  # Controls how aggressively frequency increases
            f_t = f0 + (f1 - f0) * (count / T)**a

            # Compute phase
            phi_t = 2 * np.pi * np.cumsum(f_t * 0.01)  # Integrate frequency with time step 0.01

            j1 = 0.4 * np.sin(phi_t)
                    
        elif 10000 < count < 12000:
            #j1 = 0.1
            dj1 = 0
            ddj1 = 0

            f0 = 1   # Start frequency (Hz)
            f1 = 10  # End frequency (Hz)
            T = 40.0  # Duration (seconds)

            # Softening factor: reduce rapid frequency changes
            a = 1  # Controls how aggressively frequency increases
            f_t = f0 + (f1 - f0) * (count / T)**a

            # Compute phase
            phi_t = 2 * np.pi * np.cumsum(f_t * 0.01)  # Integrate frequency with time step 0.01

            j1 = 0.4 * np.sin(phi_t)        

        elif 12000 < count < 12500:
            j1 = 0.0
            dj1 = 0
            ddj1 = 0

        elif 12500 < count < 14000:
            j1 = 0.1
            dj1 = 0
            ddj1 = 0


        elif 14000 < count < 16000:
            j1 = -0.1
            dj1 = 0
            ddj1 = 0

        elif 16000 < count < 18000:
            j1 = -0.0001*(count-16000)
            dj1 = 0
            ddj1 = 0  


        elif 18000 < count < 20000:

            dj1 = 0
            ddj1 = 0

            f0 = 1   # Start frequency (Hz)
            f1 = 10  # End frequency (Hz)
            T = 160.0  # Duration (seconds)

            # Softening factor: reduce rapid frequency changes
            a = 0.9  # Controls how aggressively frequency increases
            f_t = f0 + (f1 - f0) * (count / T)**a

            # Compute phase
            phi_t = 2 * np.pi * np.cumsum(f_t * 0.01)  # Integrate frequency with time step 0.01

            j1 = 0.3 * np.sin(phi_t) 
     
        

            
        else:
            j1 = 0
            dj1 = 0
            ddj1 = 0
            self.j1last = j1

        Rfin = self.expBivector(-j1 * self.e2e3 / 2)         #Este rotor significa nada? mantenerse quieto?

        Rfin = Rfin / self.multivectorNorm(Rfin)
        dRfin = -dj1 * np.matmul(self.e1e2 / 2,Rfin)
        ddRfin = -ddj1 * np.matmul(self.e1e2 / 2,Rfin)  +  dj1**2 * np.matmul(self.e1e2**2,Rfin / 4)

        return Rfin, dRfin, ddRfin

    def myang(self, R):
        R=R/self.multivectorNorm(R)
        q0=self.multiVectorParts(R)[0]
        q3=self.multiVectorParts(R)[4]
        q1=self.multiVectorParts(R)[5]
        q2=self.multiVectorParts(R)[6]
        
        phi=math.atan2(2*(q0*q1+q2*q3),1-2*(q1**2+q2**2))
        if (abs(2*(q0*q2-q1*q3))>=1):
            theta=0
        else:
            theta=math.asin(2*(q0*q2-q1*q3))
        psi=math.atan2(2*(q0*q3+q1*q2),1-2*(q2**2+q3**2))
        return phi,theta,psi   

    def power_even(self, A, p):
        powerAux = (A)
        if (self.multivectorNorm(A)) < 1e-15 and p != 0:
            return (0)
        elif p == 0:
            return self.e0  
        else:
            bivect = self.kVectorPart(A,2)
            if self.multivectorNorm(bivect) < 1e-15:
                powerAuxscalar = (self.multiVectorParts(A)[0]**p) * self.e0
                return (powerAuxscalar)                                 # Scalar power case
            else:
                return (self.exp_even(p * self.log_even(A)))                    

    def log_even(self, A):

        q0=self.multiVectorParts(A)[0]
        q3=self.multiVectorParts(A)[4]
        q1=self.multiVectorParts(A)[5]
        q2=self.multiVectorParts(A)[6]

        parts = [q0,q3,q1,q2]
        norm_A = np.linalg.norm(parts)  

        val = (np.log(norm_A))*self.e0

        scalar_part = parts[0]  
        bivector_part = parts[1:] 

        if np.linalg.norm(bivector_part) > 0:
           
            log_norm = np.arccos(scalar_part / norm_A) / np.linalg.norm(bivector_part)
            bivector_log = np.array(bivector_part) * log_norm

            #This is not in matrix form IT SHOULD RETURN A MULTIVECTOR EVEN

            bivector_log = bivector_log[0] * self.e1e2 + bivector_log[1] * self.e2e3 + bivector_log[2] * self.e3e1   

            bivector_log = self.kVectorPart(bivector_log,2)   #prob not necessary
           
            result = val + bivector_log   
            
        else:
            result = val      
        return result

    def exp_even(self, A):
        a0 = self.kVectorPart(A,0)

        BV=self.kVectorPart(A,2)    
        a23,a31,a12 = self.multiVectorParts(BV)[4:7]   #Double check order
        a12 = -a12
        a23 = -a23

        aS = -a12**2 -a23**2 - a31**2

        magc = -aS
        aP = 0
        aM = np.sqrt(-aS)

        b0 = np.cos(aM) * np.cosh(aP)
        b12 = np.cosh(aP) * np.sin(aM) * (aM * a12) + np.sinh(aP) * np.cos(aM) * (aP * a12)
        b13 = -np.cosh(aP) * np.sin(aM) * (-aM * a31) + np.sinh(aP) * np.cos(aM) * (aP * a31)
        b23 = np.cosh(aP) * np.sin(aM) * (aM * a23) + np.sinh(aP) * np.cos(aM) * (aP * a23)

        alpha = np.exp(a0)

        result = alpha*b0*self.e0     #el scalar * alpha*b0 con 0 en el bivector y cero en todo lo demas

        if magc > 0:

            s1 = alpha*b0*self.e0
            b1 = -alpha * b23 / magc * self.e1e2     #Check if this is true and check order
            b2 = alpha * b13 / magc * self.e2e3
            b3 = -alpha * b12 / magc * self.e3e1

            result = s1+b1+b2+b3

        return result

    def extractjth(self, Re):
        jethe = -2 * self.log_even(Re)  
        return jethe

    def control(self, xv, w, Rq, m, g, JJ, Bir, Lp, Lr, count):
        # Define reference
        # 'ramps' : rampas
        # 'circle' : ascenso y trayectoria circular
        ref = 'ramps'

        # ------------------------------------------------------------
        print(count)
        refs = np.array(self.refsxyz(count,ref))

        posErr = xv - refs[0:6]                         # E 30
        self.posErrors.append(posErr[0:6])
        self.zErrors.append([posErr[2], posErr[5]])
        self.savedz.append(xv[2])
        self.savedRefz.append(refs[2])

        #test PD
        error_z = xv[2] - refs[2]                       # E 37
        e_error_z = xv[5] - refs[5]                     # Part of E 38
        
            # E 38

        if ref == 'circlegood':
            kpdx=-0.31*1.3
            kddx=-0.63*1.8

            kpdy=-0.47
            kddy=-0.58
            kpdz=0.02*8*17
            kddz=0.06*4*23
        elif ref == 'circle':
            # kpdx=-0.31*1.3
            # kddx=-0.63*1.8

            # kpdy=-0.47
            # kddy=-0.58

            kpdx=-1.12
            kddx=-1.92

            kpdy=-3.4*0.5
            kddy=-3.8*0.5
            kpdz=0.02*8*17
            kddz=0.06*4*23
        elif ref == 'rose':
            # kpdx=-0.31*1.3
            # kddx=-0.63*1.8

            # kpdy=-0.47
            # kddy=-0.58

            kpdx=-1.12
            kddx=-1.92

            kpdy=-3.4*0.5
            kddy=-3.8*0.5   
            kpdz=0.02*8*17
            kddz=0.06*4*23 
        elif ref == 'ramps':
            kpdx=-1.12 *0 
            kddx=-1.92*0

            kpdy=-3.4*0
            kddy=-3.8*0
            kpdz=0.02*8*17
            kddz=0.06*4*23
        elif ref == 'fall':
            if count >5000:
                kpdx=-3.4 * 0.0
                kddx=-3.8 * 0.0
                kpdy=-3.4 * 0
                kddy=-3.8  * 0
                kpdz=0.02*8*21 
                kddz=0.06*4*24
            else:
                kpdx=-3.4 * 0.5
                kddx=-3.8 * 0.5
                kpdy=-3.4* 0
                kddy=-3.8 * 0
                kpdz=0.02*8*21 
                kddz=0.06*4*24 
        else:
            kpdx=None
            kddx=None
            kpdy=None
            kddy=None  
            kpdz=None
            kddz=None

        Lp = np.array([[kpdx, 0, 0, kddx, 0 ,0],[0, kpdy, 0, 0, kddy, 0],[0, 0, kpdz, 0, 0, kddz]])

        dv = np.matmul(-Lp, np.reshape((posErr), (6, 1)))-np.vstack([0,0,g])+np.reshape(np.array(refs[6:9]), (3, 1))  # E 34

        dv = dv[0] * self.e1 + dv[1] * self.e2 + dv[2] * self.e3

        Trust = m * self.multivectorNorm(-dv)                   #Missing g*e3
   
        Rq_mat = Rq[0] * self.e0 + Rq[1] * self.e1e2 + Rq[2] * self.e2e3 + Rq[3] * self.e3e1        # Quadcopter angular position
        Rq_mat = Rq_mat / self.multivectorNorm(Rq_mat)

        actualAngles = self.myang(Rq_mat)
        self.saved_angles.append(actualAngles)
        self.saved_phi.append(actualAngles[0])

        b3rot = -dv#m*(-dv) / Trust           

        if self.multivectorNorm(b3rot) > 1e-9:                                  # Recovering the Desired Rotor (Dorian)
            b3rot = b3rot / self.multivectorNorm(b3rot)
            Rd = self.estR(b3rot, self.e3)
        else:
            Rd = 1*self.e0  

        Rdes, dRdes, ddRdes = self.funRfin2(count)                              #Heading reference
        Rdes = self.kVectorPart(Rdes, 0) + self.kVectorPart(Rdes, 2)
        Rdes = Rdes / self.multivectorNorm(Rdes)
       
        # R1des = self.multiVectorParts(self.kVectorPart(Rdes, 0))[0]
        # R2des = self.multiVectorParts(self.kVectorPart(Rdes, 2))[4]
        # R3des = self.multiVectorParts(self.kVectorPart(Rdes, 2))[5]
        # R4des = self.multiVectorParts(self.kVectorPart(Rdes, 2))[6]
        # myDesiredRot = np.array([R1des,R2des,R3des,R4des])
        # self.saved_Desired_Rot.append(myDesiredRot)


        w = w[0]*self.e1e2 + w[1]*self.e2e3 + w[2]*self.e3e1   
        w = self.kVectorPart(w,2) 
        w4, w5, w6 = self.multiVectorParts(w)[4:7]

        # -- I BELIEVE THE ANGULAR VELOCTITIES ARE IN THE BODY FRAME

        wsaves = np.array([w4,w5,w6])
        self.saved_w.append(wsaves)

        Rd = np.matmul(Rd,Rdes)                             #Including rotational rotor and heading rotor
        Rd = Rd / self.multivectorNorm(Rd)


        R1d = self.multiVectorParts(self.kVectorPart(Rd, 0))[0]
        R2d = self.multiVectorParts(self.kVectorPart(Rd, 2))[4]
        R3d = self.multiVectorParts(self.kVectorPart(Rd, 2))[5]
        R4d = self.multiVectorParts(self.kVectorPart(Rd, 2))[6]
        myDesiredRd = np.array([R1d,R2d,R3d,R4d])
        self.saved_Desired_Rd.append(myDesiredRd)

        # TODO: Calcular jd thetad
        jdthd = self.extractjth(Rd)
        

        desiredAngles2 = self.myang(Rd)
        self.saved_desired_angles2.append(desiredAngles2)


        # Errors
        Error = self.rotationalError2(Rq_mat,Rd)
  
        Er_parts = self.multiVectorParts(Error)
        Er4, Er5, Er6 = Er_parts[4:7]                           #e1e2 e2e3 e3e1

        Errors = np.vstack([Er4,Er5,Er6])[:,0]
        self.saved_rot_error.append(Errors)


     

        # Rotational Gains
        
        if ref == 'circlegood':
            Lr = np.array([[65.3, 0, 0, 5.01, 0 ,0],[0, 115.5, 0, 0, 8.16, 0],[0, 0, 115.5, 0, 0, 19.16]])
        elif ref == 'circle':
            Lr = np.array([[165.3, 0, 0, 7.01, 0 ,0],[0, 275.5, 0, 0, 8.16, 0],[0, 0, 375.5, 0, 0, 19.16]])
        elif ref == 'rose':
            Lr = np.array([[165.3, 0, 0, 7.01, 0 ,0],[0, 275.5, 0, 0, 8.16, 0],[0, 0, 375.5, 0, 0, 22.16]])
        elif ref == 'fall':
            if 5000<count < 5245:
                Lr = np.array([[0, 0, 0, 0, 0 ,0],[0, 400.5, 0, 0, 5.16, 0],[0, 0, 0, 0, 0, 0]])  
            else:
                Lr = np.array([[10, 0, 0, 2, 0 ,0],[0, 100.5, 0, 0, 20.16, 0],[0, 0, 100.5, 0, 0, 18.16]])  
        elif ref == 'ramps':
            Lr = np.array([[165.3, 0, 0, 7.01, 0 ,0],[0, 275.5, 0, 0, 8.16, 0],[0, 0, 175.5, 0, 0, 12.16]])  
        else:
            Lr = None
        # ----------------------

        # self.current_time = rospy.Time.now()
        # real_time = self.current_time-self.time_zero
        # real_time=real_time.to_sec()

        # #Test derivative with a sine
        # jdthdseno = np.cos(real_time)*self.e0 + np.cos(real_time)*self.e1e2 + 0*self.e2e3 + 0*self.e3e1
        # jdthd = np.sin(real_time)*self.e0 + np.sin(real_time)*self.e1e2 + 0*self.e2e3 + 0*self.e3e1

        sind = self.multiVectorParts(self.kVectorPart(jdthd, 0))[0]
        sin2d = self.multiVectorParts(self.kVectorPart(jdthd, 2))[4]
        sin3d = self.multiVectorParts(self.kVectorPart(jdthd, 2))[5]
        sin4d = self.multiVectorParts(self.kVectorPart(jdthd, 2))[6]
        mysinRd = np.array([sind,sin2d,sin3d,sin4d])
        self.saved_sin_Rd.append(mysinRd)

        # ANALITICA
        # dsind = self.multiVectorParts(self.kVectorPart(jdthdseno, 0))[0]
        # dsin2d = self.multiVectorParts(self.kVectorPart(jdthdseno, 2))[4]
        # dsin3d = self.multiVectorParts(self.kVectorPart(jdthdseno, 2))[5]
        # dsin4d = self.multiVectorParts(self.kVectorPart(jdthdseno, 2))[6]
        # dmysinRd = np.array([dsind,dsin2d,dsin3d,dsin4d])

        # self.saved_sin_Rdseno.append(dmysinRd)

        alpha = 0.1  # Smaller alpha means more smoothing

        if count > 1:
            if count > 2:
                myOmegadsenos = (3 / 2. * mysinRd - 2 * self.jdthdantsenos + 1 / 2. * self.jdthdant2senos) / (self.dt)
                
                # Apply the low-pass filter
                if len(self.saved_omega_Rdsenos) > 0:
                    myOmegadsenos = alpha * myOmegadsenos + (1 - alpha) * self.saved_omega_Rdsenos[-1]

                self.saved_omega_Rdsenos.append(myOmegadsenos)

            self.jdthdant2senos = self.jdthdantsenos  # Store old value before updating
            self.jdthdantsenos = mysinRd
        else:
            self.jdthdantsenos = mysinRd


        # ----------------------
        Lr = np.matmul(JJ,Lr)  #Including JJ
        #TODO incorporate effect of Lr in gains

        #Tracking
        x = np.reshape(np.concatenate(([Er4, Er5, Er6], [w4-sin2d, w5-sin3d, w6-sin4d])), (6, 1))
        v = np.matmul(-Lr, x)
  
        tau = v
   
        tauvec = tau[0] * self.e1e2 + tau[1] * self.e2e3 + tau[2] * self.e3e1

        tauB_parts = self.kVectorPart(tauvec,2)
        tauB_parts = self.multiVectorParts(tauB_parts)[4:7]
        
        tauB = np.array(tauB_parts)

        self.saved_refs.append(refs[0:3])
        
        return tauB, Trust

    def odomcallback(self, data):
        self.quaternion = data.pose.pose.orientation  #Supposed to be NED
        self.q0 = self.quaternion.w
        self.q1 = self.quaternion.x
        self.q2 = self.quaternion.y
        self.q3 = self.quaternion.z
        self.myQuat = np.array([self.q0,self.q3,self.q1,self.q2])

        self.saved_quat.append(self.myQuat)
         
        self.xi = data.pose.pose.position
        self.x = self.xi.x
        self.y = self.xi.y
        self.z = self.xi.z
        
        self.dxi = data.twist.twist.linear
        self.dx = self.dxi.x
        self.dy = self.dxi.y
        self.dz = self.dxi.z

        self.angularRates = data.twist.twist.angular
        self.p = self.angularRates.x
        self.q = self.angularRates.y
        self.r = self.angularRates.z

        self.xv = np.array([self.x,self.y,self.z,self.dx,self.dy,self.dz])
        self.saved_xyz.append(self.xv)
        self.saved_z.append(self.xv[2])
        self.saved_y.append(self.xv[1])

        #     p  -> e2e3           q ->  e3e1       r -> e1e2
        self.w = np.array([self.r, self.p, self.q]) 

        self.tausB, Thrust = self.control(self.xv, self.w, self.myQuat,  self.m, self.g, self.JJ, self.Bir, self.Lp, self.Lr,self.count)

        self.saved_tau.append(self.tausB)
           
        self.saved_Thrust.append(Thrust)
      
        weight = self.m*self.g                          # = 8.24 N  Thats the max thrust output
        hoverThrottle = 0.0864                          # Throttle per motor needed to maintain hover
        normFactor = hoverThrottle/weight

        scaledThrust = Thrust*normFactor
        scaledTau = self.tausB*normFactor


        
        self.saved_sc_Tau.append(scaledTau)
        self.saved_sc_Thrust.append(scaledThrust)

        if 4000 < self.count < 7000:
            factor = 0.000
            self.Throttle1 = scaledThrust + scaledTau[0] - scaledTau[1] + scaledTau[2] + factor
            self.Throttle2 = scaledThrust - scaledTau[0] - scaledTau[1] - scaledTau[2] + factor
            self.Throttle3 = scaledThrust + scaledTau[0] + scaledTau[1] - scaledTau[2] - factor
            self.Throttle4 = scaledThrust - scaledTau[0] + scaledTau[1] + scaledTau[2] - factor
            Throttles = [self.Throttle1, self.Throttle2, self.Throttle3, self.Throttle4]
        else:
            self.Throttle1 = scaledThrust + scaledTau[0] - scaledTau[1] + scaledTau[2]
            self.Throttle2 = scaledThrust - scaledTau[0] - scaledTau[1] - scaledTau[2]
            self.Throttle3 = scaledThrust + scaledTau[0] + scaledTau[1] - scaledTau[2]
            self.Throttle4 = scaledThrust - scaledTau[0] + scaledTau[1] + scaledTau[2]
            Throttles = [self.Throttle1, self.Throttle2, self.Throttle3, self.Throttle4]

        self.saved_Throttles.append(Throttles)

        self.data = [\
                [25,self.Throttle1, self.Throttle2, self.Throttle3, self.Throttle4, -998, -998, -998, -998],\
           ]

        #self.poscurrent = [37.524, -122.06899, 2500, 0, 0, 0, 1.0]

        althold = False
        if althold:
            self.z_hold= -200.0   #-213.78
            dref = "sim/flightmodel/position/local_y"  # Z
            self.client.sendDREF(dref, self.z_hold) 

        
 
    
           
        self.count += 1

    def sendToXPlane(self, event):
        motors = self.data 
        
        #print 'Motors: %s' %motors[0][1:5]
        #print('pre time',rospy.get_time())
        self.client.sendDATA(motors)



        # try to publish it in another node to verify the hz and echo
        #motor_data = [float(x) for x in motors[0]]
        #myMsg = Float64MultiArray()
        #myMsg.data = motor_data
        #motorSignals.publish(myMsg)

def save_data_to_mat(quadcon):
    # Convert lists to numpy arrays
  

    savedz = np.array(quadcon.savedz)
    saved_y = np.array(quadcon.saved_y)
    saved_phi = np.array(quadcon.saved_phi)
    
    
    
    # Generate the count (time) vector
    time_vector = np.arange(len(savedz)) * quadcon.dt  # Assuming dt is the time step

    # Save to a .mat file
    sio.savemat('quadcopter_data_workingcase.mat', {
        'time': time_vector,
        'saved_y': saved_y,
        'saved_z': savedz,
        'saved_phi': saved_phi
        
    })
    
    print("Data saved to quadcopter_data.mat")    

def signal_handler(sig, frame):
    """ Handle keyboard interrupt """
    print("\nInterrupt received, plotting quaternion data...")
    if QUADcon:

        save_data_to_mat(QUADcon)

    #    plot_quaternion_data(QUADcon.saved_omega_Rd)
        #plot_w_data(QUADcon.saved_w)
        # plot_tau_data(QUADcon.saved_tau)
        plot_sc_tau_data(QUADcon.saved_sc_Tau)
        plot_xyz_data(QUADcon.saved_xyz, QUADcon.saved_refs)
        plot_xyz_errors(QUADcon.posErrors)
        # plot_thrust_data(QUADcon.saved_Thrust)
        plot_sc_thrust_data(QUADcon.saved_sc_Thrust)
        plot_xyz_data_3d(QUADcon.saved_xyz, QUADcon.saved_refs)

    #    plot_Rot_data(QUADcon.saved_sin_Rd)

        plot_error_data(QUADcon.saved_rot_error)
    #    plot_RotNQuat_data(QUADcon.saved_sin_Rd[1:len(QUADcon.saved_omega_Rd)+1], QUADcon.saved_omega_Rd)
        plot_motor_data(QUADcon.saved_Throttles)

        #plot_angles(QUADcon.saved_angles, QUADcon.saved_desired_angles)        #Rdes
        plot_angles(QUADcon.saved_angles, QUADcon.saved_desired_angles2)       #Rot   Im not sure about the angles
        plot_angleszoom(QUADcon.saved_angles, QUADcon.saved_desired_angles2) 
        plot_allrots_data(QUADcon.saved_quat, QUADcon.saved_Desired_Rd)
        plot_allrots_dataZOOM(QUADcon.saved_quat, QUADcon.saved_Desired_Rd)
        #plot_prueba(QUADcon.saved_sin_Rd, QUADcon.saved_sin_Rdseno)

        #plot_2rots_data(QUADcon.saved_sin_Rd, QUADcon.saved_omega_Rdsenos)

        #plot_prueba2(QUADcon.saved_omega_Rdsenos,QUADcon.saved_sin_Rdseno)

    sys.exit(0)

def plot_prueba(quat_data, Rot):
    """ Plot quaternion data """
    quat_data = np.array(quat_data[:len(Rot)])      #Rd
    Rot = np.array(Rot)                  #Rot is omegad
    time_steps = np.arange(len(Rot))
    
    plt.figure()
    plt.subplot(4, 1, 1)
    plt.plot(time_steps, quat_data[:, 0], label='Rd0')
    plt.plot(time_steps, Rot[:, 0], label='der0')
    plt.ylabel('R0')
    plt.legend()

    plt.subplot(4, 1, 2)
    plt.plot(time_steps, quat_data[:, 1], label='Rd1')
    plt.plot(time_steps, Rot[:, 1], label='der1')
    plt.ylabel('R1')
    plt.legend()

    plt.subplot(4, 1, 3)
    plt.plot(time_steps, quat_data[:, 2], label='Rd2')
    plt.plot(time_steps, Rot[:, 2], label='der2')
    plt.ylabel('R2')
    plt.legend()

    plt.subplot(4, 1, 4)
    plt.plot(time_steps, quat_data[:, 3], label='Rd3')
    plt.plot(time_steps, Rot[:, 3], label='der3')
    plt.ylabel('R3')
    plt.legend()

    plt.xlabel('Time Step')
    plt.tight_layout()
    plt.savefig('XPlane_Rd_and_analitic.png')
    plt.close()

def plot_prueba2(omega, seno):
    """ Plot quaternion data """
    quat_data = np.array(omega)
    seno_data = np.array(seno[:len(quat_data)])  # Fixing indexing issue
    time_steps = np.arange(len(quat_data))

    plt.figure()
    
    plt.subplot(4, 1, 1)
    plt.plot(time_steps, quat_data, label='omega', color='red')
    plt.plot(time_steps, seno_data, label='seno')
    plt.ylabel('q0')
    plt.legend(loc='upper right')  # Move legend to the right

    plt.subplot(4, 1, 2)
    plt.plot(time_steps, seno_data, label='seno')
    plt.ylabel('q1')
    plt.legend(loc='upper right')  # Move legend to the right

    plt.xlabel('Time Step')
    plt.tight_layout()
    plt.savefig('XPlane_prueba_second.png')
    plt.close()

def plot_2rots_data(quat_data, Rot):
    """ Plot quaternion data """
    quat_data = np.array(quat_data[:len(Rot)])      #Rd
    Rot = np.array(Rot)                  #Rot is omegad
    time_steps = np.arange(len(Rot))
    
    plt.figure()
    plt.subplot(4, 1, 1)
    plt.plot(time_steps, quat_data[:, 0], label='Rd0')
    plt.plot(time_steps, Rot[:, 0], label='der0')
    plt.ylabel('R0')
    plt.legend()

    plt.subplot(4, 1, 2)
    plt.plot(time_steps, quat_data[:, 1], label='Rd1')
    plt.plot(time_steps, Rot[:, 1], label='der1')
    plt.ylabel('R1')
    plt.legend()

    plt.subplot(4, 1, 3)
    plt.plot(time_steps, quat_data[:, 2], label='Rd2')
    plt.plot(time_steps, Rot[:, 2], label='der2')
    plt.ylabel('R2')
    plt.legend()

    plt.subplot(4, 1, 4)
    plt.plot(time_steps, quat_data[:, 3], label='Rd3')
    plt.plot(time_steps, Rot[:, 3], label='der3')
    plt.ylabel('R3')
    plt.legend()

    plt.xlabel('Time Step')
    plt.tight_layout()
    plt.savefig('XPlane_Rd_and_der_plot.png')
    plt.close()


def plot_quaternion_data(quat_data):
    """ Plot quaternion data """
    quat_data = np.array(quat_data)
    time_steps = np.arange(len(quat_data))
    
    plt.figure()
    plt.subplot(4, 1, 1)
    plt.plot(time_steps, quat_data[:, 0], label='w')
    plt.ylabel('q0')
    plt.legend()

    plt.subplot(4, 1, 2)
    plt.plot(time_steps, quat_data[:, 1], label='x')
    plt.ylabel('q1')
    plt.legend()

    plt.subplot(4, 1, 3)
    plt.plot(time_steps, quat_data[:, 2], label='y')
    plt.ylabel('q2')
    plt.legend()

    plt.subplot(4, 1, 4)
    plt.plot(time_steps, quat_data[:, 3], label='z')
    plt.ylabel('q3')
    plt.legend()

    plt.xlabel('Time Step')
    plt.tight_layout()
    plt.savefig('XPlane_quaternion_plot.png')
    plt.close()

def plot_motor_data(motor_data):
    """ Plot motor data """
    motor_data = np.array(motor_data)
    time_steps = np.arange(len(motor_data))
    vector_zero = [0 for i in range(len(motor_data))]
    
    plt.figure()
    plt.subplot(4, 1, 1)
    plt.plot(time_steps, motor_data[:, 0], label='M1')
    plt.plot(time_steps, vector_zero)
    plt.ylabel('M1')
    plt.legend()

    plt.subplot(4, 1, 2)
    plt.plot(time_steps, motor_data[:, 1], label='M2')
    plt.plot(time_steps, vector_zero)
    plt.ylabel('M2')
    plt.legend()

    plt.subplot(4, 1, 3)
    plt.plot(time_steps, motor_data[:, 2], label='M3')
    plt.plot(time_steps, vector_zero)
    plt.ylabel('M3')
    plt.legend()

    plt.subplot(4, 1, 4)
    plt.plot(time_steps, motor_data[:, 3], label='M4')
    plt.plot(time_steps, vector_zero)
    plt.ylabel('M4')
    plt.legend()

    plt.xlabel('Time Step')
    plt.tight_layout()
    plt.savefig('XPlane_motor_plot.png')
    plt.close()

def plot_Rot_data(quat_data):
    """ Plot quaternion data """
    quat_data = np.array(quat_data)
    time_steps = np.arange(len(quat_data))
    
    plt.figure()
    plt.subplot(4, 1, 1)
    plt.plot(time_steps, quat_data[:, 0], label='R0')
    plt.ylabel('R0')
    plt.legend()

    plt.subplot(4, 1, 2)
    plt.plot(time_steps, quat_data[:, 1], label='R3')
    plt.ylabel('R3')
    plt.legend()

    plt.subplot(4, 1, 3)
    plt.plot(time_steps, quat_data[:, 2], label='R1')
    plt.ylabel('R1')
    plt.legend()

    plt.subplot(4, 1, 4)
    plt.plot(time_steps, quat_data[:, 3], label='R2')
    plt.ylabel('R2')
    plt.legend()

    plt.xlabel('Time Step')
    plt.tight_layout()
    plt.savefig('XPlane_Rot_plot.png')
    plt.close()



def plot_RotNQuat_data(quat_data, Rot):
    """ Plot quaternion data """
    quat_data = np.array(quat_data)      #Quaternion measured from XPlane directly
    Rot = np.array(Rot)                  #Rot = Rd Rdes
    time_steps = np.arange(len(quat_data))
    
    plt.figure()
    plt.subplot(4, 1, 1)
    plt.plot(time_steps, quat_data[:, 0], label='quat0')
    plt.plot(time_steps, Rot[:, 0], label='R0')
    plt.ylabel('R0')
    plt.legend()

    plt.subplot(4, 1, 2)
    plt.plot(time_steps, quat_data[:, 1], label='quat3')
    plt.plot(time_steps, Rot[:, 1], label='R3')
    plt.ylabel('R3')
    plt.legend()

    plt.subplot(4, 1, 3)
    plt.plot(time_steps, quat_data[:, 2], label='quat1')
    plt.plot(time_steps, Rot[:, 2], label='R1')
    plt.ylabel('R1')
    plt.legend()

    plt.subplot(4, 1, 4)
    plt.plot(time_steps, quat_data[:, 3], label='quat2')
    plt.plot(time_steps, Rot[:, 3], label='R2')
    plt.ylabel('R2')
    plt.legend()

    plt.xlabel('Time Step')
    plt.tight_layout()
    plt.savefig('XPlane_RotNQuat_plot.png')
    plt.close()

# ----------------------------------------------------------------

def plot_w_data(w_data):
    """ Plot w data """
    # Remember this come in the order e1e2, e2e3, e3e1 - > r,p,q
    w_data = np.array(w_data)
    time_steps = np.arange(len(w_data))
    
    plt.figure()
    plt.subplot(3, 1, 1)
    plt.plot(time_steps, w_data[:, 0], label='wz')
    plt.ylabel('wz')
    plt.legend()

    plt.subplot(3, 1, 2)
    plt.plot(time_steps, w_data[:, 1], label='wx')
    plt.ylabel('wx')
    plt.legend()

    plt.subplot(3, 1, 3)
    plt.plot(time_steps, w_data[:, 2], label='wy')
    plt.ylabel('wy')
    plt.legend()
    plt.xlabel('Time Step')
    plt.tight_layout()
    plt.savefig('XPlane_w_plot.png')
    plt.close()

def plot_angles(w_data, des_data):
    """ Plot w data """
    # Remember this come in the order e1e2, e2e3, e3e1 - > r,p,q
    w_data = np.array(w_data)
    des_data = np.array(des_data)
    time_steps = np.arange(len(w_data))
    
    plt.figure()
    plt.suptitle('Angular positions')
    plt.subplot(3, 1, 1)
    plt.plot(time_steps, w_data[:, 0], label='phi')
    plt.plot(time_steps, des_data[:, 0], label='phi d')
    plt.ylabel('phi')
    plt.legend()

    plt.subplot(3, 1, 2)
    plt.plot(time_steps, w_data[:, 1], label='theta')
    plt.plot(time_steps, des_data[:, 1], label='theta d')
    plt.ylabel('theta')
    plt.legend()

    plt.subplot(3, 1, 3)
    plt.plot(time_steps, w_data[:, 2], label='psi')
    plt.plot(time_steps, des_data[:, 2], label='psi d')
    plt.ylabel('psi')
    plt.legend()
    plt.xlabel('Time Step')
    plt.tight_layout()
    plt.savefig('XPlane_angles_plot.png')
    plt.close()

def plot_angleszoom(w_data, des_data):
    """ Plot angular positions from count = 5000 onward """
    w_data = np.array(w_data)
    des_data = np.array(des_data)
    time_steps = np.arange(len(w_data))
    
    # Ensure we only plot from count = 5000 onward
    start_idx = 5000 if len(time_steps) > 5000 else 0
    time_steps = time_steps[start_idx:]
    w_data = w_data[start_idx:, :]
    des_data = des_data[start_idx:, :]

    plt.figure()
    plt.suptitle('Angular positions')
    
    plt.subplot(3, 1, 1)
    plt.plot(time_steps, w_data[:, 0], label='phi')
    plt.plot(time_steps, des_data[:, 0], label='phi d')
    plt.ylabel('phi')
    plt.legend(loc='upper left')

    plt.subplot(3, 1, 2)
    plt.plot(time_steps, w_data[:, 1], label='theta')
    plt.plot(time_steps, des_data[:, 1], label='theta d')
    plt.ylabel('theta')
    plt.legend(loc='upper left')

    plt.subplot(3, 1, 3)
    plt.plot(time_steps, w_data[:, 2], label='psi')
    plt.plot(time_steps, des_data[:, 2], label='psi d')
    plt.ylabel('psi')
    plt.legend(loc='upper left')

    plt.xlabel('Time Step')
    plt.tight_layout()
    plt.savefig('XPlane_angles_plotZOOM.png')
    plt.close()    

# def plot_tau_data(tau_data):
#     """ Plot tau data """
#     tau_data = np.array(tau_data)
#     time_steps = np.arange(len(tau_data))
    
#     plt.figure()
#     plt.subplot(3, 1, 1)
#     plt.plot(time_steps, tau_data[:, 0], label='taue1e2')
#     plt.ylabel('taue1e2')
#     plt.legend()

#     plt.subplot(3, 1, 2)
#     plt.plot(time_steps, tau_data[:, 1], label='taue2e3')
#     plt.ylabel('taue2e3 - roll')
#     plt.legend()

#     plt.subplot(3, 1, 3)
#     plt.plot(time_steps, tau_data[:, 2], label='taue3e1')
#     plt.ylabel('taue3e1 - pitch')
#     plt.legend()

#     plt.xlabel('Time Step')
#     plt.tight_layout()
#     plt.savefig('XPlane_tau_plot.png')
#     plt.close()

def plot_sc_tau_data(tau_data):
    """ Plot tau data """
    tau_data = np.array(tau_data)
    time_steps = np.arange(len(tau_data))
    
    plt.figure()
    plt.suptitle('Control inputs sent')
    plt.subplot(3, 1, 1)
    plt.plot(time_steps, tau_data[:, 0], label='taue1e2')
    plt.ylabel('taue1e2')
    plt.legend()

    plt.subplot(3, 1, 2)
    plt.plot(time_steps, tau_data[:, 1], label='taue2e3')
    plt.ylabel('taue2e3 - roll')
    plt.legend()

    plt.subplot(3, 1, 3)
    plt.plot(time_steps, tau_data[:, 2], label='taue3e1')
    plt.ylabel('taue3e1 - pitch')
    plt.legend()

    plt.xlabel('Time Step')
    plt.tight_layout()
    plt.savefig('XPlane_scaled_tau_plot.png')
    plt.close()

def plot_error_data(error_data):
    """ Plot error data """
    error_data = np.array(error_data)
    time_steps = np.arange(len(error_data))
    
    plt.figure()
    plt.suptitle('Rotational Error')
    plt.subplot(3, 1, 1)
    plt.plot(time_steps, error_data[:, 0], label='errore1e2')  # in which order
    plt.axhline(0, color='black', linestyle='--', linewidth=1)
    plt.ylabel('errore1e2')
    plt.legend()

    plt.subplot(3, 1, 2)
    plt.plot(time_steps, error_data[:, 1], label='errore2e3')
    plt.axhline(0, color='black', linestyle='--', linewidth=1)
    plt.ylabel('errore2e3 - roll')
    plt.legend()

    plt.subplot(3, 1, 3)
    plt.plot(time_steps, error_data[:, 2], label='errore3e1')
    plt.axhline(0, color='black', linestyle='--', linewidth=1)
    plt.ylabel('errore3e1 - pitch')
    plt.legend()

    plt.xlabel('Time Step')
    plt.tight_layout()
    plt.savefig('XPlane_error_plot.png')
    plt.close()

def plot_xyz_data(xyz_data, refs, zoom_x=None, zoom_y=None):
    """ Plot xyz data with optional zooming """
    # Convert to numpy arrays
    xyz_data = np.array(xyz_data)
    refs = np.array(refs)
    time_steps = np.arange(len(xyz_data))
    
    plt.figure()

    # Subplot for x data
    plt.subplot(3, 1, 1)
    plt.plot(time_steps, xyz_data[:, 0], label='x')
    plt.plot(time_steps, refs[:, 0], label='x_d')
    plt.ylabel('x')
    plt.legend()
    if zoom_x:
        plt.xlim(zoom_x)  # Zoom into x-axis range if specified
    if zoom_y:
        plt.ylim(zoom_y)  # Zoom into y-axis range if specified

    # Subplot for y data
    plt.subplot(3, 1, 2)
    plt.plot(time_steps, xyz_data[:, 1], label='y')
    plt.plot(time_steps, refs[:, 1], label='y_d')
    plt.ylabel('y')
    plt.legend()
    if zoom_x:
        plt.xlim(zoom_x)
    if zoom_y:
        plt.ylim(zoom_y)

    # Subplot for z data
    plt.subplot(3, 1, 3)
    plt.plot(time_steps, xyz_data[:, 2], label='z')
    plt.plot(time_steps, refs[:, 2], label='z_d')
    plt.ylabel('z')
    plt.legend()
    if zoom_x:
        plt.xlim(zoom_x)
    if zoom_y:
        plt.ylim(zoom_y)

    plt.xlabel('Time Step')
    plt.tight_layout()

    # Save the figure
    plt.savefig('XPlane_xyz_plot.png')
    plt.close()

def plot_xyz_data_3d(xyz_data, refs, xlim=None, ylim=None, zlim=None):
    """ Plot xyz data in 3D with optional axis limits """
    # Convert the input data to numpy arrays
    xyz_data = np.array(xyz_data)
    refs = np.array(refs)
    
    # Initialize a 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot the x, y, and z data in 3D space
    ax.plot(xyz_data[:, 0], xyz_data[:, 1], -xyz_data[:, 2], label='Trajectory')
    ax.plot(refs[:, 0], refs[:, 1], -refs[:, 2], label='Reference')
    
    # Label the axes
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    # Apply axis limits if specified
    if xlim:
        ax.set_xlim(xlim)
    if ylim:
        ax.set_ylim(ylim)
    if zlim:
        ax.set_zlim(zlim)

    # Add a legend
    ax.legend()
    
    # Set the title of the plot
    ax.set_title('3D Trajectory of Quadcopter')
    
    # Save the plot as a PNG file
    plt.savefig('XPlane_xyz_plot_3D.png')
    plt.close()


def plot_xyz_errors(xyz_data, zoom_x=None, zoom_y=None):
    """ Plot xyz data with optional zooming """
    # Convert to numpy arrays
    xyz_data = np.array(xyz_data)
  
    time_steps = np.arange(len(xyz_data))
    
    plt.figure()

    # Subplot for x data
    plt.subplot(2, 1, 1)
    plt.plot(time_steps, xyz_data[:, 2], label='Error _z')
    
    plt.ylabel('Error z')
    plt.legend()
    if zoom_x:
        plt.xlim(zoom_x)  # Zoom into x-axis range if specified
    if zoom_y:
        plt.ylim(zoom_y)  # Zoom into y-axis range if specified

    # Subplot for y data
    plt.subplot(2, 1, 2)
    plt.plot(time_steps, xyz_data[:, 5], label='Error dz')
  
    plt.ylabel('Error dz')
    plt.legend()
    if zoom_x:
        plt.xlim(zoom_x)
    if zoom_y:
        plt.ylim(zoom_y)



    plt.xlabel('Time Step')
    plt.tight_layout()

    # Save the figure
    plt.savefig('XPlane_xyz_errors.png')
    plt.close()    

# def plot_thrust_data(thrust_data):
#     """ Plot thrust data """
#     # Remember this come in the order e1e2, e2e3, e3e1 - > r,p,q
#     thrust_data = np.array(thrust_data)
#     time_steps = np.arange(len(thrust_data))
    
#     plt.figure()
#     plt.subplot(1, 1, 1)
#     plt.plot(time_steps,thrust_data, label='thrust')
#     plt.ylabel('thrust')
#     plt.legend()

#     plt.xlabel('Time Step')
#     plt.tight_layout()
#     plt.savefig('XPlane_thrust_plot.png')
#     plt.close()

def plot_sc_thrust_data(thrust_data):
    """ Plot thrust data """
    # Remember this come in the order e1e2, e2e3, e3e1 - > r,p,q
    thrust_data = np.array(thrust_data)
    time_steps = np.arange(len(thrust_data))
    
    plt.figure()
    plt.subplot(1, 1, 1)
    plt.plot(time_steps,thrust_data, label='thrust')
    plt.ylabel('thrust')
    plt.legend()

    plt.xlabel('Time Step')
    plt.tight_layout()
    plt.savefig('XPlane_scaled_thrust_plot.png')
    plt.close()


# Test all Rd's
def plot_allrots_data(quat_data, Rot):
    """ Plot R's data """
    quat_data = np.array(quat_data)      # Quaternion measured from XPlane directly
    Rot = np.array(Rot)                  # Rot = Rd Rdes
    time_steps = np.arange(len(quat_data))
    
    plt.figure()
    
    plt.subplot(4, 1, 1)
    plt.plot(time_steps, quat_data[:, 0], label='quat0')
    plt.plot(time_steps, Rot[:, 0], label='Rd0')
    plt.ylabel('R0')
    plt.legend(loc='upper left')

    plt.subplot(4, 1, 2)
    plt.plot(time_steps, quat_data[:, 1], label='quat3')
    plt.plot(time_steps, Rot[:, 1], label='Rd3')
    plt.ylabel('R12')
    plt.legend(loc='upper left')

    plt.subplot(4, 1, 3)
    plt.plot(time_steps, quat_data[:, 2], label='quat1')
    plt.plot(time_steps, Rot[:, 2], label='Rd1')
    plt.ylabel('R23')
    plt.legend(loc='upper left')

    plt.subplot(4, 1, 4)
    plt.plot(time_steps, quat_data[:, 3], label='quat2')
    plt.plot(time_steps, Rot[:, 3], label='Rd2')
    plt.ylabel('R31')
    plt.legend(loc='upper left')

    plt.xlabel('Time Step')
    plt.tight_layout()
    plt.savefig('XPlane_allrots_plot.png')
    plt.close()

def plot_allrots_dataZOOM(quat_data, Rot):
    """ Plot R's data from count = 5000 onward """
    quat_data = np.array(quat_data)      # Quaternion measured from XPlane directly
    Rot = np.array(Rot)                  # Rot = Rd Rdes
    time_steps = np.arange(len(quat_data))

    # Ensure we only plot from count = 5000 onward
    start_idx = 5000 if len(time_steps) > 5000 else 0
    time_steps = time_steps[start_idx:]
    quat_data = quat_data[start_idx:, :]
    Rot = Rot[start_idx:, :]

    plt.figure()

    plt.subplot(4, 1, 1)
    plt.plot(time_steps, quat_data[:, 0], label='quat0')
    plt.plot(time_steps, Rot[:, 0], label='Rd0')
    plt.ylabel('R0')
    plt.legend(loc='upper left')

    plt.subplot(4, 1, 2)
    plt.plot(time_steps, quat_data[:, 1], label='quat3')
    plt.plot(time_steps, Rot[:, 1], label='Rd3')
    plt.ylabel('R12')
    plt.legend(loc='upper left')

    plt.subplot(4, 1, 3)
    plt.plot(time_steps, quat_data[:, 2], label='quat1')
    plt.plot(time_steps, Rot[:, 2], label='Rd1')
    plt.ylabel('R23')
    plt.legend(loc='upper left')

    plt.subplot(4, 1, 4)
    plt.plot(time_steps, quat_data[:, 3], label='quat2')
    plt.plot(time_steps, Rot[:, 3], label='Rd2')
    plt.ylabel('R31')
    plt.legend(loc='upper left')

    plt.xlabel('Time Step')
    plt.tight_layout()
    plt.savefig('XPlane_allrots_plotZOOM.png')
    plt.close()

# ===============================


    
if __name__ == '__main__':
    signal.signal(signal.SIGINT, signal_handler)
    rospy.init_node('listener', anonymous=True)
    motorSignals = rospy.Publisher('ControlSignals', Float64MultiArray, queue_size=10)
    QUADcon = QUADController()
    rospy.spin()
