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

import GAsimplePython as GA

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
        self.time_zero = rospy.get_time()

        self.Plotting = False
        self.saved_quat = []
        self.saved_w = []
        self.saved_tau = []
        self.saved_xyz = []
        self.saved_refs = []
        self.saved_Thrust = []
        self.saved_Rot = []
        self.saved_rot_error = []
        self.saved_Throttles = []
        self.saved_sc_Thrust = []
        self.saved_sc_Tau = []
        self.saved_angles = []

        self.count = 0

        self.xv = np.zeros([6])
        self.w = np.zeros([3])
        self.myQuat = np.array([1,0,0,0])
        self.data = [\
                [25, -998, -998, -998, -998, -998, -998, -998, -998],\
            ]
        self.timeos_before_first_measure = 0

        # Scalar basis
        # Is e0 what is now bold(1)?
        # self.e0 = np.array([[1, 0, 0, 0],
        #                [0, 1, 0, 0],
        #                [0, 0, 1, 0],
        #                [0, 0, 0, 1]])
        self.e0 = GA.EvenGrade(GA.SimpleQuaternion(1, 0, 0, 0))

        # Vector basis
        # self.e1 = np.array([[0, 0, 1, 0],
        #                [0, 0, 0, 1],
        #                [1, 0, 0, 0],
        #                [0, 1, 0, 0]])

        # self.e2 = np.array([[0, 0, 0, -1],
        #                [0, 0, 1, 0],
        #                [0, 1, 0, 0],
        #                [-1, 0, 0, 0]])

        # self.e3 = np.array([[1, 0, 0, 0],
        #                [0, 1, 0, 0],
        #                [0, 0, -1, 0],
        #                [0, 0, 0, -1]])
        
        self.e1 = GA.OddGrade(GA.SimpleQuaternion(0, 1, 0, 0))  # Vector e1 (odd grade)
        self.e2 = GA.OddGrade(GA.SimpleQuaternion(0, 0, 1, 0))  # Vector e2 (odd grade)
        self.e3 = GA.OddGrade(GA.SimpleQuaternion(0, 0, 0, 1))  # Vector e3 (odd grade)

        # Bivector basis
        # self.e1e2 = np.array([[0, 1, 0, 0],
        #                  [-1, 0, 0, 0],
        #                  [0, 0, 0, -1],
        #                  [0, 0, 1, 0]])

        # self.e2e3 = np.array([[0, 0, 0, 1],
        #                  [0, 0, -1, 0],
        #                  [0, 1, 0, 0],
        #                  [-1, 0, 0, 0]])

        # self.e3e1 = np.array([[0, 0, 1, 0],
        #                  [0, 0, 0, 1],
        #                  [-1, 0, 0, 0],
        #                  [0, -1, 0, 0]])

        self.e12 = GA.EvenGrade(GA.SimpleQuaternion(0, 0, 0, -1))  # Bivector e12
        self.e23 = GA.EvenGrade(GA.SimpleQuaternion(0, -1, 0, 0))  # Bivector e23
        self.e31 = GA.EvenGrade(GA.SimpleQuaternion(0, 0, -1, 0))  # Bivector e31

        # Trivector basis
        # self.e1e2e3 = np.array([[0, 1, 0, 0],
        #                    [-1, 0, 0, 0],
        #                    [0, 0, 0, 1],
        #                    [0, 0, -1, 0]])

        self.e123 = GA.OddGrade(GA.SimpleQuaternion(1, 0, 0, 0))

        self.dt = 0.01   # Debe coincidir con el Ros Rate =  100hz
        self.g = 9.81
        #self.m = 0.025
        self.m = 0.84  # Xplanes 1.86 lb = 0.84 kg                            
        # Is Inertia I=mr^2 ???
        # Radius of gyration
        # roll = pitch = 0.52 ft, yaw = 0.75
        #convertion factor = 0.06715
        self.Jx = self.m * 0.52**2 * 0.06715  #1
        self.Jy = self.m * 0.52**2 * 0.06715  #2
        self.Jz = self.m * 0.75**2 * 0.06715  #0

        #self.JJ = [self.Jx, self.Jy, self.Jz]
        self.JJ = np.diag([self.Jz, self.Jx, self.Jy])

        #LRQ Design
        # Design Position controller
        Ar = np.block([
            [np.zeros((3, 3)), np.eye(3)],
            [np.zeros((3, 6))]
        ])
        Br = np.block([
            [np.zeros((3, 3))],
            [np.eye(3)]
        ])

        Mr = np.block([
            [Ar, Br],
            [np.zeros((3, 9))]
        ])
        Mdr = expm(self.dt * Mr)
        Adr = Mdr[:6, :6]
        Bdr = Mdr[:6, 6:9]

        # Design Position controller
        Ap = np.block([
            [np.zeros((3, 3)), np.eye(3)],
            [np.zeros((3, 6))]
        ])
        Bp = np.block([
            [np.zeros((3, 3))],
            [np.eye(3)]
        ])

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
        ]) * 1e-4  # Weighting matrix for state

        Rp = np.diag([1,1,4])*1e-1  # Weighting matrix for input

        # Qr = np.block([    
        #     [np.diag([100000, 1, 1]) , np.zeros((3, 3))],
        #     [np.zeros((3, 3)), np.diag([0.1, 0.1, 0.1]) ]
        # ])* 1e-1  # Weighting matrix for state

        # Rr = np.diag([1,1,1]) *1e-3 # Weighting matrix for input
#______________________________________________________________________________-

        # Qp = np.block([
        #     [np.diag([1,1,1])*1e6, np.zeros((3, 3))],
        #     [np.zeros((3, 3)), np.diag([1, 1, 1])*1e-2]
        # ]) * 1e-3  # Weighting matrix for state

        # Rp = np.diag([1,1,1])*1e1  # Weighting matrix for input

        Qr = np.block([    
            [np.diag([0.0001,100,100])*21 , np.zeros((3, 3))],
            [np.zeros((3, 3)), np.diag([1,0.1,0.1])*1e1 ]
        ])  # Weighting matrix for state

        Rr = np.diag([1,0.3,0.3]) *1e1 # Weighting matrix for input

        self.Lr = np.array(self.discretelqr(Adr, Bdr, Qr, Rr))
        # self.Lr[np.abs(self.Lr)< 1e-8] = 0
        print(self.Lr)
        self.Lr[0, 0] = 0
        self.Lr[0, 3] = 0 

        # print(self.Lr)


        self.Lp = np.array(self.discretelqr(Adp, Bdp, Qp, Rp))
        
        self.Lp = np.array([[0.,     0.,         0.,         0.,    0.,         0.        ],
                    [0.,         0.,     0.,        0.,         0.,  0.        ],
                    [0.,         0.,         2,    0.,         0.,         1.38396554]])
        # self.Lp[np.abs(self.Lp)< 1e-8] = 0

        self.Bir = np.matmul(ln.inv(np.matmul(Br.T,Br)),Br.T)     

        rospy.Subscriber("/xplane/flightmodel/odom", Odometry, self.odomcallback)

        self.timer = rospy.Timer(rospy.Duration(0.01), self.sendToXPlane)

    def discretelqr(self,A, B, Q, R):
        X = np.matrix(scipy.linalg.solve_discrete_are(A, B, Q, R))
        #compute the LQR gain      print(Bir)
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
    def rotationalError(self, b1, b1d, b2, b2d):
        R1 = np.matmul(b1, b1d)  # Rotation equation
        B1 = self.kVectorPart(R1, 2)  # Wedge product
        if self.multivectorNorm(B1) < 1e-7:
            R10 = self.multiVectorParts(R1)[0]
            if np.sign(R10 + 0.5) == -1.0:  # Parallel Bivector
                if (self.multivectorNorm(b1 - self.e1) < 1e-7) or (self.multivectorNorm(-b1 - self.e1) < 1e-7):
                    a = self.kVectorPart(np.matmul(b1, self.e2), 2)
                else:
                    a = self.kVectorPart(np.matmul(b1, self.e1), 2)
                j1 = a / self.multivectorNorm(a)
                th1 = np.pi
            else:
                j1 = self.kVectorPart(np.zeros((4, 4)), 2)
                th1 = 0
        else:
            j1 = B1 / self.multivectorNorm(B1)
            y = self.multivectorNorm(B1)
            x = self.multiVectorParts(self.kVectorPart(R1, 0))[0]
            th1 = np.arctan2(y, x)

        ExpmJ1th1 = ln.expm(-j1 * th1 / 2)
        #ExpmJ1th1 = self.expBivector(-j1 * th1 / 2)
        b2p = np.matmul(np.matmul(ExpmJ1th1, b2),self.expBivector(j1 * th1 / 2))
        #b2p = np.matmul(np.matmul(ExpmJ1th1, b2),ln.expm(j1 * th1 / 2))
        #b2p = ExpmJ1th1 @ b2 @ ln.expm(j1 * th1 / 2)
        R2 = np.matmul(b2p, b2d)
        B2 = self.kVectorPart(R2, 2)
        if self.multivectorNorm(B2) < 1e-7:
            R20 = self.multiVectorParts(R2)[0]
            if np.sign(R20 + 0.5) == -1.0:  # Parallel Bivector
                if (self.multivectorNorm(b2 - self.e2) < 1e-7) or (self.multivectorNorm(-b2 - self.e2) < 1e-7):
                    a = self.kVectorPart(np.matmul(b2, self.e1), 2)
                else:
                    a = self.kVectorPart(np.matmul(b2, self.e2), 2)
                j2 = a / self.multivectorNorm(a)
                th2 = np.pi
            else:
                j2 = self.kVectorPart(np.zeros((4, 4)), 2)
                th2 = 0
        else:
            j2 = B2 / self.multivectorNorm(B2)
            y = self.multivectorNorm(B2)
            x = self.multiVectorParts(self.kVectorPart(R2, 0))[0]
            th2 = np.arctan2(y, x)

        #Re = np.matmul(self.expBivector(-j2 * th2 / 2), ExpmJ1th1)
        Re = np.matmul(ln.expm(-j2 * th2 / 2), ExpmJ1th1)
        Be = self.kVectorPart(Re, 2)

        if self.multivectorNorm(Be) < 1e-7:
            je = self.kVectorPart(np.zeros((4, 4)), 2)
        else:
            je = Be / self.multivectorNorm(Be)

        y = self.multivectorNorm(Be)
        x = self.multiVectorParts(self.kVectorPart(Re, 0))[0]
        the = np.arctan2(y, x)

        if self.multivectorNorm(2 * je * the) < 1e-7:
            jethe = self.kVectorPart(np.zeros((4, 4)), 2)
        else:
            jethe = self.kVectorPart(2 * je * the, 2)
        return jethe  # Rotational error  #Falta llamar th1 th2 the


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

        return jethe # Equation (27) in [1]


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

    def refsxyz(self, count):
        t=count#*0.001
        print(t)
        f=0.05

        if t<1700:
            x=t/200#-count*(0.01/3)#np.sin(2*np.pi*f*t)
        else:
            x=6.6
        #x=2#np.sin(2*np.pi*f*t+np.pi/6)
        dx=0#np.cos(2*np.pi*f*t+np.pi/6)*2*np.pi*f
        ddx=0#-np.sin(2*np.pi*f*t+np.pi/6)*(2*np.pi*f)**2
        dddx=0#-np.cos(2*np.pi*f*t+np.pi/6)*(2*np.pi*f)**3
        ddddx=0#np.sin(2*np.pi*f*t+np.pi/6)*(2*np.pi*f)**4

        y=0#count*(0.01/6)#np.cos(2*np.pi*f*t)
        dy=0#(0.01/6)#-np.sin(2*np.pi*f*t)*2*np.pi*f
        ddy=0#-np.cos(2*np.pi*f*t)*(2*np.pi*f)**2
        dddy=0#np.sin(2*np.pi*f*t)*(2*np.pi*f)**3
        ddddy=0#np.cos(2*np.pi*f*t)*(2*np.pi*f)**4

        if t<1700:
            z=-10#-count*(0.01/3)#np.sin(2*np.pi*f*t)
        elif t<3500:
            z=-20
        else:
            z = -15
        dz=0#-(0.01/3)#np.cos(2*np.pi*f*t)*2*np.pi*f
        ddz=0#-np.sin(2*np.pi*f*t)*(2*np.pi*f)**2
        dddz=0#-np.cos(2*np.pi*f*t)*(2*np.pi*f)**3
        ddddz=0#np.sin(2*np.pi*f*t)*(2*np.pi*f)**4
        return [x, y, z, dx, dy, dz, ddx, ddy, ddz, dddx, dddy, dddz, ddddx, ddddy, ddddz]


    def funRfin(self):
        #EZEiza
        q1 =  0.00177864870057
        q2 = -0.00143554212991
        q3 = 0.778165519238
        q0 = 0.628055036068

        Rfin = q0 * self.e0 + q1 * self.e1e2 + q2 * self.e2e3 + q3 * self.e3e1
        return Rfin/self.multivectorNorm(Rfin)

    def funRfin2(self):
        # t = i * dt
        # f = 0.05
        # A = 0.9

        # j1 = sin(2 * pi * f * t) * A
        # dj1 = cos(2 * pi * f * t) * A * (2 * pi * f)
        # ddj1 = -sin(2 * pi * f * t) * A * (2 * pi * f)^2

        j1 = 0#sin(2 * pi * f * t) * A
        dj1 = 0#cos(2 * pi * f * t) * A * (2 * pi * f)
        ddj1 = 0#-sin(2 * pi * f * t) * A * (2 * pi * f)^2

        Rfin = self.expBivector(-j1 * self.e1e2 / 2)
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

    
    def findOmegadOmega(self, thr,R,dotv,ddotv,dddotv):
        # thr scalar
        # R   bivector
        # dotv,ddotv,dddotv Vectors
        b3=np.matmul(np.matmul(R,self.e3),R.T)

        dotv=dotv[0]*self.e1+dotv[1]*self.e2+dotv[2]*self.e3
        ddotv=ddotv[0]*self.e1+ddotv[1]*self.e2+ddotv[2]*self.e3
        dddotv=dddotv[0]*self.e1+dddotv[1]*self.e2+dddotv[2]*self.e3

        nge3=self.multivectorNorm(self.g*self.e3-dotv)
        dthr=self.m*(2*ddotv - self.g*np.matmul(self.e3,ddotv) - np.matmul(ddotv,self.e3)*self.g)/(2*nge3)
        nbe3=self.multivectorNorm(1*self.e0+np.matmul(b3,self.e3)) 
        db3=-self.m*ddotv/thr - np.matmul(dthr,b3)/thr
        dR = np.matmul(db3,self.e3) / nbe3 + np.matmul(np.matmul((1*self.e0+np.matmul(b3,self.e3)),(1*self.e0+np.matmul(b3,self.e3))),db3) / nbe3**3
        Omega=-2*np.matmul(dR,R.T)
        ddthr= self.m/2 * ( (2*dddotv - self.g*np.matmul(self.e3,dddotv) - np.matmul(dddotv,self.e3)*self.g) / nge3 - np.matmul((2*ddotv - self.g*np.matmul(self.e3,ddotv) - np.matmul(ddotv,self.e3)*self.g),dthr) / (self.m*nge3**2) )    # Double check + or - sign here
        ddb3=-self.m*dddotv / thr - np.matmul(ddthr,b3) / thr - 2*np.matmul(dthr,db3) / thr
        dnbe3 = np.matmul((1*self.e0 + np.matmul(b3,self.e3)),db3) / nbe3
        ddR = np.matmul(ddb3,self.e3)/nbe3 + np.matmul(np.matmul(db3,self.e3),dnbe3) / nbe3**2 + np.matmul((db3 + np.matmul(db3,self.e3)),db3) / nbe3**3 + np.matmul((1*self.e0+self.e3+b3 + np.matmul(b3,self.e3)),ddb3) / nbe3**3 + np.matmul(np.matmul((1*self.e0+self.e3+b3 + np.matmul(b3,self.e3)),db3),3*dnbe3) / nbe3**4
        dOmega=-2*np.matmul(ddR,R.T) - 2*np.matmul(R,dR.T)
        
        return Omega,dOmega

    def findOmegadOmega2(self, thr, Rd, Rfin, dRfin, ddRfin, dotv, ddotv, dddotv):
        # thr scalar
        # R,Rfin,dRfin,ddRfin   bivectors
        # dotv,ddotv,dddotv Vectors

        b3 = np.matmul(np.matmul(Rd,self.e3),Rd.T)

        dotv = dotv[0] * self.e1 + dotv[1] * self.e2 + dotv[2] * self.e3
        ddotv = ddotv[0] * self.e1 + ddotv[1] * self.e2 + ddotv[2] * self.e3
        dddotv = dddotv[0] * self.e1 + dddotv[1] * self.e2 + dddotv[2] * self.e3

        nge3 = self.multivectorNorm(self.g * self.e3 - dotv)
        dthr = self.m * (2 * np.matmul(ddotv,dotv)  -  self.g * np.matmul(self.e3,ddotv)  -  np.matmul(ddotv,self.e3) * self.g) / (2 * nge3)
        nbe3 = self.multivectorNorm(1 * self.e0 + np.matmul(b3,self.e3))
        db3 = -self.m * ddotv / thr  -  np.matmul(dthr,b3) / thr
        dRd = (np.matmul(db3,self.e3) / nbe3  -  (np.matmul((1 * self.e0 + np.matmul(b3,self.e3)),db3) )/ nbe3**3)

        ddthr = self.m / 2 * ((2 * np.matmul(dddotv,dddotv) + 2 * dddotv**2  -  self.g * np.matmul(self.e3,dddotv)  -  np.matmul(dddotv,self.e3) * self.g) / nge3  +  (2 * np.matmul(dotv,ddotv)  -  self.g * np.matmul(self.e3,ddotv)  -  np.matmul(ddotv,self.e3) * self.g)**2 / (2 * nge3**3))
        ddb3 = -self.m * dddotv / thr  -  np.matmul(ddthr,b3) / thr - 2 * np.matmul(dthr,db3) / thr

        ddRd = np.matmul(ddb3,self.e3) / nbe3  -  db3**2 / nbe3**3 - (db3**2 + np.matmul((1 * self.e0 + np.matmul(db3,self.e3)),ddb3)) / nbe3**2  +  (np.matmul((1 * self.e0 + np.matmul(b3,self.e3)),db3**2))/ nbe3**7

        # Update R with Rfin and its derivatives
        tRd = np.matmul(Rd,Rfin.T)
        dtRd = np.matmul(dRd,Rfin.T) + np.matmul(Rd,dRfin.T)
        ddtRd = np.matmul(ddRd,Rfin.T) + 2 * np.matmul(dRd,dRfin.T) + np.matmul(Rd,ddRfin.T)

        Omega = -2 * np.matmul(dtRd,tRd.T)
        dOmega = -2 * np.matmul(ddtRd,tRd.T) - 2 * np.matmul(dtRd,dtRd.T)

        return Omega, dOmega, dRd, ddRd



    def control(self, xv, w, Rq, m, g, JJ, Bir, Lp, Lr, count):
        # Compute the Rotation reference
        refs = np.array(self.refsxyz(count))
        posErr = xv - refs[0:6]

        #test PD
        error_z = xv[2] - refs[2]
        e_error_z = xv[5] - refs[5]
        kpdz=0.02*8*17
        kddz=0.06*4*22
        #pd_z = 1.5*error_z + 3.3*e_error_z
        pd_z = kpdz*error_z + kddz*e_error_z

        #When turning off the x and y position it does track the altitude
        kpdx=0.000000
        kpdy=0.000
        kddx=0.000000
        kddy=0.00

        Lp = np.array([[kpdx, 0, 0, kddx, 0 ,0],[0, kpdy, 0, 0, kddy, 0],[0, 0, kpdz, 0, 0, kddz]])

        dv = np.matmul(-Lp, np.reshape((posErr), (6, 1)))-np.vstack([0,0,g])+np.reshape(np.array(refs[6:9]), (3, 1))  #this refs output is not being what it should be
      
        dv = dv[0] * self.e1 + dv[1] * self.e2 + dv[2] * self.e3
        print(dv)
        # Compute Trust
        Trust = m * self.multivectorNorm(-dv)
      
        # Compute actual b1 and b2
        Rq_mat = Rq[0] * self.e0 + Rq[1] * self.e1e2 + Rq[2] * self.e2e3 + Rq[3] * self.e3e1
        Rq_mat = Rq_mat / self.multivectorNorm(Rq_mat)

        actualAngles = self.myang(Rq_mat)
        self.saved_angles.append(actualAngles)

        #Rq_mat = 1*self.e0 + 0*self.e1e2 + 0*self.e2e3 + 0*self.e3e1

        b1 = self.kVectorPart(np.matmul(np.matmul(Rq_mat, self.e1),Rq_mat.T), 1)
        b2 = self.kVectorPart(np.matmul(np.matmul(Rq_mat, self.e2),Rq_mat.T), 1)

        b1vector = self.multiVectorParts(b1)[1:4]
        b2vector = self.multiVectorParts(b2)[1:4]

        # Compute the rotation
        b3rot = m*(-dv) / Trust
        #print('b3rot')
        #print(b3rot)

        if self.multivectorNorm(b3rot) > 1e-9:
            b3rot = b3rot / self.multivectorNorm(b3rot)
            Rd = self.estR(b3rot, self.e3)
        else:
            Rd = 1*self.e0  # Identity matrix
        #print('Rot')
        #print(Rot)

        # Disconnect Position Controller
        #Rd = 1*self.e0 + 0*self.e1e2 + 0*self.e2e3 + 0*self.e3e1
        
        Rdes, dRdes, ddRdes = self.funRfin2()
        #print(Rdes)
        Rdes = self.kVectorPart(Rdes, 0) + self.kVectorPart(Rdes, 2)
        Rdes = Rdes / self.multivectorNorm(Rdes)
        Rot = np.matmul(Rd,Rdes)
        Rot = self.kVectorPart(Rot, 0) + self.kVectorPart(Rot, 2)
        Rot = Rot / self.multivectorNorm(Rot)      
        
        # RdesVector1 = self.multiVectorParts(Rot)[0]
        # RdesVector2 = np.array(self.multiVectorParts(Rot)[4:7])
        # RdesVector = np.concatenate(([RdesVector1], RdesVector2))
        # self.saved_Rot.append(RdesVector)
        R0f = self.multiVectorParts(self.kVectorPart(Rot, 0))[0]
        R1f = self.multiVectorParts(self.kVectorPart(Rot, 2))[4]
        R2f = self.multiVectorParts(self.kVectorPart(Rot, 2))[5]
        R3f = self.multiVectorParts(self.kVectorPart(Rot, 2))[6]
        myRot = np.array([R0f,R1f,R2f,R3f])
        # print(myRot)
        self.saved_Rot.append(myRot)

        b1d = np.matmul(np.matmul(Rot, self.e1),Rot.T)
        b1d = self.kVectorPart(b1d, 1)
        b1dvector = self.multiVectorParts(b1d)[1:4]

        b2d = np.matmul(np.matmul(Rot, self.e2),Rot.T)
        b2d = self.kVectorPart(b2d, 1)
        b2dvector = self.multiVectorParts(b2d)[1:4]

        b3d = np.matmul(np.matmul(Rot, self.e3),Rot.T)
        b3d = self.kVectorPart(b3d, 1)
        b3dvector = self.multiVectorParts(b3d)[1:4]

        #Er = self.rotationalError(b1, b1d, b2, b2d)
        Er = self.rotationalError2(Rq_mat,Rot)

        Er_parts = self.multiVectorParts(Er)
        Er4, Er5, Er6 = Er_parts[4:7]   #e1e2 e2e3 e3e1

        Errors = np.vstack([Er4,Er5,Er6])[:,0]
        self.saved_rot_error.append(Errors)

        # TODO Before inputting w we have to transform it into the intertial frame
        # Is that Rq_mat * w *Rq_mat.T    or   Rot * w *Rot.T   ????  its Rq_mat
        #     p  -> e2e3           q ->  e3e1       r -> e1e2
        # w = w[0]*self.e2e3 + w[1]*self.e3e1 + w[2]*self.e1e2
        w = w[0]*self.e1e2 + w[1]*self.e2e3 + w[2]*self.e3e1    # No hay e0?
        w = self.kVectorPart(w,2) 
        w4, w5, w6 = self.multiVectorParts(w)[4:7]

        # -- I BELIEVE THE ANGULAR VELOCTITIES ARE IN THE BODY FRAME

        # winertial = np.matmul(np.matmul(Rq_mat.T, w),Rq_mat)
        # winertial = self.kVectorPart(winertial,2) 
        # w4, w5, w6 = self.multiVectorParts(winertial)[4:7]
        wsaves = np.array([w4,w5,w6])
        self.saved_w.append(wsaves)

        #Desired omegas
        #omega,dOmega = self.findOmegadOmega(Trust,Rot,refs[6:9],refs[9:12],refs[12:15])
        omega, dOmega, dRV, ddRV = self.findOmegadOmega2(Trust, Rd, Rdes, dRdes, ddRdes, refs[6:9], refs[9:12], refs[12:15])
        
        Omega4,Omega5,Omega6=self.multiVectorParts(omega)[4:7]
        dOmega4,dOmega5,dOmega6=self.multiVectorParts(dOmega)[4:7]
        #transform omegas to? wait they dont depend on w
  
        Lr = np.array([[0, 0, 0, 0, 0 ,0],[0, 25.5, 0, 0, 7.16, 0],[0, 0, 59.5, 0, 0, 17.16]])

        #tracking
        # x = np.reshape(np.concatenate(([Er4, Er5, Er6], [w4-Omega4, w5-Omega5, w6-Omega6])), (6, 1))
        # v = np.matmul(-Lr, x) + np.vstack([dOmega4,dOmega5,dOmega6])

        #Regulation
        x = np.reshape(np.concatenate(([Er4, Er5, Er6], [w4, w5, w6])), (6, 1))
        v = np.matmul(-Lr, x)
        
        elem1 = self.myf(x, JJ)
        #print(elem1)
        elem = np.matmul(Bir, elem1)
  
        #print(elem)
        tau = np.matmul(JJ,v) - elem  #The first matmul should NOT give 1x1 output
        #print(np.shape(v)) 
 
        tauvec = tau[0] * self.e1e2 + tau[1] * self.e2e3 + tau[2] * self.e3e1

        #tauB_parts = np.matmul(np.matmul(Rq_mat, tauvec),Rq_mat.T)
        tauB_parts = self.kVectorPart(tauvec,2)
        tauB_parts = self.multiVectorParts(tauB_parts)[4:7]
        
        tauB = np.array(tauB_parts)

        self.saved_refs.append(refs[0:3])
        
        #return tauB, pd_z#Trust
        return tauB, Trust

    def odomcallback(self, data):
        self.quaternion = data.pose.pose.orientation  #Supposed to be NED
        self.q0 = self.quaternion.w
        self.q1 = self.quaternion.x
        self.q2 = self.quaternion.y
        self.q3 = self.quaternion.z
        self.myQuat = np.array([self.q0,self.q3,self.q1,self.q2])
        #self.myQuat = np.array([self.q0,self.q1,self.q2,self.q3])

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

        #     p  -> e2e3           q ->  e3e1       r -> e1e2
        self.w = np.array([self.r, self.p, self.q])   #This is resulting in the proper signals both in the expected disorder
        #self.w = np.array([self.p, self.q, self.r])

        self.tausB, Thrust = self.control(self.xv, self.w, self.myQuat,  self.m, self.g, self.JJ, self.Bir, self.Lp, self.Lr,self.count)

        #This tausB should also come in the order e1e2,e2e3,e3e1
        self.saved_tau.append(self.tausB)

        Thrust = Thrust
        if self.count < 300:
            Thrust = Thrust# + 15
        else:
            Thrust = Thrust
           
        self.saved_Thrust.append(Thrust)
        
        #Normalize and then /4
        weight = self.m*self.g # = 8.24 N  Thats the max thrust output
        hoverThrottle = 0.0864  # Throttle per motor needed to maintain hover
        normFactor = hoverThrottle/weight

        # print 'tausb: %s' %self.tausB
        # print 'Thrust: %s' %Thrust
     
        #Mixer
        # self.Throttle1 = Thrust + self.tausB[0] - self.tausB[1] - self.tausB[2]
        # self.Throttle2 = Thrust + self.tausB[0] + self.tausB[1] + self.tausB[2]
        # self.Throttle3 = Thrust - self.tausB[0] + self.tausB[1] - self.tausB[2]
        # self.Throttle4 = Thrust - self.tausB[0] - self.tausB[1] + self.tausB[2]

        #ThrustNorm = Thrust*normFactor
        # print 'ThrustNorm: %s' %ThrustNorm

        scaledThrust = Thrust*normFactor
        scaledTau = self.tausB*normFactor

        # print 'scaled tausb: %s' %scaledTau
        # print 'scaled Thrust: %s' %scaledThrust

        #print 'Control Signals: %s %s' % (scaledThrust, scaledTau)

        self.Throttle1 = scaledThrust + scaledTau[0] - scaledTau[1] + scaledTau[2]
        self.Throttle2 = scaledThrust - scaledTau[0] - scaledTau[1] - scaledTau[2]
        self.Throttle3 = scaledThrust + scaledTau[0] + scaledTau[1] - scaledTau[2]
        self.Throttle4 = scaledThrust - scaledTau[0] + scaledTau[1] + scaledTau[2]

        # Mixer according to Ignacios convention
        # self.Throttle1 = scaledThrust - scaledTau[0] + scaledTau[1] + scaledTau[2]
        # self.Throttle2 = scaledThrust - scaledTau[0] - scaledTau[1] - scaledTau[2]
        # self.Throttle3 = scaledThrust + scaledTau[0] - scaledTau[1] + scaledTau[2]
        # self.Throttle4 = scaledThrust + scaledTau[0] + scaledTau[1] - scaledTau[2]

        # self.Throttle1 = 0.5#scaledThrust - scaledTau[0] + scaledTau[1] + scaledTau[2]
        # self.Throttle2 = 0#scaledThrust - scaledTau[0] - scaledTau[1] - scaledTau[2]
        # self.Throttle3 = 0#scaledThrust + scaledTau[0] - scaledTau[1] + scaledTau[2]
        # self.Throttle4 = 0.5#scaledThrust + scaledTau[0] + scaledTau[1] - scaledTau[2]
        
        self.saved_sc_Tau.append(scaledTau)
        self.saved_sc_Thrust.append(scaledThrust)
        
        # Assuming tau comes in the order e1e2 e2e3 e3e1
        # self.Throttle1 = (Thrust + self.tausB[0] - self.tausB[1] + self.tausB[2])*normFactor
        # self.Throttle2 = (Thrust - self.tausB[0] - self.tausB[1] - self.tausB[2])*normFactor
        # self.Throttle3 = (Thrust + self.tausB[0] + self.tausB[1] - self.tausB[2])*normFactor
        # self.Throttle4 = (Thrust - self.tausB[0] + self.tausB[1] + self.tausB[2])*normFactor

        Throttles = [self.Throttle1, self.Throttle2, self.Throttle3, self.Throttle4]
        self.saved_Throttles.append(Throttles)

        self.data = [\
                [25,self.Throttle1, self.Throttle2, self.Throttle3, self.Throttle4, -998, -998, -998, -998],\
           ]
        self.count += 1
        

    def sendToXPlane(self, event):
        motors = self.data
        #print 'Motors: %s' %motors[0][1:5]
        #print('pre time',rospy.get_time())
        self.client.sendDATA(motors)   # try to publish it in another node to verify the hz and echo

        motor_data = [float(x) for x in motors[0]]

        #myMsg = Float64MultiArray()
        #myMsg.data = motor_data

        #motorSignals.publish(myMsg)

def signal_handler(sig, frame):
    """ Handle keyboard interrupt """
    print("\nInterrupt received, plotting quaternion data...")
    if QUADcon:
        plot_quaternion_data(QUADcon.saved_quat)
        plot_w_data(QUADcon.saved_w)
        plot_tau_data(QUADcon.saved_tau)
        plot_sc_tau_data(QUADcon.saved_sc_Tau)
        plot_xyz_data(QUADcon.saved_xyz, QUADcon.saved_refs)
        plot_thrust_data(QUADcon.saved_Thrust)
        plot_sc_thrust_data(QUADcon.saved_sc_Thrust)
        plot_xyz_data_3d(QUADcon.saved_xyz, QUADcon.saved_refs)
        plot_Rot_data(QUADcon.saved_Rot)
        plot_error_data(QUADcon.saved_rot_error)
        plot_RotNQuat_data(QUADcon.saved_quat, QUADcon.saved_Rot)
        plot_motor_data(QUADcon.saved_Throttles)
        plot_angles(QUADcon.saved_angles)
    sys.exit(0)

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
    quat_data = np.array(quat_data)
    Rot = np.array(Rot)
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

def plot_angles(w_data):
    """ Plot w data """
    # Remember this come in the order e1e2, e2e3, e3e1 - > r,p,q
    w_data = np.array(w_data)
    time_steps = np.arange(len(w_data))
    
    plt.figure()
    plt.subplot(3, 1, 1)
    plt.plot(time_steps, w_data[:, 0], label='phi')
    plt.ylabel('phi')
    plt.legend()

    plt.subplot(3, 1, 2)
    plt.plot(time_steps, w_data[:, 1], label='theta')
    plt.ylabel('theta')
    plt.legend()

    plt.subplot(3, 1, 3)
    plt.plot(time_steps, w_data[:, 2], label='psi')
    plt.ylabel('psi')
    plt.legend()
    plt.xlabel('Time Step')
    plt.tight_layout()
    plt.savefig('XPlane_angles_plot.png')
    plt.close()

def plot_tau_data(tau_data):
    """ Plot tau data """
    tau_data = np.array(tau_data)
    time_steps = np.arange(len(tau_data))
    
    plt.figure()
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
    plt.savefig('XPlane_tau_plot.png')
    plt.close()

def plot_sc_tau_data(tau_data):
    """ Plot tau data """
    tau_data = np.array(tau_data)
    time_steps = np.arange(len(tau_data))
    
    plt.figure()
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
    plt.savefig('XPlane_sc_tau_plot.png')
    plt.close()

def plot_error_data(error_data):
    """ Plot error data """
    error_data = np.array(error_data)
    time_steps = np.arange(len(error_data))
    
    plt.figure()
    plt.subplot(3, 1, 1)
    plt.plot(time_steps, error_data[:, 0], label='errore1e2')  # in which order
    plt.ylabel('errore1e2')
    plt.legend()

    plt.subplot(3, 1, 2)
    plt.plot(time_steps, error_data[:, 1], label='errore2e3')
    plt.ylabel('errore2e3 - roll')
    plt.legend()

    plt.subplot(3, 1, 3)
    plt.plot(time_steps, error_data[:, 2], label='errore3e1')
    plt.ylabel('errore3e1 - pitch')
    plt.legend()

    plt.xlabel('Time Step')
    plt.tight_layout()
    plt.savefig('XPlane_error_plot.png')
    plt.close()

def plot_xyz_data(xyz_data,refs):
    """ Plot xyz data """
    # Remember this come in the order e1e2, e2e3, e3e1 - > r,p,q
    xyz_data = np.array(xyz_data)
    refs = np.array(refs)
    time_steps = np.arange(len(xyz_data))
    
    plt.figure()
    plt.subplot(3, 1, 1)
    plt.plot(time_steps,xyz_data[:, 0], label='x')
    plt.plot(time_steps,refs[:, 0], label='x_d')
    plt.ylabel('x')
    plt.legend()

    plt.subplot(3, 1, 2)
    plt.plot(time_steps, xyz_data[:, 1], label='y')
    plt.plot(time_steps,refs[:, 1], label='y_d')
    plt.ylabel('y')
    plt.legend()

    plt.subplot(3, 1, 3)
    plt.plot(time_steps, xyz_data[:, 2], label='z')
    plt.plot(time_steps,refs[:, 2], label='z_d')
    plt.ylabel('z')
    plt.legend()

    plt.xlabel('Time Step')
    plt.tight_layout()
    plt.savefig('XPlane_xyz_plot.png')
    plt.close()

def plot_xyz_data_3d(xyz_data, refs):
    """ Plot xyz data in 3D """
    # Convert the input data to a numpy array
    xyz_data = np.array(xyz_data)
    refs = np.array(refs)
    
    # Create a time array assuming uniform time steps
    time_steps = np.arange(len(xyz_data))
    
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
    
    # Add a legend
    ax.legend()
    
    # Set the title of the plot
    ax.set_title('3D Trajectory of Quadcopter')
    
    # Save the plot as a PNG file
    plt.savefig('XPlane_xyz_plot_3D.png')
    plt.close()

def plot_thrust_data(thrust_data):
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
    plt.savefig('XPlane_thrust_plot.png')
    plt.close()

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
    plt.savefig('XPlane_sc_thrust_plot.png')
    plt.close()
    
if __name__ == '__main__':
    signal.signal(signal.SIGINT, signal_handler)
    rospy.init_node('listener', anonymous=True)
    motorSignals = rospy.Publisher('ControlSignals', Float64MultiArray, queue_size=10)
    QUADcon = QUADController()
    rospy.spin()
