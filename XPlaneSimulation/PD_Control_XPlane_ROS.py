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

        self.count = 0

        self.xv = np.zeros([6])
        self.w = np.zeros([3])
        self.myQuat = np.array([1,0,0,0])
        self.data = [\
                [25, -998, -998, -998, -998, -998, -998, -998, -998],\
            ]
        self.timeos_before_first_measure = 0
        self.x_des = 3
        self.dx_des = 0
        self.y_des = 5
        self.dy_des = 0
        self.z_des = -10
        self.dz_des = 0

        self.refxyz = np.array([self.x_des, self.y_des, self.z_des, self.dx_des, self.dy_des, self.dz_des])

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

        self.dt = 0.01   # Debe coincidir con el Ros Rate =  100hz

        self.g = 9.81
        #self.m = 0.025
        self.m = 0.84  # Xplanes 1.86 lb = 0.84 kg                            
        # self.Jx = 4.856e-3  # Is Inertia I=mr^2 ???
        # self.Jy = 4.856e-3
        # self.Jz = 8.801e-3
        # Radius of gyration
        # roll = pitch = 0.52 ft, yaw = 0.75
        #convertion factor = 0.06715
        self.Jx = self.m * 0.52**2 * 0.06715
        self.Jy = self.m * 0.52**2 * 0.06715
        self.Jz = self.m * 0.75**2 * 0.06715

        self.JJ = [self.Jx, self.Jy, self.Jz]

        #LRQ Design
        # Design Position controller
        Ar = np.block([
            [np.zeros((3, 3)), -2.0 * np.eye(3)],
            [np.zeros((3, 6))]
        ])
        Br = np.block([
            [np.zeros((3, 3))],
            [np.diag([1 / self.Jz, 1 / self.Jx, 1 / self.Jy])]
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

        Qp = np.block([
            [np.diag([4500, 4500, 2500]), np.zeros((3, 3))],
            [np.zeros((3, 3)), np.diag([1, 1, 1400])]
        ]) * 1e-4  # Weighting matrix for state

        Rp = np.diag([1,1,4])*1e-1  # Weighting matrix for input

        Qr = np.block([    
            [np.diag([1, 1, 1]) , np.zeros((3, 3))],
            [np.zeros((3, 3)), np.diag([0.1, 0.1, 0.1]) ]
        ])* 1e-2  # Weighting matrix for state

        Rr = np.diag([1,1,1]) *1e-1 # Weighting matrix for input

        self.Lr = np.array(self.discretelqr(Adr, Bdr, Qr, Rr))
        self.Lr[np.abs(self.Lr)< 1e-8] = 0

        self.Lp = np.array(self.discretelqr(Adp, Bdp, Qp, Rp))
        self.Lp[np.abs(self.Lp)< 1e-8] = 0

        self.Bir = np.matmul(ln.inv(np.matmul(Br.T,Br)),Br.T)     

        rospy.Subscriber("/xplane/flightmodel/odom", Odometry, self.odomcallback)

        self.timer = rospy.Timer(rospy.Duration(0.01), self.sendToXPlane)

    def discretelqr(self,A, B, Q, R):
        X = np.matrix(scipy.linalg.solve_discrete_are(A, B, Q, R))
        
        #compute the LQR gain
        K = np.matrix(scipy.linalg.inv(B.T*X*B+R)*(B.T*X*A))
        
        eigVals, eigVecs = scipy.linalg.eig(A-B*K)
        return K

    # Decomposes the multivector M into its scalar, vector, bivector, and trivector parts
    def multiVectorParts(self, M):
        m1 = np.trace((np.matmul(M,self.e1) + np.matmul(self.e1, M)) / 2) / 4
        m2 = np.trace((np.matmul(M, self.e2) + np.matmul(self.e2, M)) / 2) / 4
        m3 = np.trace((np.matmul(M, self.e3) + np.matmul(self.e3, M)) / 2) / 4
        m4 = -np.trace((np.matmul(M, self.e1e2) + np.matmul(self.e1e2, M)) / 2) / 4
        m5 = -np.trace((np.matmul(M, self.e2e3) + np.matmul(self.e2e3, M)) / 2) / 4
        m6 = -np.trace((np.matmul(M, self.e3e1) + np.matmul(self.e3e1, M)) / 2) / 4
        m7 = -np.trace((np.matmul(M, self.e1e2e3) + np.matmul(self.e1e2e3, M)) / 2) / 4

        # m1 = np.trace((M @ self.e1 + self.e1 @ M) / 2) / 4
        # m2 = np.trace((M @ self.e2 + self.e2 @ M) / 2) / 4
        # m3 = np.trace((M @ self.e3 + self.e3 @ M) / 2) / 4
        # m4 = -np.trace((M @ self.e1e2 + self.e1e2 @ M) / 2) / 4
        # m5 = -np.trace((M @ self.e2e3 + self.e2e3 @ M) / 2) / 4
        # m6 = -np.trace((M @ self.e3e1 + self.e3e1 @ M) / 2) / 4
        # m7 = -np.trace((M @ self.e1e2e3 + self.e1e2e3 @ M) / 2) / 4
        m0 = np.trace(M - m1 * self.e1 - m2 * self.e2 - m3 * self.e3 - m4 * self.e1e2 - m5 * self.e2e3 - m6 * self.e3e1 - m7 * self.e1e2e3) / 4
        return m0, m1, m2, m3, m4, m5, m6, m7

    # Norm of a multivector M
    def multivectorNorm(self, M):
        m0, m1, m2, m3, m4, m5, m6, m7 = self.multiVectorParts(M)
        return np.sqrt(m0 ** 2 + m1 ** 2 + m2 ** 2 + m3 ** 2 + m4 ** 2 + m5 ** 2 + m6 ** 2 + m7 ** 2)

    # Extracts the k-vector part of the multivector M
    def kVectorPart(self, M, k):
        m0, m1, m2, m3, m4, m5, m6, m7 = self.multiVectorParts(M)
        if k == 0:
            return m0 * self.e0
        elif k == 1:
            return m1 * self.e1 + m2 * self.e2 + m3 * self.e3
        elif k == 2:
            return m4 * self.e1e2 + m5 * self.e2e3 + m6 * self.e3e1
        elif k == 3:
            return m7 * self.e1e2e3

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
        b2p = np.matmul(np.matmul(ExpmJ1th1, b2),ln.expm(j1 * th1 / 2))
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

        return jethe  # Rotational error

    # Vector field for rotational dynamics
    def myf(self, x, JJ):
        f = np.zeros(6)
        f[3] = (x[4] * x[5] * (JJ[1] - JJ[2]) / JJ[0])[0]
        f[4] = (x[5] * x[3] * (JJ[2] - JJ[0]) / JJ[1])[0]
        f[5] = (x[3] * x[4] * (JJ[0] - JJ[1]) / JJ[2])[0]
        return np.reshape(f, (6, 1))

    # Estimates the rotation matrix R from the vector field f3
    def estR(self, f3, e3):
        Rb = self.e0 + np.matmul(f3, e3)
        Rb = self.kVectorPart(Rb, 0) + self.kVectorPart(Rb, 2)
        Rb = Rb / self.multivectorNorm(Rb)
        return Rb

    def funRfin(self):
        # q1 = 0.00185055157635#0
        # q2 = 0.00134157820139#0
        # q3 = 0.809623241425
        # q0 = -0.586945474148#np.sqrt(1 - (q1 ** 2 + q2 ** 2 + q3 ** 3))

        #EZEiza
        q1 =  0.00177864870057
        q2 = -0.00143554212991
        q3 = 0.778165519238
        q0 = 0.628055036068



        Rfin = q0 * self.e0 + q3 * self.e1e2 + q1 * self.e2e3 + q2 * self.e3e1

        return Rfin/self.multivectorNorm(Rfin)

    def myang(self, R):
        R=R/self.multivectorNorm(R)
        q0=self.multiVectorParts(R)[0]
        q1=self.multiVectorParts(R)[4]
        q2=self.multiVectorParts(R)[5]
        q3=self.multiVectorParts(R)[6]
        

        phi=math.atan2(2*(q0*q1+q2*q3),1-2*(q1**2+q2**2))
        if (abs(2*(q0*q2-q1*q3))>=1):
            theta=0
        else:
            theta=math.asin(2*(q0*q2-q1*q3))
        
        psi=math.atan2(2*(q0*q3+q1*q2),1-2*(q2**2+q3**2))

        return phi,theta,psi    

    def control(self, xv, w, Rq, refxyz, m, g, JJ, Bir, Lp, Lr):
        # Compute the Rotation reference
        posErr = xv - refxyz
        dv = np.matmul(-Lp, np.reshape((posErr), (6, 1)))
        dv = dv[0] * self.e1 + dv[1] * self.e2 + dv[2] * self.e3
        # print 'dv: %s' %dv
        # Compute Trust
        Trust = m * self.multivectorNorm(-dv + g * self.e3)

        # print 'Thrust: %s' %Trust
        
        # Compute actual b1 and b2
        Rq_mat = Rq[0] * self.e0 + Rq[1] * self.e1e2 + Rq[2] * self.e2e3 + Rq[3] * self.e3e1
        Rq_mat = Rq_mat / self.multivectorNorm(Rq_mat)

        #Rq_mat = 1*self.e0 + 0*self.e1e2 + 0*self.e2e3 + 0*self.e3e1

        b1 = self.kVectorPart(np.matmul(np.matmul(Rq_mat, self.e1),Rq_mat.T), 1)
        b2 = self.kVectorPart(np.matmul(np.matmul(Rq_mat, self.e2),Rq_mat.T), 1)

        b1vector = self.multiVectorParts(b1)[1:4]
        # print('b1vector')
        # print(b1vector)
        b2vector = self.multiVectorParts(b2)[1:4]
        # print('b2vector')
        # print(b2vector)

        # b1 = self.kVectorPart(Rq_mat @ self.e1 @ Rq_mat.T, 1)
        # b2 = self.kVectorPart(Rq_mat @ self.e2 @ Rq_mat.T, 1)

        # Compute the rotation
        b3rot = -(m * dv - m * g * self.e3) / Trust

        if self.multivectorNorm(b3rot) > 1e-9:
            b3rot = b3rot / self.multivectorNorm(b3rot)
            Rot = self.estR(b3rot, self.e3)
        else:
            Rot = self.e0  # Identity matrix


        Rdes = self.funRfin()
        Rdes = self.kVectorPart(Rdes, 0) + self.kVectorPart(Rdes, 2)
        Rdes = Rdes / self.multivectorNorm(Rdes)
        Rot = np.matmul(Rot,Rdes)
        Rot = self.kVectorPart(Rot, 0) + self.kVectorPart(Rot, 2)
        Rot = Rot / self.multivectorNorm(Rot)



        #
        RdesVector1 = self.multiVectorParts(Rot)[0]
        RdesVector2 = np.array(self.multiVectorParts(Rot)[4:7])
        RdesVector = np.concatenate(([RdesVector1], RdesVector2))
        self.saved_Rot.append(RdesVector)
        
  

        b1d = np.matmul(np.matmul(Rot, self.e1),Rot.T)
        #b1d = Rot @ self.e1 @ Rot.T
        b1d = self.kVectorPart(b1d, 1)
        b1dvector = self.multiVectorParts(b1d)[1:4]

        b2d = np.matmul(np.matmul(Rot, self.e2),Rot.T)
        #b2d = Rot @ self.e2 @ Rot.T
        b2d = self.kVectorPart(b2d, 1)
        b2dvector = self.multiVectorParts(b2d)[1:4]
        # print('b2dvector')
        # print(b2dvector)
        # print('b1dvector')
        # print(b1dvector)

        Er = self.rotationalError(b1, b1d, b2, b2d)
       
        Er_parts = self.multiVectorParts(Er)
        Er4, Er5, Er6 = Er_parts[4:7]
        # print(Er4)
        # print(Er5)
        # print(Er6)

        # TODO Before inputting w we have to transform it into the intertial frame
        # Is that Rq_mat * w *Rq_mat.T    or   Rot * w *Rot.T   ????  its Rq_mat
        #     p  -> e2e3           q ->  e3e1       r -> e1e2
        # w = w[0]*self.e2e3 + w[1]*self.e3e1 + w[2]*self.e1e2
        w = w[0]*self.e1e2 + w[1]*self.e2e3 + w[2]*self.e3e1

        winertial = np.matmul(np.matmul(Rq_mat, w),Rq_mat.T)
        winertial = self.kVectorPart(winertial,2) 
        winertial = self.multiVectorParts(winertial)[4:7]

        self.saved_w.append(winertial)
        

        x = np.reshape(np.concatenate(([Er4, Er5, Er6], winertial)), (6, 1))

        v = np.matmul(-Lr, x)
        tau = v - np.matmul(Bir, self.myf(x, JJ))     # I believe Js are al reves
        #print(tau)
        #print(np.shape(tau))
        #tau = np.array([[0.0],[1.0],[0.0]])
        #print(tau)
        # print(np.shape(tau))
        # print(tau)
        tauvec = tau[0] * self.e1e2 + tau[1] * self.e2e3 + tau[2] * self.e3e1

        


        tauB_parts = np.matmul(np.matmul(Rq_mat.T, tauvec),Rq_mat)
        tauB_parts = self.kVectorPart(tauB_parts,2)
        tauB_parts = self.multiVectorParts(tauB_parts)[4:7]
        
        tauB = np.array(tauB_parts)
        #print(tauB)
        return tauB, Trust

    def odomcallback(self, data):
        self.quaternion = data.pose.pose.orientation  #Supposed to be NED
        self.q0 = self.quaternion.w
        self.q1 = self.quaternion.x
        self.q2 = self.quaternion.y
        self.q3 = self.quaternion.z
        self.myQuat = np.array([self.q0,self.q3,self.q1,self.q2])

        #self.quatMatrix = self.q0*self.e0 + self.q1*self.e1e2 + self.q2*self.e2e3 + self.q3*self.e3e1 
        #print(self.quatMatrix)

        #print(self.myang(self.quatMatrix))
        #phi,theta,psi

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
        self.saved_refs.append(self.refxyz)
        if self.count < 2500:
            self.refxyz = [3,5,-10,0,0,0]
        elif self.count < 4500:
            self.refxyz = [0,5,-10,0,0,0]
        elif self.count < 5500:
            self.refxyz = [0,0,-10,0,0,0]
        elif self.count < 6500:
            self.refxyz = [3,0,-10,0,0,0]
        elif self.count < 7500:
            self.refxyz = [3,5,-10,0,0,0]
        else:
            print('completed')
        

        #GA Controller TAUS
        self.tausB, Thrust = self.control(self.xv, self.w, self.myQuat, self.refxyz,  self.m, self.g, self.JJ, self.Bir, self.Lp, self.Lr)

        #This tausB should also come in the order e1e2,e2e3,e3e1
        #print(self.tausB)
        self.saved_tau.append(self.tausB)

        Thrust = Thrust
        # Thrust = Thrust*(0.1/4)
        self.saved_Thrust.append(Thrust)
        

        #Normalize and then /4
        weight = self.m*self.g # = 8.24 N  Thats the max thrust output
        hoverThrottle = 0.0864  # Throttle per motor needed to maintain hover
        normFactor = hoverThrottle/weight

        #Mixer
        # self.Throttle1 = Thrust + self.tausB[0] - self.tausB[1] - self.tausB[2]
        # self.Throttle2 = Thrust + self.tausB[0] + self.tausB[1] + self.tausB[2]
        # self.Throttle3 = Thrust - self.tausB[0] + self.tausB[1] - self.tausB[2]
        # self.Throttle4 = Thrust - self.tausB[0] - self.tausB[1] + self.tausB[2]

        #ThrustNorm = Thrust*normFactor
        # print 'ThrustNorm: %s' %ThrustNorm

        # Assuming tau comes in the order e1e2 e2e3 e3e1
        self.Throttle1 = (Thrust + self.tausB[0] - self.tausB[1] + self.tausB[2])*normFactor
        self.Throttle2 = (Thrust - self.tausB[0] - self.tausB[1] - self.tausB[2])*normFactor
        self.Throttle3 = (Thrust + self.tausB[0] + self.tausB[1] - self.tausB[2])*normFactor
        self.Throttle4 = (Thrust - self.tausB[0] + self.tausB[1] + self.tausB[2])*normFactor

        

        self.data = [\
                [25,self.Throttle1, self.Throttle2, self.Throttle3, self.Throttle4, -998, -998, -998, -998],\
           ]

        # print 'taus Body: %s' % self.tausB
        # print 'motors: %s' % self.data[0][1:5]
        self.count += 1
        

    def sendToXPlane(self, event):
        #print(self.data)
        motors = self.data
        #print(motors)
        #print('pre time',rospy.get_time())
        self.client.sendDATA(motors)   # try to publish it in another node to verify the hz and echo
        #print('post time',rospy.get_time())

        motor_data = [float(x) for x in motors[0]]

        myMsg = Float64MultiArray()
        myMsg.data = motor_data


        motorSignals.publish(myMsg)

def signal_handler(sig, frame):
    """ Handle keyboard interrupt """
    print("\nInterrupt received, plotting quaternion data...")
    if QUADcon:
        plot_quaternion_data(QUADcon.saved_quat)
        plot_w_data(QUADcon.saved_w)
        plot_tau_data(QUADcon.saved_tau)
        plot_xyz_data(QUADcon.saved_xyz, QUADcon.saved_refs)
        plot_thrust_data(QUADcon.saved_Thrust)
        plot_xyz_data_3d(QUADcon.saved_xyz, QUADcon.saved_refs)
        plot_Rot_data(QUADcon.saved_Rot)
        plot_RotNQuat_data(QUADcon.saved_quat, QUADcon.saved_Rot)

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
    plt.plot(time_steps, quat_data[:, 1], label='z')
    plt.ylabel('q3')
    plt.legend()

    plt.subplot(4, 1, 3)
    plt.plot(time_steps, quat_data[:, 2], label='x')
    plt.ylabel('q1')
    plt.legend()

    plt.subplot(4, 1, 4)
    plt.plot(time_steps, quat_data[:, 3], label='y')
    plt.ylabel('q2')
    plt.legend()

    plt.xlabel('Time Step')
    plt.tight_layout()
    plt.savefig('XPlane_quaternion_plot.png')
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
    
if __name__ == '__main__':
    signal.signal(signal.SIGINT, signal_handler)
    rospy.init_node('listener', anonymous=True)
    motorSignals = rospy.Publisher('ControlSignals', Float64MultiArray, queue_size=10)
    QUADcon = QUADController()
    rospy.spin()



