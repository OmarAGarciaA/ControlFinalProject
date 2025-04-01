#!/usr/bin/env python3

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
from scipy import linalg as ln
# import control as ct
# from scipy.linalg import expm, inv

import xplane_ros.msg as xplane_msgs
import rosplane_msgs.msg as rosplane_msgs
from nav_msgs.msg import Odometry

QUADcon = None
UDP_PORT = 49005

# Open a Socket on UDP Port 49000
UDP_IP = "127.0.0.1"
sock = socket.socket(socket.AF_INET, # Internet
                     socket.SOCK_DGRAM) # UDP
sock.bind((UDP_IP, UDP_PORT))

class QUADController():
    def __init__(self):
        rospy.init_node('listener', anonymous=True)
        self.Plotting = False
        self.saved_data = []

        self.yaw_init = None
        self.is_yaw_set = False

        self.time_init = None
        self.is_time_set = False

        self.xv = np.zeros([6])
        self.w = np.zeros([3])
        self.myQuat = np.array([1,0,0,0])
        # self.q = 0
        # self.p = 0
        # self.r = 0
        # self.altura = 0
        # self.vz = 0
        self.x_des = 0
        self.dx_des = 0
        self.y_des = 0
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

        self.m = 0.025
        self.g = 9.81

        self.Jx = 4.856e-3
        self.Jy = 4.856e-3
        self.Jz = 8.801e-3

        self.JJ = [self.Jx, self.Jy, self.Jz]

        Br = np.vstack((np.zeros((3, 3)), np.array([[1 / self.Jx, 0, 0], [0, 1 / self.Jy, 0], [0, 0, 1 / self.Jz]])))

        #self.Bir = ln.inv(Br.T @ Br) @ Br.T

        self.Lp = np.array([[99.2936, -0.0, -0.0, 14.127, -0.0, -0.0],
                       [-0.0, 99.2936, -0.0, -0.0, 14.127, -0.0],
                       [-0.0, -0.0, 99.2936, -0.0, -0.0, 14.127]])

        self.Lr = np.array([[-369.383, -0.0, 0.0, 4.56281, -0.0, -0.0],
                       [-0.0, -369.383, 0.0, -0.0, 4.56281, -0.0],
                       [-0.0, -0.0, -531.154, -0.0, -0.0, 6.84918]])

        #self.Lp, self.Lr = self.LQRgains()



        rospy.Subscriber("/xplane/flightmodel/odom", Odometry, self.odomcallback)
        rospy.Subscriber("/fixedwing/xplane/state", rosplane_msgs.State, self.callbackpqr)
        rospy.Subscriber("/xplane/flightmodel/global_state", xplane_msgs.GlobalState, self.callback)

    # def LQRgains(self):
    #     dt = 1e-3
    #     Ar = np.block([[np.zeros((3, 3)), -2.0 * np.eye(3)],
    #         [np.zeros((3, 6))]
    #     ])
    #     Br = np.block([
    #         [np.zeros((3, 3))],
    #         [np.diag([1 / self.Jx, 1 / self.Jy, 1 / self.Jz])]
    #     ])

    #     Mr = np.block([
    #         [Ar, Br],
    #         [np.zeros((3, 9))]
    #     ])
    #     Mdr = expm(dt * Mr)
    #     Adr = Mdr[:6, :6]
    #     Bdr = Mdr[:6, 6:9]

    #     Qr = np.block([
    #         [np.diag([0.1, 1, 1]) * 1e2, np.zeros((3, 3))],
    #         [np.zeros((3, 3)), np.diag([0.1, 1, 1]) * 1e-2]
    #     ])  # Weighting matrix for state
    #     Rr = np.diag([1e-5] * 3)  # Weighting matrix for input


    #     Lr = ct.dlqr(Adr, Bdr, Qr, Rr)[0]
    #     Lr[np.abs(Lr)< 1e-8]= 0

    #     # Design Position controller
    #     Ap = np.block([
    #         [np.zeros((3, 3)), np.eye(3)],
    #         [np.zeros((3, 6))]
    #     ])
    #     Bp = np.block([
    #         [np.zeros((3, 3))],
    #         [np.eye(3)]
    #     ])

    #     Mp = np.block([
    #         [Ap, Bp],
    #         [np.zeros((3, 9))]
    #     ])
    #     Mdp = expm(dt * Mp)
    #     Adp = Mdp[:6, :6]
    #     Bdp = Mdp[:6, 6:9]

    #     Qp = np.block([
    #         [np.diag([1] * 3) * 1e2, np.zeros((3, 3))],
    #         [np.zeros((3, 3)), np.diag([1] * 3) * 1e-2]
    #     ])  # Weighting matrix for state
    #     Rp = np.diag([1] * 3)  # Weighting matrix for input


    #     Lp = ct.dlqr(Adp, Bdp, Qp, Rp)[0]
    #     Lp[np.abs(Lp)< 1e-8]= 0

    #     return Lp, Lr


    # Decomposes the multivector M into its scalar, vector, bivector, and trivector parts
    def multiVectorParts(self, M):
        m1 = np.trace((M @ self.e1 + self.e1 @ M) / 2) / 4
        m2 = np.trace((M @ self.e2 + self.e2 @ M) / 2) / 4
        m3 = np.trace((M @ self.e3 + self.e3 @ M) / 2) / 4
        m4 = -np.trace((M @ self.e1e2 + self.e1e2 @ M) / 2) / 4
        m5 = -np.trace((M @ self.e2e3 + self.e2e3 @ M) / 2) / 4
        m6 = -np.trace((M @ self.e3e1 + self.e3e1 @ M) / 2) / 4
        m7 = -np.trace((M @ self.e1e2e3 + self.e1e2e3 @ M) / 2) / 4
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
        R1 = b1 @ b1d  # Rotation equation
        B1 = self.kVectorPart(R1, 2)  # Wedge product
        if self.multivectorNorm(B1) < 1e-7:
            R10 = self.multiVectorParts(R1)[0]
            if np.sign(R10 + 0.5) == -1.0:  # Parallel Bivector
                if (self.multivectorNorm(b1 - self.e1) < 1e-7) or (self.multivectorNorm(-b1 - self.e1) < 1e-7):
                    a = self.kVectorPart(b1 @ self.e2, 2)
                else:
                    a = self.kVectorPart(b1 @ self.e1, 2)
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
        b2p = ExpmJ1th1 @ b2 @ ln.expm(j1 * th1 / 2)
        R2 = b2p @ b2d
        B2 = self.kVectorPart(R2, 2)
        if self.multivectorNorm(B2) < 1e-7:
            R20 = self.multiVectorParts(R2)[0]
            if np.sign(R20 + 0.5) == -1.0:  # Parallel Bivector
                if (self.multivectorNorm(b2 - self.e2) < 1e-7) or (self.multivectorNorm(-b2 - self.e2) < 1e-7):
                    a = self.kVectorPart(b2 @ self.e1, 2)
                else:
                    a = self.kVectorPart(b2 @ self.e2, 2)
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

        Re = ln.expm(-j2 * th2 / 2) @ ExpmJ1th1
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
        Rb = self.e0 + f3 @ e3
        Rb = self.kVectorPart(Rb, 0) + self.kVectorPart(Rb, 2)
        Rb = Rb / self.multivectorNorm(Rb)
        return Rb

    def control(self, xv, w, Rq, refxyz, m, g, JJ, Bir, Lp, Lr):
        # Compute the Rotation reference
        dv = -Lp @ np.reshape((xv - refxyz), (6, 1))
        dv = dv[0] * self.e1 + dv[1] * self.e2 + dv[2] * self.e3

        # Compute Trust
        Trust = m * self.multivectorNorm(-dv + g * self.e3)

        # Compute actual b1 and b2
        Rq_mat = Rq[0] * self.e0 + Rq[1] * self.e1e2 + Rq[2] * self.e2e3 + Rq[3] * self.e3e1
        Rq_mat = Rq_mat / self.multivectorNorm(Rq_mat)

        b1 = self.kVectorPart(Rq_mat @ self.e1 @ Rq_mat.T, 1)
        b2 = self.kVectorPart(Rq_mat @ self.e2 @ Rq_mat.T, 1)

        # Compute the rotation
        b3rot = -(m * dv - m * g * self.e3) / Trust

        if self.multivectorNorm(b3rot) > 1e-9:
            b3rot = b3rot / self.multivectorNorm(b3rot)
            Rot = self.estR(b3rot, self.e3)
        else:
            Rot = self.e0  # Identity matrix

        b1d = Rot @ self.e1 @ Rot.T
        b1d = self.kVectorPart(b1d, 1)

        b2d = Rot @ self.e2 @ Rot.T
        b2d = self.kVectorPart(b2d, 1)

        Er = self.rotationalError(b1, b1d, b2, b2d)
        Er_parts = self.multiVectorParts(Er)
        Er4, Er5, Er6 = Er_parts[4:7]

        x = np.reshape(np.concatenate(([Er4, Er5, Er6], w)), (6, 1))

        v = -Lr @ x
        τ = v - Bir @ self.myf(x, JJ)
        τvec = τ[0] * self.e1e2 + τ[1] * self.e2e3 + τ[2] * self.e3e1
        τB_parts = self.multiVectorParts(Rq_mat.T @ τvec @ Rq_mat)
        τB = np.array(τB_parts[4:7])

        return τB, Trust


    def odomcallback(self, data):
        self.quaternion = data.pose.pose.orientation  #Supposed to be NED
        self.q0 = self.quaternion.w
        self.q1 = self.quaternion.x
        self.q2 = self.quaternion.y
        self.q3 = self.quaternion.z
        self.myQuat = np.array([self.q0,self.q1,self.q2,self.q3])
        
        self.xi = data.pose.pose.position
        self.x = self.xi.x
        self.y = self.xi.y
        self.z = self.xi.z
        
        self.dxi = data.twist.twist.linear
        self.dx = self.dxi.x
        self.dy = self.dxi.y
        self.dz = self.dxi.z

        self.xv = np.array([self.x,self.y,self.z,self.dx,self.dy,self.dz])

        self.secs = data.header.stamp.secs
    
    def callbackpqr(self, data):
        #  Omegas xyz  Body frame
        self.p = data.p
        self.q = data.q
        self.r = data.r

        self.w = np.array([self.p, self.q, self.r])

        # Could I get the quaternion NED from here?

    def callback(self, data):
        self.phi = data.roll 
        self.theta = data.pitch
        self.psi = data.heading

        #GA Controller TAUS
        tausB, Thrust = self.control(self.xv, self.w, self.myQuat, self.refxyz,  self.m, self.g, self.JJ, self.Bir, self.Lp, self.Lr)

        Thrust = Thrust

        #Mixer
        self.Throttle1 = Thrust + tausB[0] - tausB[1] - tausB[2]
        self.Throttle2 = Thrust + tausB[0] + tausB[1] + tausB[2]
        self.Throttle3 = Thrust - tausB[0] + tausB[1] - tausB[2]
        self.Throttle4 = Thrust - tausB[0] - tausB[1] + tausB[2]
    
        with utlis.XPlaneConnect() as client:
    
            self.data = [\
                    [25,self.Throttle1, self.Throttle2, self.Throttle3, self.Throttle4, -998, -998, -998, -998],\
                    [8, -998,  -998, -998,  -998, -998, -998, -998, -998],\
    	       ]
            client.sendDATA(self.data)

        # Saving data for plotting
        if self.Plotting:
            self.saved_data.append([self.actual_time])
            self.saved_data[-1].append(self.roll2)

def handle_interrupt(signal, frame):
    t = QUADcon.saved_data
    time = [row[0] for row in t]
    roll = [row[1] for row in t]

    # Exit the program
    sys.exit(0)
    
def main():
    global QUADcon
    QUADcon = QUADController()
    signal.signal(signal.SIGINT, handle_interrupt)
    rospy.spin()

if __name__ == '__main__':
    main()


