# -*- coding: utf-8 -*-
"""
Created on Sat Jun 25 15:03:11 2022

@author: subhasish
Controllers for cartpole
"""
import numpy as np
from cartpole import CartPole
import matplotlib.pyplot as plt 

class CartPoleController:
    """
    Base class for cart-pole controllers
    """
    def __init__(self,cartpole):
        self.cartpole = cartpole
        
    def command(self, ref, state):
        tau = np.zeros((self.K.shape[0],1))
        return tau
    
class FullyActuated(CartPoleController):
    """
    Fully actuated inverse dynamics joint-space controller
    """
    def __init__(self, cartpole, Kp, Kd, max_command_mag= np.array([[np.inf,np.inf]]).T):
        super(FullyActuated,self).__init__(cartpole)
        self.Kp = Kp
        self.Kd = Kd
        self.max_mag = max_command_mag
        
    def command(self,ref,state):
        
        q = state[0:2]
        dq = state[2:4]
        ddq = self.Kp * (ref - q) + self.Kd * -dq
        # tau = ddq
        tau = np.clip(self.cartpole.getM(q) @ ddq + self.cartpole.getCG(q, dq),-self.max_mag, self.max_mag)
        return tau 
    
class FullyActuatedTask(FullyActuated):
    """
    Fully actuated inverse dynamics task-space controller
    """
    def __init__(self,cartpole,Kp,Kd,max_command_mag = np.array([[np.inf,np.inf]]).T,  lambd = 0.01):
        super(FullyActuatedTask,self).__init__(cartpole,Kp,Kd,max_command_mag)
        self.lambd = lambd
    
    def command(self,ref, state):
        q = state[0:2]
        dq = state[2:4]
        curr = np.array([[q[0,0]+self.cartpole.length*np.sin(q[1,0])],
                         [self.cartpole.length*np.cos(q[1,0])]])
        J = np.array([[1, (self.cartpole.length*np.cos(q[1,0]))],
                      [0, (-self.cartpole.length*np.sin(q[1,0]))]])
        dJ = np.array([[0,(-self.cartpole.length*np.sin(q[1,0]))],
                       [0,(-self.cartpole.length*np.cos(q[1,0]))]])
        J_ = J+self.lambd*np.eye(2)
        J_inv = np.linalg.pinv(J_)
        ddq = J_inv @ (self.Kp * (ref-curr) - self.Kd * J @ dq - dJ @ dq) + (np.eye(2)-J_inv @ J_) @ (-100*q - 20*dq)
        tau = self.cartpole.getM(q) @ ddq + self.cartpole.getCG(q, dq)
        tau = np.clip(tau,-self.max_mag, self.max_mag)
        return tau

class UnderActuated(CartPoleController):
    def __init__(self, cartpole, Kp,Kd):
        super(UnderActuated, self).__init__(cartpole)
        self.Kp = Kp
        self.Kd = Kd
        
    def command(self, q2d, state):
        q1 = state[0,0]
        q2 = state[1,0]
        dq1 = state[2,0]
        dq2 = state[3,0]
        ddq2 = self.Kp * (q2d-q2) - self.Kd * dq2
        tau = ((self.cartpole.m1+self.cartpole.m2)*self.cartpole.g*np.sin(q2)/(np.cos(q2))
               + self.cartpole.m2*(-(dq2**2)*self.cartpole.length*np.sin(q2)+ddq2*self.cartpole.length*np.cos(q2))
               -(self.cartpole.m1+self.cartpole.m2)*self.cartpole.length*ddq2/(np.cos(q2)))
        return tau
    
    
if __name__ == "__main__":
    cartpole = CartPole(np.array([0,1]),1,1,1,9.81,0.1,0.1)
    Kp = 100
    Kd = 20
    controller = UnderActuated(cartpole,Kp,Kd)
    q0 = np.array([[0,np.pi]]).T
    dq0 = np.array([[0,0]]).T
    state = np.zeros((4,1))
    state[0:2] = q0
    state[2:4] = dq0
    h=0.01
    T_end = 10
    iters =  int(T_end/h)
    history = np.zeros((iters+1, 5))
    history[0,0] = 0
    history[0,1:] = state[:,0]
    Tau = []
    for i in range(iters):
        q2d = 0#np.pi - (np.pi/4)*np.sin(2*np.pi*2*(history[i,0]+h))
        tau =  np.array([[controller.command(q2d, state), 0]]).T
        Tau.append(tau[0,0])
        state += cartpole.rk4Step(state, tau, h)
        history[i+1,0] = history[i,0] + h
        history[i+1,1:] = state[:,0]
    plt.figure()
    plt.plot(history[:,0],history[:,1])
    plt.figure()
    plt.plot(history[:,0],history[:,2])
    plt.figure()
    plt.plot(history[:,0],history[:,3])
    plt.figure()
    plt.plot(history[:,0],history[:,4])
    plt.figure()
    plt.plot(Tau[:])
    print(max(Tau))
    
        
    
        
