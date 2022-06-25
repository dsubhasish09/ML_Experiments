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
        
        tau = np.clip(self.cartpole.getM(q) @ ddq + self.cartpole.getCG(q, dq),-self.max_mag, self.max_mag)
        return tau

if __name__ == "__main__":
    cartpole = CartPole(np.array([0,1]),1,1,1,9.81,0.1,0.1)
    Kp = np.array([[10,10]]).T
    Kd = np.array([[2,2]]).T
    controller = FullyActuatedTask(cartpole,Kp,Kd)
    ref = np.array([[cartpole.length*np.sin(0)],
                    [cartpole.length*np.cos(0)]])
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
    for i in range(iters):
        tau =  controller.command(ref, state)
        state += cartpole.rk4Step(state, tau, h)
        history[i+1,0] = history[i,0] + h
        history[i+1,1:] = state[:,0]
    plt.plot(history[:,0],history[:,2])
        
    
        
