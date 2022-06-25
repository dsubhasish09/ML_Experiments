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
    def __init__(self,cartpole):
        self.cartpole = cartpole
        
    def command(self, ref, state):
        tau = np.zeros((self.K.shape[0],1))
        return tau
    
class FullyActuated(CartPoleController):
    def __init__(self, cartpole, Kp, Kd):
        super(FullyActuated,self).__init__(cartpole)
        self.Kp = Kp
        self.Kd = Kd
        
    def command(self,ref,state):
        
        q = state[0:2]
        dq = state[2:4]
        ddq = self.Kp * (ref - q) + self.Kd * -dq
        tau = self.cartpole.getM(q) @ ddq + self.cartpole.getCG(q, dq)
        return tau 

if __name__ == "__main__":
    cartpole = CartPole(np.array([0,1]),1,1,1,9.81,0.1,0.1)
    Kp = np.array([[100,100]]).T
    Kd = np.array([[25,25]]).T
    controller = FullyActuated(cartpole,Kp,Kd)
    ref = np.array([[0,np.pi/2]]).T
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
    plt.plot(history[:,0],history[:,1])
        
    
        
