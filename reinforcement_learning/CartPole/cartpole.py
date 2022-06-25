# -*- coding: utf-8 -*-
"""
Created on Thu Jun 23 17:59:24 2022

@author: subhasish
Cart-pole simulation
"""
import numpy as np
import matplotlib.pyplot as plt
class CartPole:
    
    def __init__(self,cart_lim: np.ndarray, length: float, mass_cart: float, mass_rod: float, g: float):
        self.cart_lim = cart_lim
        self.length = length
        self.m1 = mass_cart
        self.m2 = mass_rod
        self.g = g
    
    def inverseDynamics(self, q, dq, ddq):
        tau = np.array([[self.m2 * (ddq[0,0] - (dq[1,0]**2)*self.length*np.sin(q[1,0]) + ddq[1,0]*self.length*np.cos(q[1,0])) + self.m1*ddq[0,0],
                         self.m2 * self.length *(-self.g*np.sin(q[1,0]) + ddq[0,0]*np.cos(q[1,0]) + ddq[1,0]*self.length)]]).T
        return tau
    
    def getM(self,q):
        M = np.zeros((2,2))
        dq = np.zeros((2,1))
        g = self.g
        self.g = 0
        
        ddq = np.array([[1, 0]]).T
        M[:,0] = self.inverseDynamics(q, dq, ddq).squeeze()
        
        ddq = np.array([[0, 1]]).T
        M[:,1] = self.inverseDynamics(q, dq, ddq).squeeze()
        
        self.g = g
        return M
    
    def getCG(self,q,dq):
        ddq = np.zeros((2,1))
        return self.inverseDynamics(q, dq, ddq)
    
    def directDynamics(self,state,tau):
        q = state[0:2]
        dq = state[2:4]
        dState = np.zeros((4,1))
        dState[0:2] = dq
        dState[2:4] = np.linalg.inv(self.getM(q)) @ (tau - self.getCG(q, dq))
        return dState
    
    def rk4Step(self,state,tau,h):
        k1 = self.directDynamics(state,tau)
        k2 = self.directDynamics(state + h * k1/2,tau)
        k3 = self.directDynamics(state + h * k2/2,tau)
        k4 = self.directDynamics(state + h * k3,tau)
        return (1/6) * h * (k1 + 2*k2 + 2*k3 + k4)
    
    def unforcedIntegration(self,q0, dq0, iters, h):
        state = np.zeros((4,1))
        state[0:2] = q0
        state[2:4] = dq0
        tau = np.zeros((2,1))
        history = np.zeros((iters+1,5))
        history[0,0] = 0
        history[0,1:] = state[:,0]
        for i in range(iters):
            state = state + self.rk4Step(state, tau, h)
            history[i+1,1:] = state[:,0] 
            history[i+1,0] = history[i,0] + h
        return history
    
    def drawCartPole(self, q, ax):
        ax.plot(self.cart_lim,[0,0],'b')
        ax.plot(q[0,0],0,'ro')
        ax.plot([q[0,0]-0.05, q[0,0]-0.05, q[0,0]+0.05, q[0,0]+0.05, q[0,0]-0.05],[0.05,-0.05,-0.05,0.05,0.05])
        ax.plot([q[0,0], q[0,0] + self.length * np.sin(q[1,0])],[0,self.length * np.cos(q[1,0])])
        ax.plot(q[0,0] + self.length * np.sin(q[1,0]),self.length * np.cos(q[1,0]),'go')
        
   
if __name__ == "__main__":
    cartpole = CartPole(np.array([0,1]),1,1,1,9.81)
    q = np.array([[0,np.pi/2]]).T
    dq = np.array([[0,0]]).T
    ddq = np.array([[0,1]]).T
    tau = cartpole.inverseDynamics(q, dq, ddq)
    M = cartpole.getM(q)
    CG = cartpole.getCG(q, dq)
    tau = np.array([[0,0]]).T
    state = np.array([[0,np.pi/2,0,0]]).T
    dState = cartpole.rk4Step(state, tau, 0.001)
    history = cartpole.unforcedIntegration(q, dq, 10000, 0.001)
    plt.plot(history[:,0],history[:,1])
    