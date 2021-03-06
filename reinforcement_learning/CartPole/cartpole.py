# -*- coding: utf-8 -*-
"""
Created on Thu Jun 23 17:59:24 2022

@author: subhasish
Cart-pole simulation
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
# import time 
from functools import partial

class CartPole:
    
    def __init__(self,cart_lim: np.ndarray, length: float, mass_cart: float, mass_rod: float, g: float, fp: float=0, fr: float=0):
        self.cart_lim = cart_lim
        self.length = length
        self.m1 = mass_cart
        self.m2 = mass_rod
        self.g = g
        self.fp = fp
        self.fr = fr
    
    def inverseDynamics(self, q, dq, ddq):
        tau = np.array([[self.m2 * (ddq[0,0] - (dq[1,0]**2)*self.length*np.sin(q[1,0]) + ddq[1,0]*self.length*np.cos(q[1,0])) + self.m1*ddq[0,0],
                         self.m2 * self.length *(-self.g*np.sin(q[1,0]) + ddq[0,0]*np.cos(q[1,0]) + ddq[1,0]*self.length)]]).T
        tau += self.getFriction(q, dq)
        return tau
    
    def getM(self,q):
        M = np.array([[(self.m1+self.m2), (self.m2*self.length*np.cos(q[1,0]))],
                      [(self.m2*self.length*np.cos(q[1,0])), self.m2*self.length**2]])
        return M
    
    def getCG(self,q,dq):
        CG = np.array([[(-self.m2*(dq[1,0]**2)*self.length*np.sin(q[1,0]))],
                       [(-self.m2*self.g*self.length*np.sin(q[1,0]))]])
        return CG
    def getFriction(self, q, dq):
        return np.array([[self.fp * dq[0,0], self.fr * dq[1,0]]]).T 
    def directDynamics(self,state,tau):
        q = state[0:2]
        dq = state[2:4]
        dState = np.zeros((4,1))
        dState[0:2] = dq
        dState[2:4] = np.linalg.inv(self.getM(q)) @ (tau - self.getFriction(q, dq) - self.getCG(q, dq))
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
    
    def drawCartPole(self, i, history, ax, xlim):
        q = history[i,1:3].reshape((2,1))
        cart_pivot, = ax.plot(q[0,0],0,'ro',animated = True)
        cart, = ax.plot([q[0,0]-0.05, q[0,0]-0.05, q[0,0]+0.05,
                         q[0,0]+0.05, q[0,0]-0.05],[0.05,-0.05,-0.05,0.05,0.05],'k',
                         animated = True)
        pole, = ax.plot([q[0,0], q[0,0] + self.length * np.sin(q[1,0])],
                        [0,self.length * np.cos(q[1,0])],'k',animated = True)
        tip, = ax.plot(q[0,0] + self.length * np.sin(q[1,0]),self.length * np.cos(q[1,0]),'go',animated = True)
        text = ax.text(np.average(xlim),-self.length,
                       "Time = {time:.2f} seconds".format(time=round(history[i,0],2)),
                       horizontalalignment='center',
                       animated = True)
        return [cart_pivot, cart, pole, tip, text]
   
    def animateHistory(self, history,title = "Cart-Pole Animation",interval=10):
        xlim = [np.min([np.min(history[:,1]+self.length* np.sin(history[:,2])),np.min(history[:,1])])-0.25,
                np.max([np.max(history[:,1]+self.length* np.sin(history[:,2])),np.max(history[:,1])])+0.25]
        fig, ax = plt.subplots()
        cart_track, = ax.plot(self.cart_lim,[0,0],'b')
        cart_pivot, = ax.plot([],[],'ro',animated = True)
        cart, = ax.plot([],[],animated = True)
        pole, = ax.plot([],[],animated = True)
        tip, = ax.plot([],[],'go',animated = True)
        text = ax.text(np.average(xlim),self.length/2,"",horizontalalignment='center',animated = True)
        plt.xlim(xlim)
        plt.ylim([-self.length-0.05,self.length+0.05])
        title = plt.title(title)
        plt.gca().set_aspect('equal')
        animate = partial(self.drawCartPole,history = history, ax = ax, xlim = xlim)
        plt.show()
        plots = []
        for i in range(len(history)):
            plots.append(animate(i))
        anim = animation.ArtistAnimation(fig, plots, interval=interval, blit=True,
                                        repeat_delay=1000)
        plt.show()
        return anim

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
    history = cartpole.unforcedIntegration(q, dq, 10000, 0.01)
    plt.plot(history[:,0],history[:,1])
    cartpole.animateHistory(history)  