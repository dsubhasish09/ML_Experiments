#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 17 21:09:39 2022

@author: dsubhasish
"""
import gym
import numpy as np
from gym import spaces
from gym.utils import seeding
from omegaconf import DictConfig
import matplotlib.pyplot as plt

class CartPole_Base(gym.Env):
    def __init__(self, cfg: DictConfig, seed:int = 0):
        self.m1 = cfg.m1
        self.m2 = cfg.m2
        self.l = cfg.l
        self.fp = cfg.fp
        self.fr = cfg.fr
        self.g = cfg.g

        self.state_lim = np.array([[cfg.x_lim,
                                    cfg.theta_lim,
                                    cfg.xdot_lim,
                                    cfg.thetadot_lim]]).T
        
        self.ulim = np.array([[cfg.f_lim,
                               cfg.tau_lim]]).T
        
        self.u_residual_lim = np.array([[cfg.f_residual_lim,
                                         cfg.tau_residual_lim]]).T
        
        self.u_residual_scale = np.array([[cfg.f_residual_scale,
                                         cfg.tau_residual_scale]]).T
        
        self.u = np.zeros((2,1)).T
        
        self.Q = np.array([[cfg.x_penalty, 0, 0, 0],
                           [0, cfg.theta_penalty, 0, 0],
                           [0, 0, cfg.xdot_penalty, 0],
                           [0, 0, 0, cfg.thetadot_penalty]])
        
        self.R = np.array([[cfg.f_penalty, 0],
                           [0, cfg.tau_penalty]])
        
        self.survival_reward = cfg.survival_reward
        
        self.state = np.zeros((4,1))
        self.delta_t = cfg.delta_t
        self.t_max = cfg.t_max
        self.t = 0
        self.max_episode_steps = int(self.t_max / self.delta_t)
        self.obs_lim = np.vstack([self.state_lim, self.ulim])
        self.observation_space = spaces.Box(low=-self.obs_lim.squeeze(), high=self.obs_lim.squeeze(), dtype = np.float32)
        self.action_space = spaces.Box(low=-self.u_residual_lim.squeeze(), high=self.u_residual_lim.squeeze(), dtype = np.float32)
   
    def getM(self, state):
        return np.array([[(self.m1 + self.m2), self.m2 * self.l * np.cos(state[1,0])],
                         [ self.m2 * self.l * np.cos(state[1,0]), self.m2 * self.l ** 2]])
    
    def getCG(self, state):
        return np.array([[-self.m2 * state[3,0] ** 2 * self.l * np.sin(state[1,0]),
                          -self.m2 * self.g * self.l * np.sin(state[1,0])]]).T
    
    def getFriction(self, state):
        return np.array([[self.fp * state[2,0],
                          self.fr * state[3,0]]]).T
    
    def forward_dynamics(self, state, u):
        statedot = np.zeros((4,1))
        statedot[0:2,0] = state[2:4,0]
        statedot[2:4] = np.linalg.inv(self.getM(state)) @ (u - self.getFriction(state) - self.getCG(state))
        return statedot
    


    def step(self, u):
        u_residual = u.reshape((2,1))

        u_ = self.u + self.u_residual_scale * u_residual

        self.u = np.array([[max(-self.ulim[0,0], min(self.ulim[0,0],u_[0,0])),
                            max(-self.ulim[1,0], min(self.ulim[1,0],u_[1,0]))]]).T

        ds1 = self.forward_dynamics(self.state, self.u)
        ds2 = self.forward_dynamics(self.state + self.delta_t * ds1/2, self.u)
        ds3 = self.forward_dynamics(self.state + self.delta_t * ds2/2, self.u)
        ds4 = self.forward_dynamics(self.state + self.delta_t * ds3, self.u)

        state = self.state + (1/6) * self.delta_t * (ds1 + 2*ds2 + 2*ds3 + ds4)

        # angle wrap over-
        if state[1,0] > np.pi:
            state[1,0] = -2*np.pi + state[1,0]
        if state[1,0] < -np.pi:
            state[1,0] = 2*np.pi + state[1,0]

        #velocity limit
        if abs(state[2,0]) > self.state_lim[2,0]:
            state[2,0] = np.sign(state[2,0]) * self.state_lim[2,0]

        if abs(state[3,0]) > self.state_lim[3,0]:
            state[3,0] = np.sign(state[3,0]) * self.state_lim[3,0]

        self.state = state

        reward = self.reward_fcn(state, self.u)
        self.t += self.delta_t
        done = self.termination_fcn(state, u)

        info = {}

        return np.vstack([self.state,self.u]).squeeze(), reward.squeeze(), done, info        
    
    def reward_fcn(self,state, u):
        return  (self.survival_reward - (state.T @ self.Q @ state + u.T @ self.R @ u)) 
    
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    
    def termination_fcn(self, state, u):

        if np.any(np.abs(state) > self.state_lim) or self.t > self.t_max:
            return True
        else: return False
    
    def reset(self, init_state = None):   
        self.state = np.zeros((4,1))
        try:
            if init_state==None:
                self.state[0,0] = np.random.uniform(low = -self.state_lim[0,0], high = self.state_lim[0,0])
                self.state[1,0] = np.random.uniform(low=-np.pi, high=np.pi)
        except: 
            self.state = init_state
        self.u = np.zeros((2,1))
        self.t = 0
        # self.state[3,0] = 0.5
        return np.vstack([self.state,self.u]).squeeze()
    
    def animate(self, states):
        fig = plt.figure(1)
        q = states[0,:]
        xlim = [-self.state_lim[0,0]-self.l-0.05, self.state_lim[0,0]+self.l+0.05]
        ax = fig.add_subplot(111)
        plt.ion()
        plt.gca().set_aspect('equal')
        plt.xlim(xlim)
        plt.ylim([-self.l-0.05,self.l+0.05])
        cart_rail, = ax.plot([-self.state_lim[0,0],self.state_lim[0,0]],[0,0])
        cart_joint, = ax.plot(q[0],0,'ro')
        cart_box, = ax.plot([q[0]-0.05, q[0]-0.05, q[0]+0.05, q[0]+0.05, q[0]-0.05],[0.05,-0.05,-0.05,0.05,0.05])
        pole, = ax.plot([q[0], q[0] + self.l * np.sin(q[1])],[0,self.l * np.cos(q[1])])
        pole_end, = ax.plot(q[0] + self.l * np.sin(q[1]),self.l* np.cos(q[1]),'go')
        fig.show()
        fig.canvas.draw()
        for i in range(0,states.shape[0],int(0.03/self.delta_t)):
            q = states[i,:]
            cart_joint.set_data(q[0],0)
            cart_box.set_data([q[0]-0.05, q[0]-0.05, q[0]+0.05, q[0]+0.05, q[0]-0.05],[0.05,-0.05,-0.05,0.05,0.05])
            pole.set_data([q[0], q[0] + self.l * np.sin(q[1])],[0,self.l * np.cos(q[1])])
            pole_end.set_data(q[0] + self.l * np.sin(q[1]),self.l* np.cos(q[1]))
            plt.xlim(xlim)
            plt.ylim([-self.l-0.05,self.l+0.05])
            fig.show()
            fig.canvas.draw()