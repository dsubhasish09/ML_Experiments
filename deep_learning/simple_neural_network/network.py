# -*- coding: utf-8 -*-
"""
Created on Wed Feb  2 20:56:29 2022

@author: devdu

This file implements a class that abstracts a neural network.
"""

import numpy as np
import matplotlib.pyplot as plt

class Network(object):
    def __init__(self,layers,loss):
        """
        Abstracts a Neural Network
        Parameters
        ----------
        layers : list
            ordered list of layers.
        loss : Loss Object
            Object abstracting loss.
        Returns
        -------
        None.
        """
        self.layers=layers
        self.loss=loss
        
    def fprop(self,X,t,batch_size):
        """
        
        Forward Propagation
        Parameters
        ----------
        X : np.ndarray of shape (N_batch,N_in)
            Array of inputs.
        t : np.ndarray of shape (N_batch,N_class)
            Array of targets.
        batch_size : int
            Batch Size.
        Returns
        -------
        Y : np.ndarray of shape (N_batch,N_class)
            Output of the network.
        loss : float
            Current Loss.
        """
        Y=X.copy()
        for k in range(len(self.layers)):
            Y=self.layers[k].fprop(Y)
        loss=self.loss.fprop(Y,t,batch_size)
        return Y,loss
    
    def bprop(self,batch_size):
        """
        Back Propagation
        Parameters
        ----------
        batch_size : int
            Batch Size.
        Returns
        -------
        None.
        """
        dE=self.loss.bprop(batch_size)
        for k in range(len(self.layers)-1,-1,-1):
            dE=self.layers[k].bprop(dE)
        
    def update(self,rate):
        """
        Update Network Parameters
        Parameters
        ----------
        rate : float
            Update step size.
        Returns
        -------
        None.
        """
        for layer in self.layers:
            if layer.trainable:
                layer.update(rate)
        
    def train(self,X,t,rate,batch_size,n_epochs):
        """
        
        Trains the Network
        Parameters
        ----------
        X : np.ndarray of shape (N_batch,N_in)
            Array of inputs.
        t : np.ndarray of shape (N_batch,N_class)
            Array of targets.
        rate : float
            Update step size.
        batch_size : int
            Batch Size.
        n_epochs : int
            Number of Epochs.
        Returns
        -------
        loss : list
            List of losses during training.
        """
        N,D=X.shape
        Losses=[]
        fig=plt.figure()
        ax = fig.add_subplot(111)
        plt.ion()
        fig.show()
        fig.canvas.draw()
        for i in range(n_epochs):
            k=0
            condition=True
            loss=0
            while condition:
                end_idx=(k+1)*batch_size
                if end_idx>=N:
                    end_idx=N
                    condition=False
                X_batch=X[k*batch_size:end_idx:]
                t_batch=t[k*batch_size:end_idx:]
                loss+=self.fprop(X_batch,t_batch,t_batch.shape[0])[1]
                self.bprop(t_batch.shape[0])
                self.update(rate)
                k+=1
            Losses.append(loss/k)
            print("Epoch "+str(i+1)+"  Average Loss="+str(round(loss/k,4)),end="\r")
            # if visualize:
            ax.clear()
            ax.plot(np.arange(1,len(Losses)+1),Losses)
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.title("Loss vs Epoch")
            fig.canvas.draw()
        return Losses
    
    def predict(self,X):
        """
        Predict and compute accuracy on test/validation data.
        Parameters
        ----------
        X : np.ndarray of shape (N_batch,N_in)
            Array of inputs.
       
        Returns
        -------
        Y : np.ndarray of shape (N_batch,)
            Array of predicted category.
        """
        Y=X.copy()
        for k in range(len(self.layers)):
            Y=self.layers[k].fprop(Y)
        Y=np.argmax(Y,axis=1)
        return Y