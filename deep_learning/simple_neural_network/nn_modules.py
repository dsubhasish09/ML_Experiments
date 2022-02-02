# -*- coding: utf-8 -*-
"""
Created on Wed Feb  2 20:44:39 2022

@author: devdutt_subhasish

This file implements some classes which abstract different neural network layers
"""
import numpy as np
vect_diag=np.vectorize(np.diag,signature='(n)->(n,n)')

class Module(object):
    """
    Base class for all modules
    """
    def __init__(self,trainable:bool):
        self.trainable=trainable
    
    def fprop(self,X):
        raise NotImplementedError 
        
    def bprop(self,dE):
        raise NotImplementedError
    
class TrainableModule(Module):
    """
    Base class for all modules with trainable parameters
    """
    def __init__(self):
        super(TrainableModule,self).__init__(True)
        
    def update(self,rate):
        raise NotImplementedError

class UntrainableModule(Module):
    """
    Base class for all modules which do not have any trainable parameter
    """
    def __init__(self):
        super(UntrainableModule,self).__init__(False)
        
class Loss(UntrainableModule):
    """
    Base class for a loss function layer
    """
    def __init__(self):
        super(Loss,self).__init__()
    
    def fprop(self,z,t,batch_size):
        raise NotImplementedError 
    
    def bprop(self,batch_size):
        raise NotImplementedError
        
class Linear(TrainableModule):
    def __init__(self,N_in,N_out):
        """
        
        Abstracts a linear layer
        Parameters
        ----------
        N_in : int
            Number of incoming connections.
        N_out : float
            Number of outgoing connections.

        Returns
        -------
        None.

        """
        super(Linear,self).__init__()
        self.W=np.random.randn(N_in,N_out)*np.sqrt(2/(N_in+N_out))
        self.b=np.zeros((N_out,1))
        self.dW=np.zeros((N_in,N_out))
        self.db=np.zeros((N_out,1))
    
    def fprop(self,X):
        """
        Forward propagation

        Parameters
        ----------
        X : np.ndarray of shape (N_batch,N_in)
            Input.

        Returns
        -------
        np.ndarray of shape of shape (N_batch,N_out)
            Output.

        """
        self.cache=X.copy()
        return self.cache @ self.W + self.b.T
    
    def bprop(self,dE):
        """
        Back Propagation

        Parameters
        ----------
        dE : np.ndarray of shape (N_batch,N_out)
            Gradient w.r.t output.

        Returns
        -------
        np.ndarray of shape (N_batch,N_in)
            Gradient w.r.t input.

        """
        self.dW[:,:]=self.cache.T @ dE
        self.db[:,:]=dE.T.sum(axis=1,keepdims=True)
        return dE @ self.W.T
    
    def update(self,rate):
        """
        
        Updates weights and bias
        Parameters
        ----------
        rate : float
            Update step size.

        Returns
        -------
        None.

        """
        self.W=self.W - rate * self.dW
        self.b=self.b - rate * self.db
        
class SoftMax(UntrainableModule):
    def __init__(self):
        """
        
        Abstracts a softmax layer
        Parameters
        ----------
        N_class : int
            Number of categories.

        Returns
        -------
        None.

        """
        super(SoftMax,self).__init__()
        
    def fprop(self,x):
        """
        
        Forward Propagation
        Parameters
        ----------
        x : np.ndarray of shape (N_batch,N_class)
            Input.

        Returns
        -------
        np.ndarray of shape of shape (N_batch,N_class)
            Output.

        """
        exp=np.exp(x)
        softmax=(exp/exp.sum(axis=1,keepdims=True))
        
        self.Jac=vect_diag(softmax)-softmax[:,:,np.newaxis] @ softmax[:,np.newaxis,:]
        return softmax
    
    def bprop(self,dE):
        """
        
        Back Propagation
        Parameters
        ----------
        dE : np.ndarray of shape (N_batch,N_class)
            Gradient w.r.t output.

        Returns
        -------
        np.ndarray of shape (N_batch,N_class)
            Gradient w.r.t input.

        """
        return (dE[:,np.newaxis,:] @ self.Jac).reshape(dE.shape)
    
class CrossEntropy(Loss):
    """
    Abstracts a cross entropy loss
    """
    def __init__(self):
        super(CrossEntropy,self).__init__()
        
    def fprop(self,z,t,batch_size):
        """
        Forward propagation

        Parameters
        ----------
        z : np.ndarray of shape (N_batch,N_class)
            Output of Softmax.
        t : np.ndarray of shape (N-batch,N_class)
            Actual Labels.
        batch_size : int
            Batch Size.

        Returns
        -------
        float
            Cross Entropy Loss.

        """
        self.z=z 
        self.t=t
        return -np.sum(np.log(z)*t)/batch_size
    
    def bprop(self,batch_size):
        """
        Back Propagation

        Parameters
        ----------
        batch_size : int
            Batch Size.

        Returns
        -------
        np.ndarray of shape (N_batch,N_class)
            Gradient w.r.t output of Softmax.

        """
        return -self.t/(self.z*batch_size)
    
class Tanh(UntrainableModule):
    def __init__(self):
        """
        
        Abstracts a tanh layer
        Parameters
        ----------
        N_class : int
            Number of categories.

        Returns
        -------
        None.

        """
        super(Tanh,self).__init__()
    
    def fprop(self,x):
        """
        
        Forward Propagation
        Parameters
        ----------
        x : np.ndarray of shape (N_batch,N_class)
            Input.

        Returns
        -------
        np.ndarray of shape of shape (N_batch,N_class)
            Output.

        """
        exp=np.exp(x)
        tanh=(exp-(1/exp))/(exp+(1/exp))
        
        self.Jac=1-np.power(tanh,2)
        return tanh
    
    def bprop(self,dE):
        """
        
        Back Propagation
        Parameters
        ----------
        dE : np.ndarray of shape (N_batch,N_class)
            Gradient w.r.t output.

        Returns
        -------
        np.ndarray of shape (N_batch,N_class)
            Gradient w.r.t input.

        """
        return self.Jac * dE
    
    
class LogSoftMax(UntrainableModule):
    def __init__(self):
        """
        
        Abstracts a softmax layer
        Parameters
        ----------
        N_class : int
            Number of categories.

        Returns
        -------
        None.

        """
        super(LogSoftMax,self).__init__()
    
    def fprop(self,x):
        """
        
        Forward Propagation
        Parameters
        ----------
        x : np.ndarray of shape (N_batch,N_class)
            Input.

        Returns
        -------
        np.ndarray of shape of shape (N_batch,N_class)
            Output.

        """
        x=x-np.max(x,axis=1,keepdims=True)
        exp=np.exp(x)
        softmax=(exp/exp.sum(axis=1,keepdims=True))        
        self.Jac=np.repeat(-softmax[:,np.newaxis,:],x.shape[1],axis=1)
        self.Jac[:,np.arange(x.shape[1]),np.arange(x.shape[1])]+=1      
        return np.log(softmax)
    
    def bprop(self,dE):
        """
        
        Back Propagation
        Parameters
        ----------
        dE : np.ndarray of shape (N_batch,N_class)
            Gradient w.r.t output.

        Returns
        -------
        np.ndarray of shape (N_batch,N_class)
            Gradient w.r.t input.

        """
        return (dE[:,np.newaxis,:] @ self.Jac).reshape(dE.shape)

class CrossEntropyLog(Loss):
    """
    Abstracts a cross entropy log loss
    """
    def __init__(self):
        super(CrossEntropyLog,self).__init__()
    
    def fprop(self,z,t,batch_size):
        """
        Forward propagation

        Parameters
        ----------
        z : np.ndarray of shape (N-batch,N_class)
            Output of Softmax.
        t : np.ndarray of shape (N-batch,N_class)
            Actual Labels.
        batch_size : int
            Batch Size.

        Returns
        -------
        float
            Cross Entropy Loss.

        """
        self.t=t
        return -np.sum(z*t)/batch_size
    
    def bprop(self,batch_size):
        """
        Back Propagation

        Parameters
        ----------
        batch_size : int
            Batch Size.

        Returns
        -------
        np.ndarray of shape (N-batch,N_class)
            Gradient w.r.t output of Softmax.

        """
        return -self.t/batch_size
