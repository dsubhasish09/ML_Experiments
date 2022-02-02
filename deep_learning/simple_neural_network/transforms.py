# -*- coding: utf-8 -*-
"""
Created on Wed Feb  2 20:52:31 2022

@author: devdutt_subhasish

This file implements different transformations which can be applied on data
"""
import numpy as np

def one_hot(T,N_class=10):
    N=T.shape[0]
    T_hot=np.zeros((N,N_class))
    T_hot[np.arange(N),T]=1
    return T_hot