# -*- coding: utf-8 -*-
"""
Created on Thu Feb  3 21:55:47 2022

@author: devdutt_subhasish
"""
import numpy as np
from math import factorial 

def random_poly(x:np.ndarray,degree:int,low:float=-1.0,high:float=1.0,
                noise:bool=True,noise_frac:float=0.05):
    """
    Computes a random polynomial using the elements of x

    Parameters
    ----------
    x : np.ndarray
        input one dimensional array.
    degree : int
        degree of polynomial.
    low : float, optional
        lowest value for the polynomial coefficients. The default is -1.0.
    high : float, optional
        highest value for the polynomial coefficients. The default is 1.0.
    noise : bool, optional
        Should noise be added. The default is True.
    noise_frac : float, optional
        Adds noise proportional to range of values from computing the polynomial function . The default is 0.05.

    Returns
    -------
    y : np.ndarray
        result of the random polynomial function evaluated using the elements of x.

    """
    coef=np.random.uniform(low,high,size=degree+1)
    y=np.zeros(x.shape)
    for d in range(degree+1):
        y+=coef[d]*x**d
    return y+noise*noise_frac*(y.max()-y.min())*np.random.randn(*x.shape)

def my_poly_features(xs, degree):
    """Generates polynomial features from given data
    
    The polynomial features should include monomials (i.e., x_i, x_i**2 etc)
    and interaction terms (x_1*x_2 etc), but no repetitions.
    The order of the samples should not be changed through the transformation.
    
    Arguments
    xs      2d numpy array of shape (N,D) containing N samples of dimension D
    degree  Maximum degree of polynomials to be considered
    
    Returns
    An (N,M) numpy array containing the transformed input
    """
    # Your implementation
    return np.concatenate([np.ones((xs.shape[0],1)),recurse_poly(xs,degree)[0]],axis=1)#adding bias column

def recurse_poly(arr,degree):
    """
    Create polynomial features upto the given degree using arr.

    Parameters
    ----------
    arr : np.ndarray
        NxD array of input data points
    degree : int
        required degree upto which polynomial features need to be generated

    Returns
    -------
    np.ndarray
        NxM matrix with polynomial features up to required degree.
    np.ndarray
        Nx[DXD...degree times] matrix which is used to pass current computation to 
        higher degree recursion, so as to help with the computation of the next 
        set of polynomial features.
    """
    if degree==1:
        return arr,arr#if degree is one, then easy!
    else:
        N,D=arr.shape
        poly_,cache_=recurse_poly(arr,degree-1)#take features and cache from degree-1
        k=round(factorial(D+degree-1)/(factorial(degree)*factorial(D-1)))#No. of features of this degree
        cache=np.zeros((N,k))#initiate current cache
        #these two are variables used to index into cache_ and cache respectively
        pos_=0
        pos=0
        for i in range(D):#iterate over features
            k=round(factorial(D-i+degree-2)/(factorial(degree-1)*factorial(D-i-1)))# this is used to update pos and pos_
            cache[:,pos:pos+k]=arr[:,i].reshape((N,1))*cache_[:,pos_:]#update cache
            #update pos and pos_
            pos=pos+k
            pos_=round(pos_+k*((degree-1)/(D-i+degree-2)))
        return np.concatenate([poly_,cache],axis=1),cache