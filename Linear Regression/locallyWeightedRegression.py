# -*- coding: utf-8 -*-
"""
Created on Tue Sep 12 20:39:09 2017

"""
from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
import scipy
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston
np.random.seed(0)

# load boston housing prices dataset
boston = load_boston()
x = boston['data']
N = x.shape[0]
x = np.concatenate((np.ones((506,1)),x),axis=1) #add constant one feature - no bias needed
d = x.shape[1]
y = boston['target']

idx = np.random.permutation(range(N))

#helper function
def l2(A,B):
    '''
    Input: A is a Nxd matrix
           B is a Mxd matirx
    Output: dist is a NxM matrix where dist[i,j] is the square norm between A[i,:] and B[j,:]
    i.e. dist[i,j] = ||A[i,:]-B[j,:]||^2
    '''
    A_norm = (A**2).sum(axis=1).reshape(A.shape[0],1)
    B_norm = (B**2).sum(axis=1).reshape(1,B.shape[0])
    dist = A_norm+B_norm-2*A.dot(B.transpose())
    return dist

 
 
#to implement
def LRLS(test_datum,x_train,y_train, tau,lam=1e-5):
    '''
    Input: test_datum is a dx1 test vector
           x_train is the N_train x d design matrix
           y_train is the N_train x 1 targets vector
           tau is the local reweighting parameter
           lam is the regularization parameter
    output is y_hat the prediction on test_datum
    '''

    A = np.zeros((len(y_train), len(y_train)))
    ## To compute each value of A, use logsum exp with test_datum entries for logsum exp and x_train for exp
    for i in range(0, A.shape[0]):
        x = np.transpose(test_datum)
        nrms = np.linalg.norm(x - x_train, axis=1)
        pos = np.exp(-1*nrms[i])/(2*np.square(tau)) / np.exp(scipy.special.logsumexp(-1*np.sum(nrms)/(2.*np.square(tau))))
        A[i][i] = pos

    sub0 = np.linalg.solve(A, x_train)
    sub1 = np.matmul(np.transpose(sub0),x_train)
    idn = np.identity((x_train.shape[1]))*lam
    sub1 = sub1 + idn
    sub2 = np.linalg.solve(A, x_train)
    sub2 = np.matmul(np.transpose(sub2), y_train)
    #inv = np.linalg.inv(sub1)
    put = np.matmul(sub1, sub2)
    w_star = put

    return np.matmul(np.transpose(test_datum), w_star)





def run_validation(x,y,taus,val_frac):
    '''
    Input: x is the N x d design matrix
           y is the N x 1 targets vector    
           taus is a vector of tau values to evaluate
           val_frac is the fraction of examples to use as validation data
    output is
           a vector of training losses, one for each tau value
           a vector of validation losses, one for each tau value
    '''
    training_losses = []
    test_losses = []
    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=val_frac)
    sumi = 0
    print(x_train.shape[0])
    print(taus.shape[0])
    for t in range(0, taus.shape[0]):
        for i in range(0, x_train.shape[0]):
            sumi = sumi + LRLS(x_train[i],x_train,y_train,taus[t])
        training_losses.append(sumi/len(taus))

    for t in range(0, taus.shape[0]):
        for i in range(0, x_train.shape[0]):
            sumi = sumi + LRLS(x_val[i],x_val,y_val,taus[t])
        training_losses.append(sumi/len(taus))

    return (training_losses,test_losses)



if __name__ == "__main__":
    
    taus = np.logspace(1.0,2,5)
    train_losses, test_losses = run_validation(x,y,taus,val_frac=0.3)
    plt.semilogx(train_losses)
    plt.semilogx(test_losses)

