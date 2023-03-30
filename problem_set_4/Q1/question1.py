#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 10 15:17:50 2020

@author: muriloandreperespereira
"""



import math as mt
import numpy as np
from dataclasses import dataclass
from typing import Any
from scipy.stats import norm, lognorm, uniform, gaussian_kde
from numba import njit, prange, vectorize, jitclass
import concurrent.futures
import scipy as sp
import numba as nu
import time
from scipy.stats import norm
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d.axes3d import Axes3D
from numba import jit, prange
import seaborn as sbn
Array = Any
Distribution = Any

    
# Set parameters
gamma = 2
rho = 0.05
alpha = 0.35
delta = 0.01

def ss(A):
    r = rho
    a = (alpha*A/(r+delta))**(1/(1-alpha))
    c =r*a
    
    return [r,a,c]

[rss,a_ss,css] = ss(A=1)

amax = 2*a_ss
amin = 0.01*a_ss
N = 400
maxit = 10**4
crit = 10**-6

# Grid for a
a = np.linspace(amin,amax,N)
da = (amax - amin)/(N-1)

dV = np.zeros(N)
dVb = np.zeros(N)
dVf = np.zeros(N)
c = np.zeros(N)
V0 = (((a**alpha)**(1-gamma))/(1-gamma))/rho  #initial guess for v is ss v
v = V0
# Solving households problem
def solve_households(w,r):
    #start = time()
    dist = crit+1
    v = V0 
    dV = np.zeros(N)
    dVb = np.zeros(N)
    dVf = np.zeros(N)
    c = np.zeros(N)
    
    for i in range(0,maxit):
        V=v
    
        #forward difference
        dVf[0:N-1] = (V[1:N] - V[0:N-1])/da
        dVf[N-1] = 1e-3 # will never be used 
        #backward difference
        dVb[1:N] = (V[1:N]-V[0:N-1])/da
        dVb[0] = 0 # will never be used
        
        #consumption and savings with forward difference
        cf = dVf**(-1/gamma)
        #lf = fsolve(solve1, 0.5*np.ones(N),cf)
        muf = w*np.ones(N) + r*a - cf
        #consumption and savings with backward difference
        cb = dVb**(-1/gamma)
        #lb = fsolve(solve, 0.5*np.ones(N),cb))
        mub = w*np.ones(N) + r*a - cb
        #consumptin and derivative of vf at ss
        c0 = w*np.ones(N) + r*a
        dV0 = c0**(-gamma)
        
        #dV_upwind makes a choice of forward or backward differences base on the sign of the drift
        If = muf > 0 #below ss
        Ib = mub < 0 #above ss
        I0 = (1 - If - Ib) #at ss
        #make sure the right approximations are used at the boundaries
        Ib[0]=0
        If[0]= 1
        Ib[N-1]=1
        If[N-1]=0
        dV_Upwind = dVf*If + dVb*Ib + dV0*I0 
    
        #dv = D@V
        c = dV_Upwind**(-1/gamma)
    
        Vchange = c**(1-gamma)/(1-gamma) + dV_Upwind*(w*np.ones(N) + r*a -c) - rho*V
    
        ## This is the update
        # the following CFL condition seems to work well in practice
        Delta = .9*da/max(w + r*a)
        v = v+Delta*Vchange
    
        dist = max(abs(Vchange))
        if dist<crit:
            print('Values Function Coverged, Iteration = %5.0f' % i)
            break
   # end = time()
    
    #print('It took %1.2f seconds' % (end-start))
    return [V,c]

w1 = 1.1
r1 = 0.05
[V1,c1] = solve_households(w=w1, r=r1)
#


fig,ax = plt.subplots(figsize = (8,5))
ax.plot(a,V1)
ax.set_ylabel(r'$v(a)$')
ax.set_xlabel(r'$a$')
ax.grid()
plt.show()

fig,ax = plt.subplots(1,2,figsize = (12,5))
ax[0].plot(a,c1)
ax[0].set_ylabel(r'$c(a)$')
ax[0].set_xlabel(r'$a$')
ax[0].grid()

adot1 = w1*np.ones(N) + r1*a - c
ax[1].plot(a,adot1)
ax[1].plot(a,np.zeros(N))
ax[1].set_ylabel(r'$s(a)$')
ax[1].set_xlabel(r'$a$')
ax[1].grid()
plt.show()


dist = []

#start = time()

for i in range(0,maxit):
    V = v
    
    # forward difference
    dVf[0:N-1] = (V[1:N]-V[0:N-1])/da
    dVf[N-1] = 0 #will never be used
    # backward difference
    dVb[1:N] = (V[1:N]-V[0:N-1])/da
    dVb[0] = 0 #will never be used
    
    I_concave = dVb > dVf  #indicator whether value function is concave (problems arise if this is not the case)
    
    #consumption and savings with forward difference
    cf = dVf**(-1/gamma)
    muf = a**alpha - delta*a - cf
    #consumption and savings with backward difference
    cb = dVb**(-1/gamma)
    mub = a**alpha - delta*a - cb
    #consumption and derivative of value function at steady state
    c0 = a**alpha - delta*a
    dV0 = c0**(-gamma)
    
    # dV_upwind makes a choice of forward or backward differences based on
    # the sign of the drift    
    If = muf > 0 #below steady state
    Ib = mub < 0 #above steady state
    I0 = (1-If-Ib) #at steady state
    #make sure the right approximations are used at the boundaries
    Ib[0] = 0
    If[0] = 1
    Ib[N-1] = 1
    If[N-1] = 0
    dV_Upwind = dVf*If + dVb*Ib + dV0*I0 #important to include third term
    
    c = dV_Upwind**(-1/gamma)
    Vchange = c**(1-gamma)/(1-gamma) + dV_Upwind*(a**alpha - delta*a - c) - rho*V
        
    ## This is the update
    # the following CFL condition seems to work well in practice
    Delta = .9*da/max(a**alpha - delta*a)
    v = v + Delta*Vchange
    
    dist.append( max(abs(Vchange)) )
    if dist[i] < crit:
        print('Value Function Converged, Iteration = %5.0f' % i)
        break
#end = time()
#print('It took %1.2f seconds' % (end-start))

fig, ax = plt.subplots(figsize = (8, 4))
ax.plot(dist)
ax.set_ylabel(r'$||V^{n+1} - V^n||$')
plt.show()

# Solving Ramsey problem
def solve_ramsey(A):

    dist = []  
    #start = time()
    v =V0
    dV = np.zeros(N)
    dVb = np.zeros(N)
    dVf = np.zeros(N)
    c = np.zeros(N)      
    for i in range(0,maxit):
        V = v
    
        # forward difference
        dVf[0:N-1] = (V[1:N]-V[0:N-1])/da
        dVf[N-1] = 0 #will never be used
        # backward difference
        dVb[1:N] = (V[1:N]-V[0:N-1])/da
        dVb[0] = 0 #will never be used
        
        I_concave = dVb > dVf  #indicator whether value function is concave (problems arise if this is not the case)
        
        #consumption and savings with forward difference
        cf = dVf**(-1/gamma)
        muf = a**alpha - delta*a - cf
        #consumption and savings with backward difference
        cb = dVb**(-1/gamma)
        mub = a**alpha - delta*a - cb
        #consumption and derivative of value function at steady state
        c0 = a**alpha - delta*a
        dV0 = c0**(-gamma)
        
        # dV_upwind makes a choice of forward or backward differences based on
        # the sign of the drift    
        If = muf > 0 #below steady state
        Ib = mub < 0 #above steady state
        I0 = (1-If-Ib) #at steady state
        #make sure the right approximations are used at the boundaries
        Ib[0] = 0
        If[0] = 1
        Ib[N-1] = 1
        If[N-1] = 0
        dV_Upwind = dVf*If + dVb*Ib + dV0*I0 #important to include third term
        
        c = dV_Upwind**(-1/gamma)
        Vchange = c**(1-gamma)/(1-gamma) + dV_Upwind*(a**alpha - delta*a - c) - rho*V
            
        ## This is the update
        # the following CFL condition seems to work well in practice
        Delta = .9*da/max(a**alpha - delta*a)
        v = v + Delta*Vchange
        
        dist.append( max(abs(Vchange)) )
        if dist[i] < crit:
            print('Value Function Converged, Iteration = %5.0f' % i)
            break
    #end = time() 
    #print('It took %1.2f seconds' % (end-start))
    
    return [dist,v,c]

[dist1,v1,c2]=solve_ramsey(A=1) 
fig,ax = plt.subplots(figsize = (8,4))
ax.plot(dist1)
ax.set_ylabel(r'$//V^{n+1} - V^n//$')
plt.show()

fig,ax = plt.subplots(figsize = (8,5))
ax.plot(a,v1)
ax.set_ylabel(r'$v(a)$')
ax.set_xlabel(r'$a$')
ax.grid()
plt.show

adot2 = a**alpha - delta*a - c2
fig,ax = plt.subplots(1,2,figsize = (12,5))
ax[0].plot(a,c1)
ax[0].set_ylabel(r'$c(a)$')
ax[0].set_xlabel(r'$a$')
ax[0].grid()          

ax[1].plot(a,adot2)
ax[1].plot(a,np.zeros(N))
ax[1].set_ylabel(r'$c(a)$')
ax[1].set_xlabel(r'$a$')
ax[1].grid()
plt.show()  
              
# Solving for the whole transition

a0 = a[80]
T = 400
T_shock = 150
a_simul = np.zeros(T+1)
a_simul[0] = a0
c_simul = np.zeros(T)
adot_simul = np.zeros(T)

[dist3,v3,c3] = solve_ramsey(A=1.1)

def find_nearest(array,value):
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        return idx

for t1 in range(T_shock):
    ind = np.where(a==a_simul[t1])
    c_simul[t1] = c2[ind]
    adot_simul[t1] = a_simul[t1]**alpha - delta*a_simul[t1] - c_simul[t1]
    aux = a_simul[t1] + adot_simul[t1]
    newind = find_nearest(a,aux)
    a_simul[t1+1]=a[newind]
        
for t2 in range(T_shock,T):
    ind = np.where(a== a_simul[t2])
    c_simul[t2] = c3[ind]
    adot_simul[t2] = 1.2*a_simul[t2]**alpha - delta*a_simul[t2] - c_simul[t2]  
    aux = a_simul[t2] + adot_simul[t2]
    newind = find_nearest(a, aux)
    a_simul[t2+1] = a[newind] 
    
fig,ax = plt.subplots(figsize = (7,5))
ax.plot(c_simul)
ax.set_title(r'$Consumption$')
ax.set_xlabel(r'$Periods$')
ax.grid()
plt.show()


    
    
    