#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  8 14:52:49 2020

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


#########################
###### QUESTION 2 #######
#########################

#### Item (a)

class Markov(object):
    
    def __init__(self,rho,sigma,mu,r,N):
        self.rho,self.sigma,self.mu,self.r,self.N = rho,sigma,mu,r,N
            
    def get_grid(self):
        sigma_z = self.sigma/np.sqrt(1-self.rho**2)
        # creating grid vector for z
        z = np.zeros(self.N) 
        # calculating boundaries
        z[0] = self.mu - self.r*sigma_z 
        z[N-1] = self.mu + self.r*sigma_z
        # calculating the distance
        d = (z[N-1]-z[0])/(self.N-1)  
        # intermediary values
        for i in range(1,self.N): 
            z[i] = z[i-1] + d
        # creating grid with borders            
        b = np.zeros(self.N-1) 
        for i in range(0,self.N-1):
            b[i] = z[i] +d/2
            
        return z, b
    
    def get_transition_matrix(self,z,b):
        
        # creating transition matrix Pi
        Pi = np.zeros((self.N,self.N))
        
        # calculating j=1 and j=N cases
        for i in range(0,self.N):
            Pi[i,0] = sp.stats.norm.cdf((b[0]-self.rho*z[i]-self.mu*(1-self.rho))/self.sigma)
            
        for i in range(0,self.N):
            Pi[i,self.N-1] = 1 - sp.stats.norm.cdf((b[self.N-2]-self.rho*z[i]-self.mu*(1-self.rho))/self.sigma)
            
        # calculating intermediary grid cases
        for j in range(1,self.N-1):
            for i in range(0,self.N):
                Pi[i,j] = sp.stats.norm.cdf((b[j]-self.rho*z[i]-self.mu*(1-self.rho))/self.sigma) \
                    - sp.stats.norm.cdf((b[j-1]-self.rho*z[i]-self.mu*(1-self.rho))/self.sigma)         
                    
        return Pi

    # code for simulation
    def markov_sim(self,Pi,z,s0,t): 
        
        rpi,cpi = Pi.shape
        z[:]
        # cum_Pi=[zeros(rpi,1) cumsum(Pi')']
        cum_Pi = np.concatenate((np.zeros((rpi,1)), np.transpose(np.cumsum(np.transpose(Pi), axis=0))), axis=1)
        cum_Pi = np.array(cum_Pi)
        
    
        sim      = np.vstack((np.random.uniform(0,1,t)))
        state    = np.zeros((t,1), dtype=int)
        state[0] = s0 
        
        for k in range(1,t):
            
            logical1 = sim[k]<=cum_Pi[state[k-1],1:cpi+1]
            logical2 = sim[k]>cum_Pi[state[k-1],0:cpi]
            state[k] = np.where(np.logical_and(logical1, logical2))[1]
        
        chain = z[state]
        
        return chain, state
    

# parameters and number of realizations    
r       = 3       # scale
rho     = 0.98    # persistence 
#sigma_z = 0.621 given in the exercice, considering also rho, we can find variância do epsilon
sigmae2 = (1-rho**2)*0.621   # variance
sigma   = np.sqrt(sigmae2)
mu_AR   = 0       # mean
t       = 1000    # number of realizations
N       = 7       # number of grid points

Mkv = Markov(rho, sigma, mu_AR, r, N) # generating the markov chain process

gridz, b = Mkv.get_grid()  # getting grid and boundaries
Pi = Mkv.get_transition_matrix(gridz, b) # getting transition matrix

mychainn = np.zeros([1000,1000])

s0 = 3 #initial state
for j in range(0,t):     
    mychain, mystate = Mkv.markov_sim(Pi, gridz, s0, 1000)    
    mychainn[:,j] = mychain[:,0]
        

plt.plot(mychainn[:,0])
plt.title('N=7')
plt.xlabel("Simulation of a productivity grid")
#plt.ylabel()
plt.show()


#Parameters
#alpha = 0.3
#theta = 0.45
eps   = 2       # CARA parameter
beta  = 0.94    # intertemporal discount factor
w     = 1     # wage
r_l   = 0.01   # interest rate
r_h   = 0.05
amax  = 500     # max of a
sizea = 500     # size of a
#lamb  = 1.5     # values for the financial restriction
psi  = 1        # cost to invest in high return asset


class TwoAssetsEconomy(object):
    
    def __init__(self,eps,beta,w,r_l,r_h,gridz,Pi,amax,sizea,psi):
        self.eps,self.beta,self.w,self.r_l,self.r_h,self.gridz,self.Pi,self.amax,self.sizea,self.psi = \
        eps,beta,w,r_l,r_h,gridz,Pi,amax,sizea,psi
        self.grida = np.linspace(0,self.amax,self.sizea) 
         
    # defining utility function    
    def utility(self,c): 
        u = (c**(1-self.eps))/(1-self.eps)
        return u 
    

    # Consumption considering asset with low return
    def calculate_c_l(self):
        c_l = np.zeros((len(self.grida),len(self.gridz),len(self.grida)))
        for a1 in range(len(self.grida)):
            for z1 in range(len(self.gridz)):
                for a2 in range(len(self.grida)):
                    c_l[a1,z1,a2] = self.w*self.gridz[z1] + (1+self.r_l)*self.grida[a1] - self.grida[a2]  
        
        return c_l
    
    # Consumption considering asset with high return
    def calculate_c_h(self):
        c_h = np.zeros((len(self.grida),len(self.gridz),len(self.grida)))
        for a1 in range(len(self.grida)):
            for z1 in range(len(self.gridz)): 
                for a2 in range(len(self.grida)):
                    c_h[a1,z1,a2] = self.w*self.gridz[z1] + (1+self.r_h)*self.grida[a1] - self.grida[a2] - self.psi
        
        return c_h
    
    
    # Defining in which asset invest acording to the fix cost
    def investment_choice(self,constrained = False,lamb=1): #Policy, 0 ou 1.
        o = np.zeros((len(self.grida),len(self.gridz)))
        for z1 in range(len(self.gridz)):
            for a1 in range(len(self.grida)):
                if self.w*gridz[z1] > self.psi:
                    o[a1,z1] = 1
                else:
                    o[a1,z1] = 0
        
        return o
    
    #Defining the VFO
    def value_function_operator(self,V):
        # V is the guess for the vf (2-D)
        # This operator return the new guess for the value function(V_new), the policy function (returns an index) and a choice
        
        # Auxiliary matrix that consider all possible utilities
        Aux = np.zeros((len(self.grida),len(self.gridz),len(self.grida)), dtype=object)
        
        # List to calculate the new V
        for a1 in range(len(self.grida)):
            for z1 in range(len(self.gridz)): 
                for a2 in range(len(self.grida)):
                  # E measures the expected utility in the next period
                   E = V[a2,z1] + (V[a2,:] @ self.Pi) #vetorial product
                   Aux[a1,z1,a2] = self.u[a1,z1,a2] + self.beta*E
                   
        # Taking the maximum Aux for each pair k1 and z1
        V_new = np.zeros((len(self.grida),len(self.gridz)))
        for z1 in range(len(self.gridz)):
            for a1 in range(len(self.grida)):
                V_new[a1,z1] = np.amax(Aux[a1,z1,:]) #fixing a1 and z1, picks the index for a2 associated with hishet utility
                    
        
        # Taking index of maximum Aux for each pair k1 and z1
        policy = [[0] * len(self.gridz) for i in range(len(self.grida))]
        for a1 in range(len(self.grida)):
            for z1 in range(len(self.gridz)):
                policy[a1][z1] = np.argwhere(Aux[a1,z1,:] == np.amax(Aux[a1,z1,:]))
        
        # Value for the policy 
        choice = [[0] * len(self.gridz) for i in range(len(self.grida))]
        for a1 in range(len(self.grida)):
            for z1 in range(len(self.gridz)):
                 choice[a1][z1] = [self.grida[policy[a1][z1][0][0]]]
                 
        return V_new, policy, choice
    
    
    # Function to determined self variables not yet calculated e correct possible negative values for consumption
    def Calculate_intermediaries(self):
        self.c_l = self.calculate_c_l()
        self.c_h = self.calculate_c_h()
        #self.c_l[self.c_l<0] = 1e-8
        #self.c_h[self.c_h<0] = 1e-8
        self.u_l = self.utility(self.c_l)
        self.u_h = self.utility(self.c_h)
        self.u = np.maximum(self.u_l,self.u_h)
        
    def value_function_iteration(self,maxit=1000000, tol=1e-10,printskip=30,showlog=True, howard = False, howard_step=20):
        
        V = np.zeros((len(self.grida),len(self.gridz)))  #Allocate space
        i = 0                                            #Initializing counter
        error = tol+1                                    #Intializing Error
        start_time = time.time()                         #Cronometer
        
        #Loop for each iteration
        while i < maxit and error > tol:
            V_new,policy,choice = self.value_function_operator(V)
            # using howard
            if howard:
                for count in range(0,howard_step):
                    V_old = V_new
                    for a1 in range(len(self.grida)):   
                        for z1 in range(len(self.gridz)):
                            E = V_old[policy[a1][z1][0],z1][0] + (self.Pi @ V_old[policy[a1][z1][0],:][0])
                            V_new[a1,z1] = self.u[a1,z1,policy[a1][z1][0]][0] + self.beta*E
                    
                            
            error = np.amax(np.abs(V_new-V))
            i=i+1
            V = V_new 
            if showlog and i % printskip == 0:
                print(f"\In Error at iteration {i} is {error}")
                hour   = round(((time.time() - start_time)/60)//60)
                minute = round(((time.time() - start_time)//60 - hour*60))
                second = round(((time.time() - start_time)- hour*60*60 - minute*60))
                print(f"\n Time elapsed: {hour}h {minute}min {second}s")
            
            
        if i == maxit:
            print(f"Maximum of {maxit} iterations reached")
            
        if error < tol:
            print(f"Error lower than tol after {i} iterations")
            
        
        return V, policy, choice
    


econ = TwoAssetsEconomy(eps, beta, w, r_l, r_h, gridz, Pi,  amax, sizea,psi) #Starting economy 
econ.Calculate_intermediaries()  
o1 = econ.investment_choice()    
V,policy,choice = econ.value_function_iteration(tol = 1e-8, howard = True) #Iterating the Value Function, using howard 


#Defining grida outside de class
grida = np.linspace(0,amax,sizea)        
#Rearraging and plotting
plot_a, plot_z = np.meshgrid(grida,gridz)   
plot_choice = np.concatenate(choice,axis=None).reshape((len(grida),len(gridz))).transpose()
fig = plt.figure(figsize=(10,8))
ax = fig.add_subplot(projection = '3d')
surf = ax.plot_surface(Y = plot_a, X = plot_z, Z = plot_choice, cmap=cm.Greys,linewidth=0,antialiased=False)
ax.set_xlabel('$Z_t$')
ax.set_ylabel('$a_t$')
ax.set_zlabel('$a_{t+1}$')
ax.text2D(0.05,0.95,"Policy for Assets",transform=ax.transAxes)
fig.colorbar(surf, shrink=0.5, aspect=10)
plt.show() 

    
    