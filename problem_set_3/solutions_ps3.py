#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 18 20:09:11 2020

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
###### QUESTION 1 #######
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
N       = 25       # number of grid points

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

#### Item e.

# Defining parameters
delta = 0.08
beta = 0.96
alpha = 0.4
gamma = 0.75
sigma = 2
# Intializing class
class Aiyagari(object):
    
    def __init__(self,gridz,Pi,delta,beta,alpha,gamma,sigma):
        
        self.gridz,self.Pi,self.delta,self.beta,self.alpha,self.gamma,self.sigma =gridz, Pi,delta,beta,alpha,gamma,sigma
        self.Pi[self.Pi<1e-10] = 0
        self.grida = np.linspace(start=0,stop=90,num=91)
        
    def aggregate_K(self):
        self.K = (self.lamb_reshape @ np.ones(len(self.gridz))) @ self.grida
        
    def aggregate_N(self):
        l_choice = np.zeros((len(self.grida),len(self.gridz)))
        for a1 in range(len(self.grida)):
            for z1 in range(len(self.gridz)):
                l_choice[a1,z1] = self.L[a1,z1,self.policy[a1][z1]]
                
        self.N = sum(sum(l_choice*self.lamb_reshape))
        
    def excess_demand(self):
        return self.k - (self.K/self.N)
    
    # Capital per capita, comes from the FOC of the firm. k = K/N
    def calculate_k_ratio(self):
        self.k = ((self.r+self.delta)/self.alpha)**(1/(self.alpha -1)) 
        
    def calculate_wage(self):
        self.w = (1-self.alpha)*self.k**self.alpha
        
    def calculate_L_grid(self):
        self.L = np.zeros((len(self.grida),len(self.gridz),len(self.grida)))
        for a1 in range(len(self.grida)):
            for z1 in range(len(self.gridz)):
                for a2 in range(len(self.grida)):
                    self.L[a1,z1,a2] = ((self.w*np.exp(self.gridz[z1])/self.gamma)**(1/self.sigma)+a2-(1+self.r)*self.grida[a1])/(self.w*np.exp(self.gridz[z1])+(self.w*np.exp(self.gridz[z1])/self.gamma)**(1/self.sigma))
                    if self.L[a1,z1,a2] < 0:
                        self.L[a1,z1,a2] = 0
    
    def calculate_C_grid(self):
        self.C = np.zeros((len(self.grida),len(self.gridz),len(self.grida)))
        for a1 in range(len(self.grida)):
            for z1 in range(len(self.gridz)):
                for a2 in range(len(self.grida)):
                    self.C[a1,z1,a2] = (1+self.r)*self.grida[a1] + self.w*np.exp(self.gridz[z1])*self.L[a1,z1,a2] - self.grida[a2]
    
    
    def calculate_U_grid(self):
        self.U = np.zeros((len(self.grida),len(self.gridz),len(self.grida)))
        for a1 in range(len(self.grida)):
            for z1 in range(len(self.gridz)):
                for a2 in range(len(self.grida)):
                    self.U[a1,z1,a2] = (self.C[a1,z1,a2]**(1-self.sigma)/(1-self.sigma)+self.gamma*(1-self.L[a1,z1,a2])**(1-self.sigma)/(1-self.sigma))
                    if self.L[a1,z1,a2]>1 or self.C[a1,z1,a2]<0:
                        self.U[a1,z1,a2] = -999999999
    
    def value_function_operator(self):
        
        # Empty list to identfy maximum values and limits
        Aux = np.zeros((len(self.grida),len(self.gridz),len(self.grida)))  
        Aux2 = np.zeros((len(self.grida),len(self.gridz),len(self.grida))) #borrowing limit
        # Borrowing limit
        a_lim = -self.w*np.exp(self.gridz)/self.r * (-1)**(self.r<0)
        
        # Filling Aux/Aux2 list
        for z1 in range(len(self.gridz)):
            for a2 in range(len(self.grida)):
                E =0
                for z2 in range(len(self.gridz)):
                    E = E + self.Pi[z1,z2]*self.V[a2,z2]
                for a1 in range(len(self.grida)):
                    Aux[a1,z1,a2] = self.U[a1,z1,a2]+self.beta*E
                    if self.U[a1,z1,a2] == -999999999:
                        Aux[a1,z1,a2] = -99999999
                    if self.grida[a2]<a_lim[z1]:
                        Aux2[a2,z1,a2] = -1e4
        
        policy = [[0] * len(self.gridz) for i in range(len(self.grida))]
        for a1 in range(len(self.grida)):
            for z1 in range(len(self.gridz)):
                # Check a2 feasibility
                if np.size(np.argwhere(Aux[a1,z1,:] == -99999999)) == 0:
                    nonfeasible = len(self.grida) + 1
                else:
                    nonfeasible = min(np.argwhere(Aux[a1,z1,:] == -99999999))[0]
                # Check the existence of not borrowable
                if np.size(np.argwhere(Aux2[a1,z1,:] == -1e4)) == 0:
                    not_borrowable = -1
                else:
                    not_borrowable = max(np.argwhere(Aux2[a1,z1,:] == -1e4))[0]
                    
                if nonfeasible <= not_borrowable + 1:
                    if nonfeasible == 0:
                        policy[a1][z1] = 0
                    else:
                        policy[a1][z1] = nonfeasible = 0 ###
                else:
                    policy[a1][z1] = np.argwhere(Aux[a1,z1,:] == np.amax(Aux[a1,z1,:]))[0][0]
                        
        # Maximum Aux for each pair a1,z1
        V_new = np.zeros((len(self.grida),len(self.gridz)))
        for a1 in range(len(self.grida)):
            for z1 in range(len(self.gridz)):
                V_new[a1,z1] = Aux[a1,z1,policy[a1][z1]]
        
        # Policy image
        a_choice = [[0] * len(self.gridz) for i in range(len(self.grida))]
        for a1 in range(len(self.grida)):
            for z1 in range(len(self.gridz)):
                a_choice[a1][z1] = self.grida[policy[a1][z1]]
        
        return V_new, policy, a_choice              
                  
                        
    def value_function_iteration(self,maxit=100000,tol=1e-10,printskip=10,showlog=True,howard=False,howard_step=20,howard_init=5):
        
        start_time = time.time() #Cronometer  
        i=0 #Initializing   
        error = tol+1 #Error
        
        # Loop for iteration
        while i<maxit and error>tol:
            V_new,policy,choice = self.value_function_operator()
            
            self.policy = policy  
            self.choice = choice
            # using howard improvement
            if howard and i >= howard_init:
                for count in range(0,howard_step):
                    V_old = V_new
                    for a1 in range(len(self.grida)):
                        for z1 in range(len(self.gridz)):
                            E = 0
                            for z2 in range(len(self.gridz)):
                                E = E + V_old[policy[a1][z1],z2]*self.Pi[z1,z2]
                            V_new[a1,z1] = self.U[a1,z1,policy[a1][z1]] + self.beta*E
                        
            error = np.amax(np.abs(V_new-self.V))
            i = i+1
            if showlog and i % printskip == 0:
                print(f"\n Error at iteration {i} of VFI is {error}")
                hour = round(((time.time() - start_time)/60//60))
                minute = round(((time.time() - start_time)//60 - hour*60))
                second = round(((time.time() - start_time)- hour*60*60 - minute*60))
                print(f"\Time elapsed: {hour}h {minute}min {second}s")
            self.V = V_new
            
        if i == maxit: 
                print(f"Maximum of {maxit} iterations reached")
                
        if error < tol:
                print(f"converged in {i} iterations")
                
    def stationary_distribution(self):
        
        self.transition = np.zeros((len(self.grida),len(self.grida),len(self.gridz),len(self.gridz)))
        
        for a1 in range(len(self.grida)):
            for z1 in range(len(self.gridz)):
                for a2 in range(len(self.grida)):
                    for z2 in range(len(self.gridz)):
                        self.transition[a1,a2,z1,z2] = (self.policy[a1][z1]==a2)*self.Pi[z1,z2]
            
        # Making a matrix of matrix. From 4-D to 2-D array 
        self.transition = np.swapaxes(self.transition,1,2).reshape(len(self.grida)*len(self.gridz),len(self.grida)*len(self.gridz))           
        # Solving (Tr' - I) X Lam' = 0
        self.eigval, self.eigvec = np.linalg.eig(np.transpose(self.transition))
        self.lamb = self.eigvec[:,np.argwhere(np.round(self.eigval,decimals = 4) == 1)[0]].real
        # Reshape solution and normalize for a fraction of the population
        self.lamb_reshape = self.lamb.reshape(len(self.grida),len(self.gridz))/(sum(self.lamb)[0])  
        
        
        
    def solve(self,tol = 1e-4,maxit = 1e+5,reportdebug = False):
        
        start_time2 = time.time() # Cronometer
        i=0 # Iteraction
        error = tol+1 
        # r bounds
        self.lower_r = -self.delta
        self.upper_r = (1/self.beta) -1
        if reportdebug: print("initial r calculated sucessfuly \n")
        
        # Declare VF
        self.V = np.zeros((len(self.grida),len(self.gridz)))
        
        while i<maxit and error>tol:
            i = i+1
            print(f"Starting iteration {i} \n")
            self.r = (self.lower_r + self.upper_r)/2
            self.calculate_k_ratio()
            self.calculate_wage()
            self.calculate_L_grid()      
            self.calculate_C_grid() 
            self.calculate_U_grid()
            if reportdebug: print("iteration k ratio, wage, l, C, U calculated \n")
            self.value_function_iteration(howard=True, tol=1e-8, howard_init=5)
            if reportdebug: print(" VFI calculated \n")
            if reportdebug: print(" Proceeding to calculate stationary distribution\n")
            self.stationary_distribution()
            if reportdebug: print(" Stationary distribution calculated \n")
            self.aggregate_K()
            self.aggregate_N()
            D = self.excess_demand()
            if reportdebug: print(" K,N and Excess demand calculated \n")
            error = abs(D)
            
            if error < tol:
                print(f"Complete algorithm converged in {i} iterations")
                hour = round(((time.time() - start_time2)/60//60))
                minute = round(((time.time() - start_time2)//60 - hour*60))
                second = round(((time.time() - start_time2)- hour*60*60 - minute*60))
                print(f"\Time elapsed: {hour}h {minute}min {second}s")
                print(f"\n r found: {np.round(self.r,decimals = 5)} \n\n")
            else:
                print(f"\ Error at iteration {i} of r iteration is {D}")
                hour = round(((time.time() - start_time2)/60//60))
                minute = round(((time.time() - start_time2)//60 - hour*60))
                second = round(((time.time() - start_time2)- hour*60*60 - minute*60))
                print(f"\n Time elapsed (total): {hour}h {minute}min {second}s")
                print(f"\n r used in this iteration: {self.r} \n")            
            # Verify excess demand
            if D>0:
                self.lower_r = self.r
            if D<0:
                self.upper_r = self.r
        
        if i == maxit:
                print(f"Maximum of {maxit} iterations in complete algorithm reached")
                
    def plot_policy_a(self):
        # Policy for a'
        # Rearranging values for plotting
        plot_a, plot_z = np.meshgrid(self.grida, self.gridz)
        plot_choice = np.concatenate(self.choice,axis=None).reshape((len(self.grida),len(self.gridz))).transpose()
        fig = plt.figure(figsize=(15,10))
        ax = fig.add_subplot(projection='3d')
        surf = ax.plot_surface(Y = plot_a, X = plot_z, Z = plot_choice, cmap=cm.Greys, linewidth=0,antialiased=False)
        ax.set_xlabel('ln(z)')
        ax.set_ylabel('a')
        ax.set_zlabel('a''\'')
        ax.text2D(0.05, 0.95,"Policy for a''",transform=ax.transAxes)
        fig.colorbar(surf, shrink=0.5, aspect=10)
        plt.show()
        
                
    def print_equilibrium(self):
        
        num_ticks = 11
        yticks = np.linspace(0, len(self.grida) - 1, num_ticks, dtype=np.int)
        yticklabels = [self.grida[i] for i in yticks] 
          
        # Plot 1 - Equilibrium Value Function
        fig = plt.figure(figsize = (10,10))
        ax1 = sbn.heatmap(self.V, robust = True, xticklabels = np.round(self.gridz,decimals = 3), yticklabels = yticklabels)
        ax1.set_yticks(yticks)
        ax1.set_ylim(-10,90)  
        plt.ylim(-10,90) 
        plt.ylabel("Asset") 
        plt.xlabel("ln(z)")
        plt.title(f" Aiyagaru Model - Equilibrium Value Function")
        plt.show()
        fig.savefig("Aiyagari Value Function", dpi=300)
        
        # Plot 2 - Equilibrium Density
        fig = plt.figure(figsize = (10,10))
        ax1 = sbn.heatmap(self.lamb_reshape, robust = True, xticklabels = np.round(self.gridz, decimals = 3), yticklabels = yticklabels)
        ax1.set_yticks(yticks)
        ax1.set_ylim(-10,90)  
        plt.ylim(-10,90)
        plt.ylabel("Asset")
        plt.xlabel("ln(z)")
        plt.suptitle(f"Aiyagari Model - Equilibrium Density")
        plt.show()
        fig.savefig(f"Ayiagari Distribution density", dpi=300)
        
    def print_histograms(self):
        
        # Plot 3 - Histograms
        #Declaring Income
        Income = np.zeros((len(Aiy.grida),len(Aiy.gridz)))   
        # Defining capital and labor income
        capital_income = Aiy.r*Aiy.grida
        labor_income = np.zeros(len(self.gridz))
        for z1 in range(len(self.gridz)):
            labor_income[z1] = Aiy.w*np.exp(Aiy.gridz[z1])
            
        for a1 in range(len(self.grida)):
            for z1 in range(len(self.gridz)):
                Income[a1,z1] = capital_income[a1] + labor_income[z1]
        
        ind_income = np.unravel_index(np.argsort(Income, axis=None), Income.shape)
        self.Income_ordered = Income[ind_income]
        self.Density_ordered = self.lamb_reshape[ind_income]
        
        self.Wealth_density = np.sum(self.lamb_reshape,axis = 1)
        self.Wealth_cumulative = np.zeros(len(self.grida))
        for i in range(len(self.grida)):
            if i != len(self.grida)-1: self.Wealth_cumulative[i+1] = self.Wealth_cumulative[i] + self.Wealth_density[i+1] 
        
        self.Density_cumulative = np.zeros(len(self.Density_ordered))
        for i in range(len(self.Density_ordered)):
            if i != len(self.Density_ordered)-1: self.Density_cumulative[i+1] = self.Density_cumulative[i] + self.Density_ordered[i+1]
 
        
        fig, ((ax1,ax2),(ax3,ax4)) = plt.subplots(2,2,figsize=(10,10))
        
        ax1.hist(self.Income_ordered,weights = self.Density_ordered,bins=30)
        ax1.set_xlabel("Income")
        ax1.set_ylabel("Density")
        
        ax2.fill_between(self.Density_cumulative, self.Income_ordered, where = (self.Density_cumulative<=1))
        ax2.fill_between(self.Density_cumulative, self.Income_ordered, where = (self.Density_cumulative<=0.99))
        ax2.fill_between(self.Density_cumulative, self.Income_ordered, where = (self.Density_cumulative<=0.95))
        ax2.fill_between(self.Density_cumulative, self.Income_ordered, where = (self.Density_cumulative<=0.90))
        ax2.fill_between(self.Density_cumulative, self.Income_ordered, where = (self.Density_cumulative<=0.50))
        ax2.fill_between(self.Density_cumulative, self.Income_ordered, where = (self.Density_cumulative<=0.10))
        ax2.fill_between(self.Density_cumulative, self.Income_ordered, where = (self.Density_cumulative<=0.05))
        ax2.fill_between(self.Density_cumulative, self.Income_ordered, where = (self.Density_cumulative<=0.01))
        ax2.set_ylabel("Agent's income")
        ax2.set_xlabel("Percentage of population with lower income than the agent's")
        ax2.legend(loc = 2)
        
        ax3.hist(Aiy.grida, weights = self.Wealth_density, bins =30)
        ax3.set_xlabel("Wealth")
        ax3.set_ylabel("Density")
        
        ax4.fill_between(self.Wealth_cumulative, self.grida, where = (self.Wealth_cumulative<=1))
        ax4.fill_between(self.Wealth_cumulative, self.grida, where = (self.Wealth_cumulative<=0.99))
        ax4.fill_between(self.Wealth_cumulative, self.grida, where = (self.Wealth_cumulative<=0.95))
        ax4.fill_between(self.Wealth_cumulative, self.grida, where = (self.Wealth_cumulative<=0.90))
        ax4.fill_between(self.Wealth_cumulative, self.grida, where = (self.Wealth_cumulative<=0.5))
        ax4.fill_between(self.Wealth_cumulative, self.grida, where = (self.Wealth_cumulative<=0.1))
        ax4.fill_between(self.Wealth_cumulative, self.grida, where = (self.Wealth_cumulative<=0.05))
        ax4.fill_between(self.Wealth_cumulative, self.grida, where = (self.Wealth_cumulative<=0.01))
        ax4.set_ylabel("Agent's wealth")
        ax4.set_xlabel("Percentage of population with lower wealth than the agent's")
        ax4.legend(loc=2)
        
        fig.suptitle(f"Aiyagari Model - Histograms")  
        fig.savefig("Aiyagari Histograms", dpi=300)
    
    def print_quantiles(self):
        # Filling Wealth
        wealth01 = self.grida[np.argwhere(self.Wealth_cumulative>=0.01)[0][0]]
        wealth05 = self.grida[np.argwhere(self.Wealth_cumulative>=0.05)[0][0]]
        wealth10 = self.grida[np.argwhere(self.Wealth_cumulative>=0.10)[0][0]]
        wealth50 = self.grida[np.argwhere(self.Wealth_cumulative>=0.50)[0][0]]
        wealth90 = self.grida[np.argwhere(self.Wealth_cumulative>=0.90)[0][0]]
        wealth95 = self.grida[np.argwhere(self.Wealth_cumulative>=0.95)[0][0]]
        wealth99 = self.grida[np.argwhere(self.Wealth_cumulative>=0.99)[0][0]]
        # Filling Income
        income01 = np.round(self.Income_ordered[np.argwhere(self.Density_cumulative>=0.01)[0][0]], decimals=3)
        income05 = np.round(self.Income_ordered[np.argwhere(self.Density_cumulative>=0.05)[0][0]], decimals=3)
        income10 = np.round(self.Income_ordered[np.argwhere(self.Density_cumulative>=0.10)[0][0]], decimals=3)
        income50 = np.round(self.Income_ordered[np.argwhere(self.Density_cumulative>=0.50)[0][0]], decimals=3)
        income90 = np.round(self.Income_ordered[np.argwhere(self.Density_cumulative>=0.90)[0][0]], decimals=3)
        income95 = np.round(self.Income_ordered[np.argwhere(self.Density_cumulative>=0.95)[0][0]], decimals=3)
        income99 = np.round(self.Income_ordered[np.argwhere(self.Density_cumulative>=0.99)[0][0]], decimals=3)
    
        print(f"Reporting Wealth Quantiles: \n \n \
          Quantiles     Wealth   \n \
              1%      {wealth01} \n \
              5%      {wealth05} \n \
              10%     {wealth10} \n \
              50%     {wealth50} \n \
              90%     {wealth90} \n \
              95%     {wealth95} \n \
              99%     {wealth99} \n \ ")
        
        print(f"Reporting Income Quantiles: \n \n \
          Quantiles     Income   \n \
              1%      {income01} \n \
              5%      {income05} \n \
              10%     {income10} \n \
              50%     {income50} \n \
              90%     {income90} \n \
              95%     {income95} \n \
              99%     {income99} \n \ ")
              
              

# Solving the problem    

Aiy = Aiyagari(gridz, Pi, delta, beta, alpha, gamma, sigma) 
Aiy.solve(reportdebug=True)
Aiy.print_equilibrium()
Aiy.print_histograms()
Aiy.plot_policy_a
Aiy.print_quantiles()


# Policy for c and l
c_choice = np.zeros((len(Aiy.grida),len(Aiy.gridz)))
c_choice = [[0] * len(Aiy.gridz) for i in range(len(Aiy.grida))]
for a1 in range(len(Aiy.grida)):
    for z1 in range(len(Aiy.gridz)):
        c_choice[a1][z1] = Aiy.C[a1,z1,Aiy.policy[a1][z1]]
                 
        
l_choice = np.zeros((len(Aiy.grida),len(Aiy.gridz)))
l_choice = [[0] * len(Aiy.gridz) for i in range(len(Aiy.grida))]
for a1 in range(len(Aiy.grida)):
    for z1 in range(len(Aiy.gridz)):         
        l_choice[a1][z1] = Aiy.L[a1,z1,Aiy.policy[a1][z1]]
  
 

#plot policy c
#plot_a, plot_z = np.meshgrid(Aiy.grida, Aiy.gridz)
#plot_c_choice = np.concatenate(c_choice,axis=None).reshape((len(Aiy.grida),len(Aiy.gridz))).transpose()
#fig = plt.figure(figsize=(15,10))
#ax = fig.add_subplot(projection='3d')
#surf = ax.plot_surface(Y = plot_a, X = plot_z, Z = plot_c_choice, cmap=cm.Greys, linewidth=0,antialiased=False)
#ax.set_zlim3d(0, 2)
#ax.set_xlabel('ln(z)')
#ax.set_ylabel('a')
#ax.set_zlabel('c\'')
#ax.text2D(0.05, 0.95,"Policy for C''",transform=ax.transAxes)
#fig.colorbar(surf, shrink=0.5, aspect=10)
        

#plot policy for L
#plot_a, plot_z = np.meshgrid(Aiy.grida, Aiy.gridz)
#plot_l_choice = np.concatenate(l_choice,axis=None).reshape((len(Aiy.grida),len(Aiy.gridz))).transpose()
#fig = plt.figure(figsize=(15,10))
#ax = fig.add_subplot(projection='3d')
#surf = ax.plot_surface(Y = plot_a, X = plot_z, Z = plot_l_choice, cmap=cm.Greys, linewidth=0,antialiased=False)
#ax.set_zlim3d(0, 2)
#ax.set_xlabel('ln(z)')
#ax.set_ylabel('a')
#ax.set_zlabel('L\'')
#ax.text2D(0.05, 0.95,"Policy for a''",transform=ax.transAxes)
#fig.colorbar(surf, shrink=0.5, aspect=10)



##### Item f.


Y = Aiy.K**Aiy.alpha * Aiy.N**(1-Aiy.alpha)
#Print results
print(f"Interest rate: {np.round(Aiy.r, decimals = 5)}")
print(f"Capital-to-Labor ratio:{np.round(Aiy.k, decimals = 4)}")
print(f"Output: {np.round(Y, decimals = 3)}")
print(f"Capital-to-Output ratio: {np.round(Aiy.K/Y, decimals = 4)}")


############################ 
######## QUESTION 2 ########
############################ 
  
    

# parameters and number of realizations    
r       = 4       # scale
rho     = 0.9    # persistence 
sigma   = 0.2
mu_AR   = 1.4      # mean  
N       = 33      # number of grid points

Mkv = Markov(rho, sigma, mu_AR, r, N) # generating the markov chain process

s_vals, b = Mkv.get_grid()  # getting grid and boundaries
Pi = Mkv.get_transition_matrix(s_vals, b) # getting transition matrix

@dataclass
class Primitives:
    # Model primitives
    beta: float  # discount rate
    D: float  # demand parameter
    alpha: float  # production function curvature
    cf: float  # production fixed cost period
    ce: float  # entry cost
    s_size: int  # grid size of productivity
    G: Distribution  # distribution of entrants' productivity
    s_min: float = None  # lower boundary of productivity
    s_max: float = None  # up boundary of productivity

    def __post_init__(self):
        self.s_vals = np.exp(s_vals)
        self.F = Pi
        self.discrete_nu()
    #Calculating probability mass 
    def discrete_nu(self):
        pdf_entry = self.G.pdf(self.s_vals)
        self.nu = pdf_entry / np.sum(pdf_entry)
        
@dataclass
class Result():
    p: float # stationary price
    v: Array # incumbent value
    x: float # exit threshold
    X: Array # exit choice 
    mu: Array # stationary distribution over firm size
    M: float # entrant mass
    s_vals: Array # state grid
    n_vals: Array # employment grid
    f_vals: Array # output grid
    Q_d: float # output demand
    
    def __post_init__(self):
        self.total_employment = self.n_vals @ self.mu 
        self.total_firm_mass = sum(self.mu)
        self.average_firm_size = self.total_employment / self.total_firm_mass
        self.entry_rate = self.M / self.total_firm_mass
        self.exit_rate = self.X@self.mu / self.total_firm_mass
        
        
@dataclass
class Hopenhayn(Primitives):

    tol: float = 1e-7
    max_iter: int = 1e4
    verbose: bool = True
    print_skip: int = 1e3

    def __post_init__(self):
        super().__post_init__()

    def employment_func(self, s, p):
        n = (self.alpha * p * s) ** (1 / (1 - self.alpha))
        return n

    def production_func(self, s, p):
        n = self.employment_func(s, p)
        f = s * (n ** self.alpha)
        return f

    def profit_func(self, p, s):
        n = self.employment_func(s, p)
        f = self.production_func(s, p)
        pi = p * f - n - self.cf
        return pi

    def state_transition_func(self, s, error):
        log_s_new = self.a + self.rho * np.log(s) + error
        s_new = np.exp(log_s_new)
        return s_new

    def T_value_operator(self, v, p):
        v_new = np.empty_like(v)

        integral = self.F @ v
        v_new = self.profit_func(
            p, self.s_vals
        ) + self.beta * integral.clip(min=0)

        return v_new 

    def value_func_iteration(self, p):

        
        v = np.ones(self.s_size) # Initialize 
        i = 0 # Set up loop
        error = self.tol + 1 #Error

        while i < self.max_iter and error > self.tol:
            v_new = self.T_value_operator(v, p)
            error = np.max(np.abs(v - v_new))
            i += 1
            if self.verbose and i % self.print_skip == 0:
                print(f"Error at iteration {i} is {error}.")
            v = v_new

        if i == self.max_iter:
            print("Failed to converge value!")


        return v  

    def entry_clearing(self, p, v_entry_func):
        v = self.value_func_iteration(p)
        v_entry = v_entry_func(v)
        error = v_entry - self.ce
        return error

    def solve_price(self,):
        v_entry_func = lambda v: v @ self.nu

        res = sp.optimize.root(self.entry_clearing, 1, args=(v_entry_func))
        if not res.success:
            print("Failed to converge price!")
        p = res.x[0]
        v = self.value_func_iteration(p)
        return p, v

    def decision_func(self, v):
        v_next = self.F @ v
        x_index = np.searchsorted(v_next, 0)
        x = self.s_vals[x_index]
        X = np.where(self.s_vals < x, 1, 0)
        return x, X

    def solve_mu(self, M, X):
        P_x = (self.F * (1 - X).reshape(self.s_size, 1)).T
        I = np.eye(self.s_size)
        temp = sp.linalg.inv(I - P_x)
        mu = M * (temp @ self.nu)
        return mu

    def output_market_clearing(self, M, mu_M1, f_vals, Q_d):
        mu = M * mu_M1
        Q_s = mu @ f_vals
        error = Q_s - Q_d
        return error
    
    def labor_market_clearing(self, M, mu_M1, n_vals, L_s):
        mu = M * mu_M1
        L_d = mu @ n_vals
        error = L_s - L_d
        return error

    def solve_M(self, X, f_vals, Q_d):
        mu_M1 = self.solve_mu(1, X)

        res = sp.optimize.root(
            self.output_market_clearing, 1, args=(mu_M1, f_vals, Q_d)
        )
        if not res.success:
            print("Failed to converge entrant mass!")
        M = res.x[0]

        mu = M * mu_M1
        return M, mu

    def solve_model(self):
        super().__post_init__()
        p, v = self.solve_price()
        x, X = self.decision_func(v)

        s_vals = self.s_vals
        n_vals = self.employment_func(s_vals, p)
        f_vals = self.production_func(s_vals, p)
        Q_d = self.D / p

        M, mu = self.solve_M(X, f_vals, Q_d)

        return Result(
            p=p,
            v=v,
            x=x,
            X=X,
            mu=mu,
            M=M,
            s_vals=s_vals,
            n_vals=n_vals,
            f_vals=f_vals,
            Q_d=Q_d,
        )
    
# use calibrated parameters to form and calculate
test = Hopenhayn(
    beta = 0.8,
    D = 200, #Demand parameter
    alpha = 2/3,
    cf = 20,  # fixed operating cost
    ce = 40,  # sunk entry cost
    s_size = 33,
    #a = 1.4,
    #rho = 0.9,
    #sigma = 0.2, 
    G = lognorm(0.6,3.5),
)

res = test.solve_model()

fig, ax = plt.subplots(figsize=(7,5))
ax.plot(res.n_vals, res.mu/res.total_firm_mass, alpha=0.7, label='firm mass share');
ax.plot(res.n_vals, res.mu*res.n_vals/res.total_employment, alpha=0.7, label='employment share')
ax.set_xlabel('$n$')
ax.set_ylabel('$pmf$')
ax.legend();

res.p
res.x
res.average_firm_size
res.entry_rate
res.exit_rate
    
    
    

##################
### QUESTION 3 ###
##################  
    

# Set parameters
gamma = 2
rho = 0.04
alpha = 0.33
delta = 0.06

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
    start = time()
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
    end = time()
    
    print('It took %1.2f seconds' % (end-start))
    return [V,c]

w1 = 1.1
r1 = 0.05
[V1,c1] = solve_households(w=w1, r=r1)
#
import matplotlib.pyplot as plt

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

start = time()

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
end = time()
print('It took %1.2f seconds' % (end-start))

fig, ax = plt.subplots(figsize = (8, 4))
ax.plot(dist)
ax.set_ylabel(r'$||V^{n+1} - V^n||$')
plt.show()

# Solving Ramsey problem
def solve_ramsey(A):

    dist = []  
    start = time()
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
    end = time() 
    print('It took %1.2f seconds' % (end-start))
    
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

[dist3,v3,c3] = solve_ramsey(A=1.2)

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


    
    
    