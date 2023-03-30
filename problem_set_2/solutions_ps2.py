# -*- coding: utf-8 

"""
MACROECONOMICS III - PROBLEM SET 2

Murilo A. P. Pereira

"""



import math as mt
import numpy as np
import scipy as scp
import numba as nu
import time
from scipy.stats import norm
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d.axes3d import Axes3D
from numba import jit, prange
import seaborn as sns




###############################
######### QUESTION 1 ##########

class Markov(object):
    
    def __init__(self,rho,sigma,mu,r,N):
        self.rho,self.sigma,self.mu,self.r,self.N = rho,sigma,mu,r,N
            
    def get_grid(self):
        sigma_z = self.sigma/np.sqrt(1-self.rho**2)
        
        z = np.zeros(self.N)            # creating grid vector for z
        
        z[0] = self.mu - self.r*sigma_z # calculating boundaries
        z[N-1] = self.mu + self.r*sigma_z
         
        d = (z[N-1]-z[0])/(self.N-1)    # calculating the distance 
      
        for i in range(1,self.N):       # intermediary values
            z[i] = z[i-1] + d
                    
        b = np.zeros(self.N-1)          # creating grid with borders
        for i in range(0,self.N-1):
            b[i] = z[i] +d/2
            
        return z, b
    
    def get_transition_matrix(self,z,b):
        
        # creating transition matrix Pi
        Pi = np.zeros((self.N,self.N))
        
        # calculating j=1 and j=N cases
        for i in range(0,self.N):
            Pi[i,0] = scp.stats.norm.cdf((b[0]-self.rho*z[i]-self.mu*(1-self.rho))/self.sigma)
            
        for i in range(0,self.N):
            Pi[i,self.N-1] = 1 - scp.stats.norm.cdf((b[self.N-2]-self.rho*z[i]-self.mu*(1-self.rho))/self.sigma)
            
        # calculating intermediary grid cases
        for j in range(1,self.N-1):
            for i in range(0,self.N):
                Pi[i,j] = scp.stats.norm.cdf((b[j]-self.rho*z[i]-self.mu*(1-self.rho))/self.sigma) \
                    - scp.stats.norm.cdf((b[j-1]-self.rho*z[i]-self.mu*(1-self.rho))/self.sigma)         
                    
        return Pi

    # code for simulation
    def markov_sim(self,Pi,z,s0,t): 
        
        rpi,cpi = Pi.shape
        z[:]
        # cum_Pi=[zeros(rpi,1) cumsum(Pi')']
        cum_Pi = np.concatenate((np.zeros((rpi,1)), np.transpose(np.cumsum(np.transpose(Pi), axis=0))), axis=1)
        cum_Pi = np.array(cum_Pi)
        
        np.random.seed(1234)
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
rho     = 0.8     # persistence 
sigmae2 = 0.01    # variance
sigma   = np.sqrt(sigmae2)
mu_AR   = 0       # mean
t       = 1000    # number of realizations

#### Item b - (i)

N=3 # number of grid points

Mkv = Markov(rho, sigma, mu_AR, r, N) # generating the markov chain process

gridz, b = Mkv.get_grid()  # getting grid and boundaries
Pi = Mkv.get_transition_matrix(gridz, b) # getting transition matrix

mychainn = np.zeros([1000,1000])

s0 = 1 #initial state
for j in range(0,t):     
    mychain, mystate = Mkv.markov_sim(Pi, gridz, s0, 1000)    
    mychainn[:,j] = mychain[:,0]
        

plt.plot(mychainn[:,0])
plt.title('N=3')
plt.xlabel("Simulation of a productivity grid")
#plt.ylabel()
plt.show()

     
#### Item b - (ii)

# changing value of N
N=7 # number of grid points

Mkv = Markov(rho, sigma, mu_AR, r, N) # generating the markov chain process with the new N

gridz, b = Mkv.get_grid() #Getting grid and boundaries
Pi = Mkv.get_transition_matrix(gridz, b) #Getting transition matrix

mychainn = np.zeros([1000,1000])

s0 = 3 # initial state
for j in range(0,t):     
    mychain, mystate = Mkv.markov_sim(Pi, gridz, s0, 1000)    
    mychainn[:,j] = mychain[:,0]

# plot
plt.plot(mychainn[:,0])
plt.title('N=7')
plt.xlabel("Simulation of a productivity grid")
#plt.ylabel()
plt.show()


################################
################################

# changing value of N
N=15 # number of grid points

Mkv = Markov(rho, 0.1, mu_AR, r, N) # generating the markov chain process with the new N

gridz, b = Mkv.get_grid() # getting grid and boundaries
Pi = Mkv.get_transition_matrix(gridz, b) # getting transition matrix

mychainn = np.zeros([1000,1000])

s0 = 7 # initial state
for j in range(0,t):     
    mychain, mystate = Mkv.markov_sim(Pi, gridz, s0, 1000)    
    mychainn[:,j] = mychain[:,0]

# plot
plt.plot(mychain)
plt.title('N=15')
plt.xlabel("Simulation of a productivity grid")
#plt.ylabel()
plt.show()



########################################
############### QUESTION 2 #############

# AR(1) parameters for the markov chain approximation 
rho = 0.95     # persistence of the shock
sigma = 0.007  # s.d of the innovation
mu_AR = 0      # mean
r = 3          # scale
N = 7          # tauchen with 7 grid points

Mkv = Markov(rho, sigma, mu_AR, r, N)    # Generating the markov chain process
gridz, b = Mkv.get_grid()                # Getting grid and boundaries
Pi = Mkv.get_transition_matrix(gridz, b) # Getting transition matrix

hss   = 1/3
zss   = 1
alpha = 1/3
beta  = 0.987
delta = 0.012
mu    = 2
gridk = 1      #Temporary
gamma = 1      #Temporary
h     = 1      #Temporary
c     = 1      #Temporary
u     = 1      #Temporary


class RBC_model(object):
    
    #Economy paramaters
    def __init__(self,hss,alpha,beta,delta,mu,Pi,gridz,gridk,zss,gamma,h,c,u):
        self.alpha, self.beta, self.delta, self.mu, self.Pi, self.gridz, self.hss, self.zss, self.gridk, self.gamma, self.h, self.c, self.u = \
            alpha,beta,delta,mu,Pi,gridz,hss,zss,gridk,gamma,h,c,u
          
    #Generating grid for k
    def create_gridk(self,kmin,kmax,n):
        # kmin: lowest value fo the grid; kmax: highest values of the grid; n: number of elements of the grid
        kss = self.calculate_kss()
        gridk = np.linspace(kmin*kss,kmax*kss,n)
        return gridk
    
        # Steady State for k
    def calculate_kss(self):
        # From Euler equation and Envelope Theorem we can find the kss, as derived in the document
        kss = (self.hss)*((self.zss*(self.alpha)*self.beta)/(1-self.beta*(1-self.delta)))**(1/(1-self.alpha))
        return kss
    
    #Calculating gamma
    def calculate_gamma(self):
        kss = self.calculate_kss()
        #css = (kss**selg.alpha)*(self.hss**(1-self.alpha)) - self.delta*kss
        #From the equilibrium conditions derived we have: 0 = ((1-gamma)/gamma)-css^(-1)*(1-hss)*(1-alpha)*(kss/hss)^alpha
        #Combining and solving for gamma
        gamma = 1/(1+((1-self.hss)/(kss*(self.zss*(self.hss/kss)**(1-self.alpha)-self.delta))*(self.zss*(1-self.alpha)*(self.hss/kss)**(-self.alpha))))
        return gamma
     
    #Calculating h
    def h_function(self,h1,k1,z1,k2):
        f = (self.gamma/(1-self.gamma)*(1-h1)*(np.exp(z1)*(1-self.alpha)*(h1/k1)**(-self.alpha))) - np.exp(z1)*(k1**(self.alpha))*(h1**(1-self.alpha)) - k1*(1-self.delta)+k2
      
        return f
    
    # From k,z grid calculating the 3-D matrix for h
    def calculate_h(self):    
        # Calculating possible h values for each grid point k1, z1, k2. Initial guess = h0
        start_time  = time.time() # time
        h0 = 1e-4 # initial value for fsolve
        h  = np.zeros((len(self.gridk),len(self.gridz),len(self.gridk)))
        for k1 in range(len(self.gridk)):
            for z1 in range(len(self.gridz)):
                for k2 in range(len(self.gridk)):
                    h[k1,z1,k2] = fsolve(self.h_function, h0, args=(gridk[k1],gridz[z1],gridk[k2]))
        elapsed = round(time.time() - start_time) 
        print(f"\ Elapsed time for h: {elapsed}s")
        return h
    
    # Optimal c grid from k,z,h grid. 3-D matrix
    def calculate_c(self):
        c = np.zeros((len(self.gridk),len(self.gridz),len(self.gridk)))
        for k1 in range(len(self.gridk)):
            for z1 in range(len(self.gridz)):
                for k2 in range(len(self.gridk)):
                    c[k1,z1,k2] = np.exp(self.gridz[z1])*self.gridk[k1]**self.alpha*h[k1,z1,k2]**(1-self.alpha) + (1-self.delta)*self.gridk[k1] - self.gridk[k2]
        return c
   
    # Calculating utility. 3-D matrix.
    def calculate_u(self):
        u = np.zeros((len(self.gridk),len(self.gridz),len(self.gridk)))
        for k1 in range(len(self.gridk)):
            for z1 in range(len(self.gridz)):
                for k2 in range(len(self.gridk)):
                    if h[k1,z1,k2] > 1:
                        u[k1,z1,k2] = -99999999
                    else:
                        u[k1,z1,k2] = (self.c[k1,z1,k2]**self.gamma*(1-self.h[k1,z1,k2])**(1-self.gamma))**(1-self.mu)/(1-self.mu)              
        return u
     
    # One iteration of Value Function
    def value_function_operator(self,V):     
        # Declare auxiliary 
        Aux = np.zeros((len(self.gridk),len(self.gridz),len(self.gridk)))
        # Here I construct an auxiliary matrix in order to find the maximum value to construct V2
        for k1 in range(len(self.gridk)):
            for z1 in range(len(self.gridz)):
                for k2 in range(len(self.gridk)):
                    if self.c[k1,z1,k2] > 0 and self.h[k1,z1,k2] < 1 and self.h[k1,z1,k2] >= 0:
                        E = 0
                        for z2 in nu.prange(len(self.gridz)):
                            E = E + self.Pi[z1,z2]*V[k2,z2]
                        Aux[k1,z1,k2] = self.u[k1,z1,k2] + self.beta*E
                    else:
                        Aux[k1,z1,k2] = -9999999999
                        
        # Maximum  value of the Aux matrix for each pair k1 and z1
        V2 = np.zeros((len(self.gridk),len(self.gridz)))
        for k1 in range(len(self.gridk)):
            for z1 in range(len(self.gridz)):
                V2[k1,z1] = np.amax(Aux[k1,z1,:])
                
        # Taking the index for the maximum value of each pair k1 and z1 to construct the policy function
        policy = [[0] * len(self.gridz) for i in range(len(self.gridk))]
        for k1 in range(len(self.gridk)):
            for z1 in range(len(self.gridz)):
                policy[k1][z1] = np.argwhere(Aux[k1,z1,:] == np.amax(Aux[k1,z1,:]))
        
        # Value for the policy 
        choice = [[0] * len(self.gridz) for i in range(len(self.gridk))]
        for k1 in range(len(self.gridk)):
            for z1 in range(len(self.gridz)):
                 choice[k1][z1] = [self.gridk[policy[k1][z1][0][0]]]

        return V2, policy, choice
    
    # Iterations of the value function
    def value_function_iteration(self,maxit=1000000,tol=1e-8,printskip=10,showlog=True,howard=False,howard_it=20):
         
        V = np.zeros((len(self.gridk),len(self.gridz)))  # declaring variables 
        loop = 0                                         # initializing counter
        error = tol+1                                    # intializing Error
        start_time = time.time()                         # time
        
        # Loop for each iteration
        while loop < maxit and error > tol:
            V2,policy,choice = self.value_function_operator(V)
            # Howard step
            if howard:
                for count in range(0,howard_it):
                    V_old = V2
                    for k1 in range(len(self.gridk)):
                        for z1 in range(len(self.gridz)):
                            E = 0
                            for z2 in nu.prange(len(self.gridz)):
                                E = E + self.Pi[z1,z2]*V_old[policy[k1][z1][0],z2]
                            V2[k1,z1] = u[k1,z1,policy[k1][z1][0]] + self.beta*E                  
                            
            error = np.amax(np.abs(V2-V)) # calculate error
            loop=loop+1                   # calculate loop
            V = V2                        # redefine V
            if showlog and loop % printskip == 0:
                print(f"\Error at {loop} is {error}")
                hour   = round(((time.time() - start_time)/60)//60)
                minute = round(((time.time() - start_time)//60 - hour*60))
                second = round(((time.time() - start_time)- hour*60*60 - minute*60))
                print(f"\Time elapsed: {hour}h {minute}min {second}s")
            
            
        if loop == maxit:
            print(f"Maximum of {maxit} reached")
            
        if error < tol:
            print(f"Error is lower than tolerance after {loop} iterations ")
            
        return V, policy, choice

    def simulation(self,n,policy,periods=100):      
        # Creating index for possible z in the gridz
        zz = np.arange(len(self.gridz))      
        # Declaring history record
        c_his=np.zeros(0)
        h_his=np.zeros(0)
        k_his=np.zeros(0)
        i_his=np.zeros(0)
        y_his=np.zeros(0)   
        for j in range(n):         
            # Random initial conditions
            z_old = np.random.choice(np.arange(len(self.gridz)))
            k_old = np.random.choice(np.arange(len(self.gridk)))         
            for i in range(periods):              
                # Next step according to policies found
                z_next   = np.random.choice(zz,p = self.Pi[z_old,:])
                k_next   = policy[k_old][z_next][0]
                h_chosen = self.h[k_old,z_next,k_next]
                c_chosen = self.c[k_old,z_next,k_next]
                i_chosen = self.gridk[k_next] - (1-self.delta)*self.gridk[k_old]
                y_chosen = c_chosen + i_chosen
                k_old    = k_next[0]
                z_old    = z_next                
            # Recording in history
            c_his = np.append(c_his,c_chosen)
            h_his = np.append(h_his,h_chosen)
            k_his = np.append(k_his,self.gridk[k_old])
            i_his = np.append(i_his,i_chosen)
            y_his = np.append(y_his,y_chosen)
                              
        # Statistics for n simulations
        c_mean = np.mean(c_his)
        c_var  = np.var(c_his)
        h_mean = np.mean(h_his)
        h_var  = np.var(h_his)
        k_mean = np.mean(k_his)
        k_var  = np.var(k_his)
        i_mean = np.mean(i_his)
        i_var  = np.var(i_his)
        y_mean = np.mean(y_his)
        y_var  = np.var(y_his)
            
        print(f"\n TMoments after {n} simulations and {periods}: \
              \n \
              \n Consumption:  \
              \n Mean: {c_mean} \
              \n Variance: {c_var} \
              \n \
              \n Hours Worked: \
              \n Mean: {24*h_mean} \
              \n Variance: {24*24*h_var} \
              \n \
              \n Capital \
              \n Mean: {k_mean} \
              \n Varaince: {k_var} \
              \n \
              \n Investiment  \
              \n Mean: {i_mean} \
              \n Variance: {i_var} \
              \n \
              \n Output \
              \n Mean: {y_mean} \
              \n Variance: {y_var}" )
                  
        return c_his, h_his, k_his, i_his, y_his
       
  
    # Plotting the policy function for chosen k 
    def plot_k(self,choice,name="Policy"):  
        # Rearranging values for plotting
        plot_k, plot_z = np.meshgrid(gridk, gridz)
        plot_choice = np.concatenate(choice,axis=None).reshape((len(gridk),len(gridz))).transpose()
        fig = plt.figure(figsize=(15,10))
        ax = fig.add_subplot(projection='3d')
        surf = ax.plot_surface(Y = plot_k, X = plot_z, Z = plot_choice, cmap=cm.Greys, linewidth=0,antialiased=False)
        ax.set_xlabel('ln(z)')
        ax.set_ylabel('k')
        ax.set_zlabel('Chosen k\'')
        ax.text2D(0.05, 0.95,"Policy for K'",transform=ax.transAxes)
        fig.colorbar(surf, shrink=0.5, aspect=10)
       
       
       
       
       
# Starting the class RBC_model
rbc = RBC_model(hss,alpha,beta,delta,mu,Pi,gridz,gridk,zss,gamma,h,c,u)
 

#### Item d.

gamma = rbc.calculate_gamma()               #Computing gamma 
rbc.__setattr__('gamma', gamma)             #Update gamma inside the class

gridk = rbc.create_gridk(0.75, 1.25, 101) #Creating a grid for k
rbc.__setattr__('gridk', gridk)           #Updating the grid for k in the class
h = rbc.calculate_h()                     #Creating a grid for h 
rbc.__setattr__('h', h)                   #Updating in the class
c = rbc.calculate_c()                     #Creating a grid for c
rbc.__setattr__('c', c)                   #Updating in the lass
u = rbc.calculate_u()                     #Creating a grid for u
rbc.__setattr__('u', u)                   #Updating in the class

#### Item e.

#Running the value function iteration algorithm
V, policy, choice = rbc.value_function_iteration(tol=1e-8, howard = False)  
rbc.plot_k(choice) # Plotting policy for k

#### Item f.

#Runningthe Valeu Function Iteration algorithm with howard's improvement
V, policy, choice = rbc.value_function_iteration(tol=1e-8, howard=True)
rbc.plot_k(choice, name = 'Policy with howard improvement') # Plotting policy for k

#### Item g.

sim = 1000 # number of simulations
per = 500  # periods 
c_his, h_his, k_his, i_his, y_his = rbc.simulation(sim, policy, periods = per)










###########################################
########## QUESTION 3 #####################

#Parameters
alpha = 0.3
theta = 0.45
eps   = 2       # CARA parameter
beta  = 0.94    # intertemporal discount factor
gamma = 0.89    # probability productivity remain the same nex period
w     = 3       # wage
r     = 0.06    # interest rate
amax  = 500     # max of a
sizea = 500     # size of a
lamb  = 1.5     # values for the financial restriction

#Grid and probabilities for productivity z
gridz = np.array((1,1.6855801,2.3711602, 3.0567403, 3.7423204))
probz = np.array((0.48782122, 0.31137317, 0.09711874, 0.04150281, 0.06218406))


class OccupationalChoice(object):
    
    def __init__(self,alpha,theta,eps,beta,gamma,w,r,gridz,probz,amax,sizea):
        self.alpha,self.theta,self.eps,self.beta,self.gamma,self.w,self.r,self.gridz,self.probz,self.amax,self.sizea = \
        alpha,theta,eps,beta,gamma,w,r,gridz,probz,amax,sizea
        self.grida = np.linspace(0,self.amax,self.sizea)
        
    # defining utility function    
    def utility(self,c):
        u = (c**(1-self.eps))/(1-self.eps)
        return u 
    
    # worker's consumption function
    def calculate_w_c(self):
        c_w = np.zeros((len(self.grida),len(self.gridz),len(self.grida)))
        for a1 in range(len(self.grida)):
            for z1 in range(len(self.gridz)):
                for a2 in range(len(self.grida)):
                    c_w[a1,z1,a2] = self.w + (1+self.r)*self.grida[a1] - self.grida[a2]  
        
        return c_w
    
    # entrepreneur consumption
    def calculate_e_c(self):
        c_e = np.zeros((len(self.grida),len(self.gridz),len(self.grida)))
        for a1 in range(len(self.grida)):
            for z1 in range(len(self.gridz)): 
                for a2 in range(len(self.grida)):
                    c_e[a1,z1,a2] = self.pi[a1,z1,a2] + (1+self.r)*self.grida[a1] - self.grida[a2]
        
        return c_e 
    
    # firms optimal decision regarding capital and labor as a function of w and r
    def calculate_firm_demands(self,z):
         l = ((1/z)*(self.w/self.theta)**(1-self.alpha)*(self.r/self.alpha)**alpha)**(1/(self.alpha+self.theta-1))   
         #l = ((self.w/(z*self.theta))**(1/(self.theta-1))*(self.r/(self.alpha*z))**((self.theta-1)/((self.alpha-1)*self.alpha)))**(self.alpha*self.theta/(self.alpha*self.theta - (self.alpha-1)*(self.theta-1)))
         #l = (self.theta*self.r/(self.alpha*self.w)*(self.r/(self.alpha*z))**(1/(self.alpha-1)))**(self.theta/(self.theta - self.alpha -1))
         #l = ((self.theta*z/self.w)**(1/(1-self.theta))*(self.alpha*z/r)**(-self.alpha/((1-self.alpha)*(1-self.alpha))))**(self.alpha*self.theta/(self.alpha*self.theta + (1-self.alpha)*(1-self.theta)))      
         k = l*(self.alpha*self.w)/(self.r*self.theta)
         return l, k
     
    def calculate_firm_demand_constrained(self,a,z,lamb): 
          k = lamb*a 
          # using foc of labor
          l = (self.w/(self.theta*z*k**self.alpha))**(1/(self.theta-1)) 
          return l, k
      
    def calculate_profit(self,a,z,constrained = False, lamb=1):
          l, k = self.calculate_firm_demands(z)      
          if constrained == False or (constrained and k<=lamb*a):
              pi = z*k**(self.alpha)*l**(self.theta) - self.r*k - self.w*l           
          # now if the friction is binding
          if k>lamb*a and constrained:
              if a == 0 or lamb == 0:
                  pi = 0
              else:
                  l,k = self.calculate_firm_demand_constrained(a,z,lamb)
                  pi = z*k**(self.alpha)*l**(self.theta) - self.r*k - self.w*l   
     # note that when there is a financial friction, the level of a influences the level of k and then prfits
          return pi 
      

    def calculate_profit_matrix(self,constrained = False, lamb =1):
        pi = np.zeros((len(self.grida),len(self.gridz),len(self.grida)))
        for a1 in range(len(self.grida)):
            for z1 in range(len(self.gridz)): 
                for a2 in range(len(self.grida)):
                    pi[a1,z1,a2] = self.calculate_profit(self.grida[a1],self.gridz[z1],constrained,lamb)
        
        return pi 
    
    def occupational_choice(self,constrained = False,lamb=1): #Policy, 0 ou 1.
        o = np.zeros((len(self.grida),len(self.gridz)))
        for z1 in range(len(self.gridz)):
            for a1 in range(len(self.grida)):
                if self.calculate_profit(self.grida[a1],self.gridz[z1],constrained,lamb) > self.w:
                    o[a1,z1] = 1
                else:
                    o[a1,z1] = 0
        
        return o
    
    #Defining the VFO
    def value_function_operator(self,V):
        # V is the guess for the vf (2-D)
        # This operator return the new guess for the value function(V_new), the policy function (returns an index) and a choice
        
        # Auxiliary matrix that consider all possible utilities
        Aux = np.zeros((len(self.grida),len(self.gridz),len(self.grida)))
        
        # List to calculate the new V
        for a1 in range(len(self.grida)):
            for z1 in range(len(self.gridz)): 
                for a2 in range(len(self.grida)):
                  # E measures the expected utility in the next period
                   E = self.gamma*V[a2,z1] + (1-self.gamma)*(V[a2,:] @ self.probz) #vetorial product
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
    def Calculate_intermediaries(self,constrained = False, lamb=1.5):
        self.pi = self.calculate_profit_matrix(constrained = constrained, lamb = lamb)
        self.c_w = self.calculate_w_c()
        self.c_e = self.calculate_e_c()
        self.c_w[self.c_w<0] = 1e-8
        self.c_e[self.c_e<0] = 1e-8
        self.u_w = self.utility(self.c_w)
        self.u_e = self.utility(self.c_e)
        self.u = np.maximum(self.u_e,self.u_w)
        
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
                            E = self.gamma*V_old[policy[a1][z1][0],z1][0] + (1-self.gamma)*(self.probz @ V_old[policy[a1][z1][0],:][0])
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
        




### Item b.

econ = OccupationalChoice(alpha, theta, eps, beta, gamma, w, r, gridz, probz,  amax, sizea) #Starting economy 
econ.Calculate_intermediaries()  #Calculating intermediaries values 
o1 = econ.occupational_choice()  #Calculation occupational choice with no restriction

heatmap_o1 = sns.heatmap(o1)     #Plot heatmap
plt.title('Occupational choice (no friction)')
plt.ylabel('Values on a grid')
plt.xlabel('Values on z grid')
plt.show()   




#### Item c.

econ.Calculate_intermediaries() #Calculating intermediaries values 
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

 


#### Item e.

econ.Calculate_intermediaries(constrained=True,lamb = 1.5)   #Calculating intermediary values 
o2 = econ.occupational_choice(constrained = True, lamb = 1.5) #Generating the occupational choice grid

heatmap_o2 = sns.heatmap(o2)
plt.title('Occupational choice (w/ financial friction)')
plt.ylabel('Values on a grid')
plt.xlabel('Values on z grid')
plt.show()




#### Item f.

V2,policy2,choice2 = econ.value_function_iteration(tol = 1e-8, howard = True) #Iterating the value functions, using howard

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

























        
    
                  
                  
      
      
    
            
                        
    
    
    
    
    
        
    
        
        
    
    
    
 


            
        
                

        
                            
            
    



    
        
        
            
        

