"""
Arno Annys
"""

import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.sparse import spdiags, diags , linalg

class propagator():
    
    def __init__(self,nt = 150):
        '''
        :param nt: number of timesteps
        '''

        #I decided to hard-code most of the parameters, this is an easy fix
        # also the parameters nead propper stability testing but this way the combination is for sure stable

        N = 40
        dt = 0.001

        self.N = N
        self.NN = N**2
        self.nt = nt
        self.dt = dt
        nx=N                               #Number of steps in space(x)
        ny=N                               #Number of steps in space(y)
        self.x=np.linspace(0,2,N)          #Range of x(0,2) and specifying the grid points
        self.y=np.linspace(0,2,N)          #Range of y(0,2) and specifying the grid points
        self.h = self.x[1]                 #spatial stepsize

        #initialize wave function
        x0 = 1                                              #Location of the center of the wavepacket
        y0 = 1                                              #Location of the center of the wavepacket
        velocity = 40                                       #Average velocity of the packet
        k0 =  velocity                                      #Average wavenumber
        sigmax0 = 1/4                                       #Standard deviation of the wavefunction
        sigmay0 = 1/4                                       #Standard deviation of the wavefunction
        Norm_psi =  1/(np.sqrt(sigmax0*np.sqrt(math.pi)))   #Normalization
        [xx, yy] = np.meshgrid(self.x, self.y)
        self.u = np.zeros((nx**2, ny**2))

        self.u = Norm_psi*np.exp(1j*k0*xx)*np.exp(-((np.transpose(xx)-x0)**2/(2*sigmax0**2)+
                                                    (np.transpose(yy)-y0)**2/(2*sigmay0**2)))  #Gaussian pulse

        self.psi = np.reshape(self.u,(self.NN))             #Creating vector for computation
  
    def shape_potential(self, shape: str = ''):
        self.shape = shape

        if shape == 'circle':
            # circular potential well (deep --> infinite)
            grid = np.zeros((self.N,self.N))
            origin = [self.N/2,self.N/2]
            [xgrid,ygrid] = np.meshgrid(np.arange(self.N)-origin[0],np.arange(self.N)-origin[1])
            grid[np.sqrt(xgrid**2 + ygrid**2) >= 0.45*self.N] = 1

            M = np.reshape(grid, (self.NN))
            height = 1000000

        if shape == 'barrier':
            #potential barrier  (small --> tunneling)
            grid = np.zeros((self.N,self.N ))
            grid[:,25] = np.ones(self.N)

            height = 300
            M = np.reshape(grid, (self.NN))

        if shape == 'free':
            #free propagation
            M = np.zeros((self.NN))
            height = 0

        if shape == 'single_slit':
            #single slit
            grid = np.zeros((self.N,self.N ))
            grid[:, 34] = np.ones(self.N)
            grid[16: 22, 34] = 0

            height = 100000
            M = np.reshape(grid, [self.NN])


        if shape == 'double_slit':
            #double slit
            grid = np.zeros((self.N,self.N ))
            grid[:, 29] = np.ones(self.N)
            grid[15: 18, 29] = 0
            grid[20: 23, 29] = 0

            height = 100000
            M = np.reshape(grid, [self.NN])

        self.pot = height*diags( [M] , [0] , shape = (self.NN,self.NN))


    def propagate(self):

        #initializing hamiltonian
        coeff = -1 / (2 * self.h ** 2) #Coëfficiënt
        ham = coeff * spdiags([np.ones(self.NN), np.ones(self.NN), -4*np.ones(self.NN) , np.ones(self.NN),  np.ones(self.NN)], [-self.N, -1, 0, 1, self.N],
                                  self.NN, self.NN) + self.pot

        #initializing Q (sparse)
        Q = 1 / 2 * (spdiags([np.ones(self.NN)], [0], self.NN, self.NN) + 1j * self.dt / 2 * ham)

        #Calculating for each time step
        for it in range(self.nt):
            psin = self.psi
            P = self.psi * np.conj(self.psi)         # Probability density
            PP = np.reshape(P, [self.N, self.N])     # Probability density vector

            #print( 'control: ' , sum(P))            # this value will be displayed in the command window, normalization requires this value
                                                     # to stay constant in time. This can be used as to check stability of the algorithm.

            (X,Y) = np.meshgrid(self.x, self.y)
            fig, ax = plt.subplots(num = 1, subplot_kw={"projection": "3d"}) # plotting
            ax.plot_surface(X, Y, np.real(PP), linewidth=0, antialiased=False, cmap=cm.jet)
            ax.set_title(f'Propagation for potential of type: {self.shape}')
            ax.axis('off')
            
            
          

            chi_tuple = linalg.lsqr(Q,self.psi)
            chi = chi_tuple[0]
            psin = chi - self.psi
            self.psi = psin #replace
        plt.close()