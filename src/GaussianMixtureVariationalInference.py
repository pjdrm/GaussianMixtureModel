'''
Created on Aug 9, 2017

@author: root
'''
from aifc import data
import numpy as np

class GaussianVI(object):
    def __init__(self, data, conv_th=0.0001):
        self.data = data
        self.u = None
        self.m = None
        self.s = None
        self.phi = None
        self.X_mat = self.data.X.reshape(self.data.n, 1)
        self.elbo_constant_f1 = self.data.K*(np.log(1.0/np.sqrt(2.0*np.pi*self.data.var**2))- (1.0/2*self.data.var**2))
        self.elbo_constant1_f2 = -1.0*self.data.n*self.data.K*np.log(self.data.K)
        self.elbo_constant2_f2 = 1.0/(2.0*np.sqrt(np.pi*2.0*self.data.var**2))*self.data.K*self.data.n
        self.conv_th = conv_th
        self.prev_elbo = None
                
    def elbo(self):
        f1 = self.elbo_constant_f1*np.sum(self.u**2 , axis=0)
        f2 =  np.sum(np.sum(self.X_mat*self.u, axis=1), axis=0) + self.elbo_constant1_f2 + self.elbo_constant2_f2
        f3 = np.sum(self.phi*np.log(self.phi), axis=0) # Entropy term (https://www.youtube.com/watch?v=2pEkWk-LHmU&t=1199s)
        f4 = np.sum(np.exp(self.m+0.5*self.s), axis=0) # Maybe I should use entropy (using this for now: https://www.youtube.com/watch?v=rQYcZZ8a64I)
        return f1+f2-f3-f4
    
    def variational_update_phi(self):
        f1 = self.m*self.X_mat #looking at the paper I dont know where uk is used.
        f2 = self.m**2/2.0
        self.phi = f1-f2
        
    def variational_update_m_s(self):
        phi_col_sum = np.sum(self.phi, axis=0)
        f1 = phi_col_sum*self.X
        f2 = phi_col_sum + (1.0/self.data.var**2.0)
        self.m = f1/f2
        self.s = 1.0/f2
        
    def gaussian_mix_model_cavi(self):
        current_elbo = self.elbo()
        while self.prev_elbo - current_elbo > self.conv_th:
            self.variational_update_phi()
            self.variational_update_m_s()
            self.prev_elbo = current_elbo
            current_elbo = self.elbo()
        print(self.m)
        
        
        
        