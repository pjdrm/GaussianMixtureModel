'''
Created on Jul 4, 2017

@author: Pedro Mota
'''
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm 

class GaussianDataGenarator(object):

    def __init__(self, K, n, var):
        self.K = K 
        self.n = n
        self.var = var
        self.u = np.random.normal(0, var, self.K)
        n_K_assignments = np.random.multinomial(n, [1.0/self.K]*self.K)
        self.c = []
        self.X = np.array([])
        for k in range(self.K):
            n_k = n_K_assignments[k]
            self.c += [k]*n_k
            self.X = np.concatenate((self.X, np.random.normal(self.u[k], 1, n_k)))
        self.c = np.array(self.c)
        
    def plot_samples(self):
        k_sampes = []
        k = 0
        colors = iter(cm.Set1(np.linspace(0, 1, self.K)))
        for i, xi in enumerate(self.X):
            if self.c[i] != k:
                plt.hist(sorted(k_sampes),normed=True, color=next(colors), alpha=0.7)
                k_sampes = []
                k += 1
            else:
                k_sampes.append(xi)
        plt.hist(sorted(k_sampes),normed=True, color=next(colors), alpha=0.5)
        plt.legend(np.round(self.u, decimals=2))
        plt.show()

K = 3
n = 3000
var = 2.0
gen = GaussianDataGenarator(K, n, var)
gen.plot_samples()        
