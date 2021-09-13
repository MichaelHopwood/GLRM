import cvxpy as cp
import numpy as np
from glrm.util import missing2mask
from glrm.convergence import *
import sys
from collections.abc import Iterable

class GLRM:
    def __init__(self, A,loss_list, k,regX=None, regY=None,missing_list =None, converge = None, scale=True):
        self.scale = scale
        self.k = k
        self.A = A
        self.loss_list = loss_list
        self.missing_list = missing_list
        self.regX = regX
        self.regY = regY
        if converge == None: 
            self.converged = Convergence()
        else: 
            self.converged = converge
        self.vals = []
        self.niter = 0
        if missing_list is not None:
            self.mask = missing2mask(A.shape,missing_list)
        else:
            self.mask = np.ones_like(A,dtype=np.bool)
        if self.scale:
            self.calc_scaling()
        else:
            self.mu = np.ones(A.shape[1])
            self.sigma = np.ones(A.shape[1])
        self._initialize_probs()
    def calc_scaling(self):
        self.mu = np.zeros(self.A.shape[1])
        self.sigma = np.zeros(self.A.shape[1])
        columns = self.A.shape[1]

        if not isinstance(self.loss_list, Iterable):
            # e is iterable
            self.loss_list = [self.loss_list]

        for loss_fxn in self.loss_list:
            for col in range(columns):
                elems = self.A[:,col][self.mask[:,col]]
                alpha =  cp.Variable()
                prob = cp.Problem(cp.Minimize(loss_fxn.loss(loss_fxn, elems, alpha * np.ones(elems.shape))))
                self.sigma[col] = prob.solve()/len(elems) #len(elems)-1 per the paper?
                self.mu[col] = alpha.value
        
                
                
    def _initialize_probs(self):
        m = self.A.shape[0]
        n = self.A.shape[1]
        
        self.Xp = cp.Parameter((m,self.k))
        self.Xv = cp.Variable((m,self.k))
        
        self.Yp = cp.Parameter((self.k,n))
        self.Yv = cp.Variable((self.k,n))
        
        #Random Intialization 
        self.Xv.value = np.random.rand(m,self.k)
        self.Xp.value = np.random.rand(m,self.k)

        self.Yp.value = np.random.rand(self.k,n)
        self.Yv.value = np.random.rand(self.k,n)        
        self._initialize_XY()
        self.objX = 0
        self.objY = 0
        Zx = self.Xv @ self.Yp
        Zy = self.Xp @ self.Yv
        columns = self.A.shape[1]
        for loss_fxn in self.loss_list:
            for col in range(columns):
                Acol = self.A[:,col][self.mask[:,col]]
                Zxcol = Zx[:,col][self.mask[:,col]]
                Zycol = Zy[:,col][self.mask[:,col]]
                
#                 Acol
#                 print(col)
#                 print((Acol,Zx[:,col]+self.mu[col].shape)
                self.objX += loss_fxn.loss(loss_fxn, Acol,Zxcol+self.mu[col])/self.sigma[col]
                self.objY += loss_fxn.loss(loss_fxn, Acol,Zycol+self.mu[col])/self.sigma[col]
                
        if self.regX is not None:
            self.objX += self.regX(self.Xv)
        if self.regY is not None:
            self.objY += self.regY(self.Yv)
        self.probX = cp.Problem(cp.Minimize(self.objX))
        self.probY = cp.Problem(cp.Minimize(self.objY))
        
        
    def _initialize_XY(self):
        B = (self.A-self.mu)/self.sigma
        B[~self.mask] = 0
        
        U, s, Vh = np.linalg.svd(B,full_matrices=False)
        S = np.diag(s)
        
        X0 = (U @ S)[:,:self.k]
        Y0 = (S @ Vh)[:self.k,:]
        
        self.Xv.value = np.copy(X0)
        self.Xp.value = np.copy(X0)
        
        self.Yv.value = np.copy(Y0)
        self.Yp.value = np.copy(Y0)
    def fit(self, solver = cp.ECOS,verboseX =False,verboseY = False,verbose=False):
        if verbose:
            verboseX = True
            verboseY = True
        print('iter \t objY')
        while not self.converged:

            self.converged.objX.append(self.probX.solve(solver,verbose=verbose))
            self.Xp.value = np.copy(self.Xv.value)

            self.converged.objY.append(self.probY.solve(solver,verbose=verbose))
            self.Yp.value = np.copy(self.Yv.value)
            self.vals.append(self.objY.value)
            sys.stdout.write(f"\r {self.niter} \t {np.round(self.converged.objY[-1],2)}" )
            sys.stdout.flush()
            self.niter+=1

        return self.Xp.value, self.Yp.value
    def predict(self):
        return (self.Xp @ self.Yp).value+self.mu
    def plot_convergence(self,**kwargs):
        self.converged.plot(**kwargs)
