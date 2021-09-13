import cvxpy as cp
import numpy as np
from util import missing2mask
from convergence import *
import sys

class GLRM:
    def __init__(self, A,loss_list, k,regX=None, regY=None,missing_list =None,scale=True):
        self.scale = scale
        self.k = k
        self.A = A
        self.m = A.shape[0]
        self.n = A.shape[1]
        self.loss_list = loss_list
        self.missing_list = missing_list
        self.regX = regX
        self.regY = regY
        self.converged = Convergence()
        self.vals = []
        self.niter = 0
        self._loss_list = []
        if missing_list is not None:
            self.mask = missing2mask(A.shape,missing_list)
        else:
            self.mask = np.ones_like(A,dtype=np.bool)
        self.prep()
#         print(self._loss_list[1][1])
        if self.scale:
            self.calc_scaling()
        else:
            print('here?')
            self.mu = np.ones(A.shape[1])
            self.sigma = np.ones(A.shape[1])
        self._equiv_indices()
        self._initialize_probs()
    def prep(self):
        for columns, loss_fxn in self.loss_list:
            for col in columns:
                data, mask = loss_fxn.encode(self.A[:,col:col+1],self.mask[:,col:col+1])
                self._loss_list.append([data,mask,loss_fxn])
    def _equiv_indices(self):
        sizes = [a[0].shape[1] for a in self._loss_list]
        self.equiv_indices = []
        self.equiv_indices.append(np.arange(0,sizes[0]))
        cur_max = 0

        for i in range(1,len(sizes)):
            cur_max = self.equiv_indices[i-1][-1]+1
            self.equiv_indices.append(np.arange(cur_max,sizes[i]+cur_max))
    def calc_scaling(self):
        self.mu = []
        self.sigma = []
        for i in range(len(self._loss_list)):
            data, mask, loss_fxn = self._loss_list[i]
            mu,sig = loss_fxn.calc_scaling(data,mask)
            if mu.size != 1:
                mu = np.tile(mu,[self.m,1])
            self.mu.append(mu)
            self.sigma.append(sig)
        
#         for columns, loss_fxn in self.loss_list:
#             for col in columns:
#                 elems = self.A[:,col][self.mask[:,col]]
#                 alpha =  cp.Variable()
#                 prob = cp.Problem(cp.Minimize(loss_fxn(elems, alpha)))
#                 self.sigma[col] = prob.solve()/len(elems) #len(elems)-1 per the paper?
#                 self.mu[col] = alpha.value
        
                
                
    def _initialize_probs(self):
        m = self.A.shape[0]
        n = sum([v[0].shape[1] for v in self._loss_list])
#         n = self.A.shape[1]
        
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
        for i,(data, mask, loss_fxn) in enumerate(self._loss_list):
#             if 
#             print(np.tile(self.mu[i],[m,1]))
#             print(i)
#             print(self.mu[i].shape)
#             print(self.equiv_indices[i])
            cols = self.equiv_indices[i]
#             print(Zx[:,cols]+self.mu[i])
            self.objX += loss_fxn(data[mask],(Zx[:,cols]+self.mu[i])[mask])/self.sigma[i]
            self.objY += loss_fxn(data[mask],(Zy[:,cols]+self.mu[i])[mask])/self.sigma[i]
            # need to grab approriate block of XY!!
#             self.objX += loss_fxn(data[mask],Zxcol+self.mu[col])/self.sigma[col]
#             for col in columns:
#                 Acol = self.A[:,col][self.mask[:,col]]
#                 Zxcol = Zx[:,col][self.mask[:,col]]
#                 Zycol = Zy[:,col][self.mask[:,col]]
                
# #                 Acol
# #                 print(col)
# #                 print((Acol,Zx[:,col]+self.mu[col].shape)
#                 self.objX += loss_fxn(Acol,Zxcol+self.mu[col])/self.sigma[col]
#                 self.objY += loss_fxn(Acol,Zycol+self.mu[col])/self.sigma[col]
                
        if self.regX is not None:
            self.objX += self.regX(self.Xv)
        if self.regY is not None:
            self.objY += self.regY(self.Yv)
        self.probX = cp.Problem(cp.Minimize(self.objX))
        self.probY = cp.Problem(cp.Minimize(self.objY))
        
        
    def _initialize_XY(self):
        B = np.hstack([a[0] for a in self._loss_list])
#         print([a[0] for a in self._loss_list])
#         print([a[1] for a in self._loss_list])
        msk = np.hstack([a[1] for a in self._loss_list])
#         B = (self.A-self.mu)/self.sigma
#         print(B.shape)
#         print(msk.shape)
        B[~msk] = 0
        B = (B-B.mean(axis=0))/(B.std(axis=0)+.0001)
        
        U, s, Vh = np.linalg.svd(B,full_matrices=False)
        S = np.diag(s)
        
        X0 = (U @ S)[:,:self.k]
        Y0 = (S @ Vh)[:self.k,:]
#         print(self.Yv.shape)
        self.Xv.value = np.copy(X0)
        self.Xp.value = np.copy(X0)
        
        self.Yv.value = np.copy(Y0)
        self.Yp.value = np.copy(Y0)
        
        self.Xv.value = np.random.rand(self.m,self.k)
        self.Yp.value = np.random.rand(self.k,B.shape[1])

        self.Xp.value = np.random.rand(self.m,self.k)
        self.Yv.value = np.random.rand(self.k,B.shape[1])
    def step(self, solver = cp.ECOS,verboseX =False,verboseY = False,verbose=False):
        if verbose:
            verboseX = True
            verboseY = True
        self.probX.solve(solver,verbose=verbose)
        self.converged.objX.append(self.objX.value)
        self.Xp.value = np.copy(self.Xv.value)

        self.probY.solve(solver,verbose=verbose)
        self.converged.objY.append(self.objY.value)
        self.Yp.value = np.copy(self.Yv.value)
        self.vals.append(self.objY.value)
        sys.stdout.write(f"\r {self.niter} \t {np.round(self.converged.objY[-1],2)}" )
        sys.stdout.flush()
        self.niter+=1

#         return self.Xp.value, self.Yp.value
    def fit(self,N=None, solver = cp.ECOS,verboseX =False,verboseY = False,verbose=False):
        print('iter \t objY')

        if N is None:
            while not self.converged:
                self.step(solver,verboseX,verboseY,verbose)
        else:
            for i in range(N):
                self.step(solver,verboseX,verboseY,verbose)
#             self.probX.solve(solver,verbose=verbose)
#             self.converged.objX.append(self.objX.value)
#             self.Xp.value = np.copy(self.Xv.value)

#             self.probY.solve(solver,verbose=verbose)
#             self.converged.objY.append(self.objY.value)
#             self.Yp.value = np.copy(self.Yv.value)
#             self.vals.append(self.objY.value)
#             sys.stdout.write(f"\r {self.niter} \t {np.round(self.converged.objY[-1],2)}" )
#             sys.stdout.flush()
#             self.niter+=1

#         return self.Xp.value, self.Yp.value
    def Z(self):
        return (self.Xp @ self.Yp).value
    def B(self):
        return np.hstack([a[0] for a in self._loss_list])
    def XY(self):
        return self.Xv.value,self.Yv.value
    def predict(self):
        Z = self.Xv.value @ self.Yv.value
        out = []
        for i in range(len(self._loss_list)):
            cols = self.equiv_indices[i]
            lf = self._loss_list[i][-1]
            out.append(lf.decode(Z[:,cols]+self.mu[i]))
        return np.hstack(out)
#         return (self.Xp @ self.Yp).value+self.mu
    def plot_convergence(self,**kwargs):
        self.converged.plot(**kwargs)
        
