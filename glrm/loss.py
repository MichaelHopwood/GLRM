import cvxpy as cp
from numpy import ones, maximum, minimum, sign, floor, ceil
from glrm.util import *

"""
Abstract loss class and canonical loss functions.
"""

# Abstract Loss class
class Loss(object):
    def __init__(self ): return
    def loss(self, A, U): raise NotImplementedError("Override me!")
    def encode(self, A,mask=None): return A,mask # default
    def decode(self, A): return A # default
    def __str__(self): return "GLRM Loss: override me!"
    def __call__(self, A, U): return self.loss(A, U)
    def calc_scaling(self,A,mask):
        alpha = cp.Variable()
        prob = cp.Problem(cp.Minimize(self.loss(A[mask],alpha)))
        sigma = prob.solve()/len(A)
        mu = alpha.value
        return mu,sigma

# Canonical loss functions
class QuadraticLoss(Loss):
    def loss(self, A, U): return cp.sum_squares(A - U)
    def __str__(self): return "quadratic loss"

class HuberLoss(Loss):
    a = 1.0 # XXX does the value of 'a' propagate if we update it?
    def loss(self, A, U): return cp.sum(cp.huber(cp.Constant(A) - U, self.a))
    def __str__(self): return "huber loss"

class HingeLoss(Loss):
    def loss(self, A, U): return cp.sum(cp.pos(ones(A.shape)-cp.multiply(cp.Constant(A), U)))
    def decode(self, A): return sign(A) # return back to Boolean
    def __str__(self): return "hinge loss"

class OrdinalLoss(Loss):
    def __init__(self, A):
        self.Amax, self.Amin = A.max(), A.min()
    def loss(self, A, U):
        return cp.sum(sum(cp.multiply(1*(b >= A),\
                cp.pos(U-b*ones(A.shape))) + cp.multiply(1*(b < A), \
                cp.pos(-U + (b+1)*ones(A.shape))) for b in range(int(self.Amin), int(self.Amax))))
    def decode(self, A): return maximum(minimum(A.round(), self.Amax), self.Amin)
    def __str__(self): return "ordinal loss"

    
class OneVsAllLoss(Loss):
    def loss(self, A, U):
        A_bool = A==1
        obj = cp.sum(cp.pos(1-U[A_bool]))
        obj += cp.sum(cp.pos(1+U[~A_bool]))
        return obj
    def __str__(self): return "scaled one vs all loss"
    def __call__(self, A,U): return self.loss(A, U)
    def encode(self,A,mask): 
#         print(mask)
        np.tile(mask,int(A.max()))
        return oneHotTransform(A), np.tile(mask,int(A.max()+1))
    def decode(self, Z):
        return np.argmax(Z,axis=1)[:,None]
    def calc_scaling(self,A,mask):
        vals, counts = np.unique(A,return_counts=True)
        mu = vals[np.argmax(counts)]
        max_index = A.max()
        mu = np.argmax(A.sum(axis=0))
        mu = oneHotTransform(np.array([[mu]]),max_index=[A.shape[1]-1])
        tot = 0
        for i in range(len(A)):
            if mask[i].all():
#                 print(A[i:i+1].shape)
                tot += self.loss(A[i:i+1],mu)
        
        return mu,tot.value/len(A)
