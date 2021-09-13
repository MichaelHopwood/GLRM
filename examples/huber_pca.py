import sys
import os
sys.path.append(os.path.join('..','glrm'))

from glrm.loss import HuberLoss
from glrm.reg import QuadraticReg
from glrm.glrm import GLRM
from glrm.util import pplot
from numpy.random import randn, choice, seed
from numpy import sign
from random import sample
from math import sqrt
from itertools import product
from matplotlib import pyplot as plt
from numpy import ones
import cvxpy as cp

seed(1)

# Generate problem data
m, n, k = 50, 50, 5

sym_noise = 0.2*sqrt(k)*randn(m,n)
asym_noise = sqrt(k)*randn(m,n) + 3*abs(sqrt(k)*randn(m,n)) # large, sparse noise
rate = 0.3 # percent of entries that are corrupted by large, outlier noise
corrupted_entries = sample(list(product(range(m), range(n))), int(m*n*rate))
data = randn(m,k).dot(randn(k,n))
A = data + sym_noise
for ij in corrupted_entries: A[ij] += asym_noise[ij]

A[:, 20] = 0

# Initialize model
loss = HuberLoss
regX, regY = QuadraticReg(0.1), QuadraticReg(0.1)
glrm_huber = GLRM(A, loss, k, regX, regY)

# Fit
X, Y = glrm_huber.fit(solver=cp.SCS)

# Results
A_hat = glrm_huber.predict() # glrm_pca.predict(X, Y) works too; returns decode(XY)
ch = glrm_huber.converged # convergence history
pplot([data, A, A_hat, data-A_hat], ["original", "corrupted", "glrm", "error"])
plt.show()

print()

# Now with missing data
from numpy.random import choice
import numpy as np
missing = list(product(range(int(0.25*m), int(0.75*m)), range(int(0.25*n), int(0.75*n))))

#dt=np.dtype('int,int')
missing = np.asarray(missing)

glrm_huber_missing = GLRM(A, loss, k, regX, regY, missing)
glrm_huber_missing.fit()
A_hat = glrm_huber_missing.predict()
pplot([data, A, missing, A_hat, data-A_hat], ["original", "corrupted", "missing", "glrm", "error"])
