import sys
import os
sys.path.append(os.path.join('..','glrm'))

from glrm.loss import HingeLoss
from glrm.reg import QuadraticReg
from glrm.glrm import GLRM
from glrm.convergence import Convergence
from glrm.util import pplot
from numpy.random import randn, choice, seed
from numpy import sign, ones
import matplotlib.pyplot as plt
from itertools import product
from math import sqrt
from random import sample
seed(1)

# Generate problem data (draw smiley with -1's, 1's)
m, n, k = 500, 500, 8
data = -ones((m, n))
for i,j in product(range(120, 190), range(120, 190)): 
    d = (155-i)**2 + (155-j)**2
    if d <= 35**2: 
        data[i,j] = 1
        data[i, m-j] = 1
for i,j in product(range(300, 451), range(100, 251)):
    d = (250 - i)**2 + (250-j)**2
    if d <= 200**2 and d >= 150**2: 
        data[i,j] = 1
        data[i,m-j] = 1

# sym_noise = 0.2*sqrt(k)*randn(m,n)
# asym_noise = sqrt(k)*randn(m,n) + 3*abs(sqrt(k)*randn(m,n)) # large, sparse noise
# rate = 0.1 # percent of entries that are corrupted by large, outlier noise
# corrupted_entries = sample(list(product(range(m), range(n))), int(m*n*rate))
# A = data + sym_noise
# for ij in corrupted_entries: A[ij] += asym_noise[ij]

A = data

plt.imshow(A)
plt.show()

loss = HingeLoss
regX, regY = QuadraticReg(0.1), QuadraticReg(0.1)
converge = Convergence(TOL = 1e-2, max_iters=1e4)
glrm_binary = GLRM(A, loss, k, regX, regY, converge = converge)

# Fit
glrm_binary.fit()

# Results
X, Y = glrm_binary.factors()
A_hat = glrm_binary.predict() # glrm_pca.predict(X, Y) works too; returns decode(XY)
ch = glrm_binary.convergence() # convergence history
pplot([A, A_hat, A - A_hat], ["original", "glrm", "error"])
