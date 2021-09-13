import sys
import os
sys.path.append(os.path.join('..','glrm'))

from glrm.loss import HuberLoss, QuadraticLoss
from glrm.reg import QuadraticReg
from glrm.glrm import GLRM
from glrm.util import pplot
from numpy.random import randn, choice, seed
from numpy import sign
from random import sample
from math import sqrt
from itertools import product
from matplotlib import pyplot as plt
from sklearn import datasets
seed(1)

import numpy as np

iris = datasets.load_iris()
A = iris.data
colors = np.array(['tab:blue','tab:orange','tab:green'])[iris.target]

glrm = GLRM(A,QuadraticLoss,2,regX = QuadraticReg(),regY=QuadraticReg(.1) )

X,Y = glrm.fit()
plt.scatter(X[:,0],X[:,1],color=colors)
plt.show()