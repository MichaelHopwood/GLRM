import matplotlib.pyplot as plt
from numpy import abs
class Convergence(object):
    def __init__(self, TOL = 1e-2, max_iters = 1e3):
        self.TOL = TOL
        self.max_iters = max_iters
        self.reset()

    def reset(self):
        self.objY = []
        self.objX = []
        self.val = []
        
    def __check_converge(self,obj):
        if len(obj) < 2: 
            return False
        elif len(obj)>self.max_iters:
            print("Max iters as set by convergence object")
            return True
        else:
            return abs((obj[-1]-obj[-2])/obj[-2]) < self.TOL
    def check_convergence_x(self):
        return self.__check_converge(self.objX)
    
    def check_convergence_y(self):
        return self.__check_converge(self.objY)
    
    def check_convergence(self):
        return self.check_convergence_x() and self.check_convergence_y()
    
    def append(self,which, value):
        if which.upper()=='X':
            self.objX.append(value)
        elif which.upper()=='Y':
            self.objY.append(value)
        else:
            raise ValueError(f'Which needs to be X or Y you gave: {which}')
    
    def __len__(self):
        return len(self.objX)+len(self.objY)

#     def __str__(self):
#         return str(self.obj)

#     def __repr__(self):
#         return str(self.obj)
    def __bool__(self):
        return bool(self.check_convergence())
    
    def plot(self,ylog=True,fmt='o--',label_stuff=True,show=True):
        plt.plot(self.objX,fmt,label='X')
        plt.plot(self.objY,fmt,label='Y')
        if ylog:
            plt.yscale('log')
            
        if label_stuff:
            plt.legend()
            plt.title("model error")
            plt.xlabel("iteration")
        if show:
            plt.show()