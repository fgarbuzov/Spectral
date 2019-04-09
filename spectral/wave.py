import numpy as np
import scipy as sp
from .mesh import Mesh1D

class Wave(Mesh1D):
    def __init__(self, k):
        self.shape = 1,
        self.k = k
    
    def grid(self):
        return np.array([]),
        
    def __eq__(self, mesh):
        if id(self) == id(mesh):
            return True
        if not isinstance(mesh, Wave):
            return False
        return mesh.k == self.k
        
    def __ne__(self, mesh):
        return not (self == mesh)    

    def diff(self, func, axis, dim, bval):
        "Calculate a derivative along a given axis"
        if dim != 0:
            raise ValueError("Wave is one-dimensional")
        return 1j*self.k*func
        
    def match_domains(self, func, axes):
        "Match mesh domains"
        # there are no domains
        return func
        
    def int(self, func, axes):
        "Calculate a definite integral along a given axis"
        axis, = axes
        if axis is None:
            return func
        return np.squeeze(func, axis)

    def eval(self, func, X, axes):
        "Calculate values for given points X"
        axis, = axes
        x, = X
        if x is None:
            return func
        func = np.squeeze(func, axis)
        A = np.exp(1j*self.k*np.asarray(x))
        return np.multiply.outer(A, func)
