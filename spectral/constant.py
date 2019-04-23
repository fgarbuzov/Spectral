import numpy as np
import scipy as sp
from .mesh import Mesh1D

class Constant(Mesh1D):
    def __init__(self):
        self.shape = 1,
    
    def grid(self):
        return np.array([0.]),
        
    def __eq__(self, mesh):
        return isinstance(mesh, Constant)
        
    def __ne__(self, mesh):
        return not (self == mesh)    

    def diff(self, func, axis, dim, bval):
        "Calculate a derivative along a given axis"
        if dim != 0:
            raise ValueError("Constant is one-dimensional")
        return np.zeros_like(func)
        
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
        return np.resize(func, np.shape(x) + func.shape)
