import numpy as np
import scipy as sp
from .mesh import Mesh1D

class Fourier(Mesh1D):
    def __init__(self, N, endpoints=(0, 2*np.pi)):
        self.N = N
        self.shape = N + 1,
        self.endpoints = endpoints
        self.X1, self.X2 = endpoints
    
    def grid(self):
        return np.linspace(self.X1, self.X2, self.N + 1, endpoint=False),
        
    def __eq__(self, mesh):
        if id(self) == id(mesh):
            return True
        if not isinstance(mesh, Fourier):
            return False
        return mesh.N == self.N and mesh.endpoints == self.endpoints
        
    def __ne__(self, mesh):
        return not (self == mesh)  

    def remesh(self, func, mesh, axes):
        if not isinstance(mesh, Fourier):
            raise ValueError("Improper mesh")
        axis, = axes
        f = np.moveaxis(np.fft.rfft(func, axis=axis), axis, 0)
        f2 = np.zeros((mesh.N//2 + 1,) + f.shape[1:], f.dtype)
        N = min(self.N, mesh.N)
        f2[:N//2 + 1] = f[:N//2 + 1]
        func = np.fft.irfft(f2, mesh.N + 1, axis=0)*(mesh.N + 1)/(self.N + 1)
        return np.moveaxis(func, 0, axis)

    def diff(self, func, axis, dim, bval):
        "Calculate a derivative along a given axis"
        if dim != 0:
            raise ValueError("Wave is one-dimensional")
        f = np.fft.rfft(func, axis=axis)
        k = 2*np.pi*np.arange(f.shape[axis])/(self.X2 - self.X1)
        np.moveaxis(f, axis, -1)[:] *= 1j*k
        return np.fft.irfft(f, self.N + 1, axis=axis)

    def match_domains(self, func, axes, masks):
        "Match mesh domains"
        # there are no domains
        return func
        
    def int(self, func, axes):
        "Calculate a definite integral along a given axis"
        axis, = axes
        if axis is None:
            return func
        return (self.X2 - self.X1)*func.sum(axis)/(self.N + 1)

    def eval(self, func, X, axes):
        "Calculate values for given points X"
        axis, = axes
        x, = X
        if x is None:
            return func
        f = np.fft.rfft(func, axis=axis)
        k = np.arange(f.shape[axis])
        x_scaled = 2*np.pi*(np.asarray(x) - self.X1)/(self.X2 - self.X1)
        A = np.exp(1j*np.multiply.outer(k, x_scaled))/(self.N + 1)
        A[1:] *= 2
        return np.tensordot(A, f, (0, axis)).real
