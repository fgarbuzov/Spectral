import numpy as np
from .mesh import Mesh1D

class Multidomain(Mesh1D):
    def __init__(self, domain, partition, periodic=False):
        "Create a one-domensional multidomain with given partitions"
        if not isinstance(domain, Mesh1D):
            raise ValueError("Domain should be a one-domensional mesh")
        self.domain = domain
        self.partition = np.asarray(partition)
        if self.partition.ndim != 1:
            raise ValueError('Partition should be a 1d array')
        if len(self.partition) < 2:
            raise ValueError('Partition should has 2 points at least')
        self.periodic = periodic
        self.M = len(self.partition) - 1
        self.shape = domain.shape[0]*self.M,
        self.endpoints = self.partition[0], self.partition[-1]
        X1, X2 = domain.endpoints
        # scaling coefficients
        self.a = (X2*self.partition[:-1] - X1*self.partition[1:])/(X2 - X1)
        self.b = (self.partition[1:] - self.partition[:-1])/(X2 - X1)

    def grid(self):
        x, = self.domain.grid()
        return np.concatenate([a + b*x for a, b in zip(self.a, self.b)]),
        
#23456789012345678901234567890123456789012345678901234567890123456789012345678            
    def __eq__(self, mesh):
        if id(self) == id(mesh):
            return True
        if not isinstance(mesh, Multidomain):
            return False
        if self.domain != mesh.domain:
            return False
        return np.all(self.partition == mesh.partition)
    
    def __ne__(self, mesh):
        return not (self == mesh)
    
    def remesh(self, func, mesh, axes):
        if not isinstance(mesh, Multidomain):
            raise ValueError("Improper mesh")
        if self.M != mesh.M:
            raise ValueError("Could not change the number of domains")
        axis, = axes
        func = np.moveaxis(func, axis, -1)
        func = func.reshape(func.shape[:-1] + (self.M, -1))
        func = self.domain.remesh(func, mesh.domain, (-1,))
        func = func.reshape(func.shape[:-2] + (-1,))
        return np.moveaxis(func, -1, axis)
    
    def coeff(self, func, axes):
        axis, = axes
        func = np.moveaxis(func, axis, -1)
        func = func.reshape(func.shape[:-1] + (self.M, -1))
        coeff = self.domain.coeff(func, (-1,))
        coeff = coeff.reshape(coeff.shape[:-2] + (-1,))
        return np.moveaxis(coeff, -1, axis)
    
    def diff(self, func, axis, dim, bval):
        "Calculate a derivative along a given axis"
        if dim != 0:
            raise ValueError("Multidomain is one-dimensional")
        func = np.moveaxis(func, axis, -1)
        func = func.reshape(func.shape[:-1] + (self.M, -1))
        if bval is not None:
            (b0, b1), = bval
            b0 = np.asarray(b0)
            b1 = np.asarray(b1)
            b0 = b0[..., None]*np.concatenate(([1], np.zeros(self.M - 1)))
            b1 = b1[..., None]*np.concatenate((np.zeros(self.M - 1), [1]))
            bval = (b0, b1),
        func = self.domain.diff(func, -1, 0, bval)/self.b[:,None]
        func = func.reshape(func.shape[:-2] + (-1,))
        return np.moveaxis(func, -1, axis)
    
    def match_domains(self, func, axes, masks):
        "Match mesh domains"
        axis, = axes
        mask, = masks
        if axis is None:
            return func
        func = np.moveaxis(func, axis, -1)
        func = func.reshape(func.shape[:-1] + (self.M, -1))
        # match boundaries inside each domain
        func = self.domain.match_domains(func, (-1,), ((),))
        # match boundaries between domains
        w = self.domain.weights()
        if not self.periodic:
            w0 = w[0]*self.b[1:]
            w1 = w[-1]*self.b[:-1]
            func0 = func[...,1:,0]
            func1 = func[...,:-1,-1]
            f = np.moveaxis((w0*func0 + w1*func1)/(w0 + w1), -1, axis)[mask]
            np.moveaxis(func0, -1, axis)[mask] = f
            np.moveaxis(func1, -1, axis)[mask] = f
        else:
            w0 = w[0]*self.b
            w1 = w[-1]*np.roll(self.b, 1)
            f = (w0*func[...,0] + w1*np.roll(func[...,-1], 1, -1))/(w0 + w1)
            func[...,0][mask] = f[mask]
            func[...,-1][mask] = np.roll(f[mask], -1, -1) # check mask
        func = func.reshape(func.shape[:-2] + (-1,))
        return np.moveaxis(func, -1, axis)
    
    def int(self, func, axes):
        "Calculate a definite integral along a given axis"
        axis, = axes
        if axis is None:
            return func
        func = np.moveaxis(func, axis, -1)
        func = func.reshape(func.shape[:-1] + (self.M, -1))
        func = self.domain.int(func, (-1,))
        return np.tensordot(func, self.b, (-1, 0))

    def scale(self, x):
        "Find a domain number and scale x"
        x = np.asarray(x)
        if self.periodic:
            x1, x2 = self.endpoints
            x = np.mod(x - x1, x2 - x1) + x1
        dom_num = sum(p < x for p in self.partition[1:-1])
        x_scaled = (x - self.a[dom_num])/self.b[dom_num]
        return dom_num, x_scaled

    def eval(self, func, X, axes):
        "Calculate values for given points X"
        axis, = axes
        x, = X
        if x is None:
            return func
        dom_num, x_scaled = self.scale(x)
        func = np.moveaxis(func, axis, -1)
        func = func.reshape(func.shape[:-1] + (self.M, -1))
        res = np.zeros(x_scaled.shape + func.shape[:-2], func.dtype)
        for m in range(self.M):
            mask = dom_num == m
            X = x_scaled[mask],
            res[mask] = self.domain.eval(func[..., m, :], X, (-1,))
        return res
    
