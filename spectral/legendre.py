import numpy as np
import scipy as sp
import itertools as it
from .mesh import Mesh1D

__all__ = ['Legendre']

class Legendre(Mesh1D):
    def __init__(self, N, endpoints=(-1, 1)):
        self.N = N
        self.shape = N + 1,
        self.endpoints = endpoints
        X1, X2 = endpoints
        #scaling coefficients
        self.a = (X1 + X2)/2
        self.b = (X2 - X1)/2
        
    def grid(self):
        return self.a + self.b*nodes(self.N),
    
    def __eq__(self, mesh):
        if id(self) == id(mesh):
            return True
        if not isinstance(mesh, Legendre):
            return False
        return mesh.N == self.N and mesh.endpoints == self.endpoints
        
    def __ne__(self, mesh):
        return not (self == mesh)
    
    def remesh(self, func, mesh, axes):
        if not isinstance(mesh, Legendre):
            raise ValueError("Improper mesh")
        axis, = axes
        coeff = legcoeff(func, axis)
        coeff2 = np.zeros((mesh.N + 1,) + coeff.shape[1:], coeff.dtype)
        N = min(self.N, mesh.N)
        coeff2[:N + 1] = coeff[:N + 1]
        func = legval(nodes(mesh.N), coeff2)
        return np.moveaxis(func, 0, axis)
    
    def coeff(self, func, axes):
        axis, = axes
        coeff = legcoeff(func, axis) 
        return np.moveaxis(coeff, 0, axis)
    
    def weights(self):
        return self.b*weights(self.N)       
        
    def diff(self, func, axis, dim, bval):
        "Calculate a derivative along a given axis"
        if dim != 0:
            raise ValueError("Legendre is one-dimensional")
        D = diff_matrix(self.N)
        func = np.moveaxis(func, axis, 0)
        res = np.tensordot(D, func, (-1, 0))/self.b
        if bval is not None:
            w = self.weights()
            (b0, b1), = bval
            res[0] += (b0 + func[0])/w[0]
            res[-1] += (b1 - func[-1])/w[-1]
        return np.moveaxis(res, 0, axis)
    
    def match_domains(self, func, axes):
        "Match mesh domains"
        # there are no domains
        return func
        
    def int(self, func, axes):
        "Calculate a definite integral along a given axis"
        axis, = axes
        if axis is None:
            return func
        w = self.b*weights(self.N)
        return np.tensordot(func, w, (axis, 0))
    
    def eval(self, func, X, axes):
        "Calculate values for given points X"
        axis, = axes
        x, = X
        if x is None:
            return func
        x_scaled = (np.asarray(x) - self.a)/self.b
        coeff = legcoeff(func, axis)
        func = legval(x_scaled, coeff)
        return func
    
        
__nodes_cache = {}
def nodes(N):
    "Get Legendre Gauss-Lobatto points"
    if N not in __nodes_cache:
        x = -np.cos(np.pi*np.arange(0, N + 1)/N);
        dx = 1
        while np.abs(dx).max() > np.finfo(float).eps:
            PN1, PN = it.islice(legendre_gen(x), N - 1, N + 1)
            dx = (x*PN - PN1)/(N*PN)
            x -= dx
        __nodes_cache[N] = x
    return __nodes_cache[N]

__weights_cache = {}
def weights(N):
    "Get Legendre Gauss-Lobatto weights"
    if N not in __weights_cache:
        x = nodes(N)
        w = 2/N/(N + 1)/legendre(N, x)**2
        __weights_cache[N] = w
    return __weights_cache[N]

__gamma_cache = {}
def gamma(N):
    "Get Legendre Gauss-Lobatto gamma"
    if N not in __gamma_cache:
        gamma = 1/(.5 + np.arange(N + 1))
        gamma[-1] = 2/N
        __gamma_cache[N] = gamma
    return __gamma_cache[N]    

__diff_matrix_cache = {}
def diff_matrix(N):
    "Return a differentiation matrix"
    if N not in __diff_matrix_cache:
        x = nodes(N)
        P = legendre(N, x)
        with np.errstate(divide='ignore'):
            res = P[:,None]/P/(x[:,None] - x)
        np.fill_diagonal(res, 0)
        res[0, 0] = -N*(N + 1)/4
        res[-1, -1] = N*(N + 1)/4
        __diff_matrix_cache[N] = res
    return __diff_matrix_cache[N]     

#23456789012345678901234567890123456789012345678901234567890123456789012345678

def legendre_gen(x, diff=0):
    "Evaluate Legendre polynomial with degree k at points x"
    u0 = np.zeros_like(x)
    u1 = np.ones_like(x)*sp.special.factorial2(2*diff - 1)
    x = np.asarray(x)
    for k in range(diff):
        yield u0
    for k in it.count(diff):
        yield u1
        A, B = (2*k + 1)/(k + 1 - diff), -(k + diff)/(k + 1 - diff)
        u1, u0 = A*x*u1 + B*u0, u1

def legendre(k, x, diff=0):
    return next(it.islice(legendre_gen(x, diff), k, k + 1))
        
def legval(x, coeff, diff=0):
    """Evaluate a Legendre series at points x.

    Returns
    -------
    Depending on the presence of dom_num the result is:
    
    res[j,k,...,l,m,...] = SUM_i P_i(x[j,k,...])*coeff[i,l,m,...]
    
    """
    shape = coeff.shape[1:]
    res = np.zeros(x.shape + shape, coeff.dtype)
    x = np.asarray(x)
    for i, Pi in zip(range(len(coeff)), legendre_gen(x, diff)):
        res += Pi[(...,) + (None,)*len(shape)]*coeff[i]
    return res

def legcoeff(func, axis=0):
    """Evaluate Legendre coefficients

    Resulting coefficients appears at the first axis

    """
    func = np.moveaxis(func, axis, 0)
    res = np.zeros_like(func)
    N = len(func) - 1
    g = gamma(N)
    x = nodes(N)
    w = weights(N)
    for i, Pi in zip(range(N + 1), legendre_gen(x)):
        res[i] = np.tensordot(func, w*Pi/g[i], (0, 0))
    return res
