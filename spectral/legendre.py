import numpy as np
import scipy as sp
import scipy.special
import itertools as it
from .mesh import Mesh1D

__all__ = ['Legendre']

class Legendre(Mesh1D):
    def __init__(self, N, endpoints=(-1, 1), quadrature='Lobatto'):
        self.N = N
        self.shape = N,
        self.endpoints = endpoints
        self.quadrature = quadrature
        X1, X2 = endpoints
        #scaling coefficients
        self.a = (X1 + X2)/2
        self.b = (X2 - X1)/2
        
    def grid(self):
        return self.a + self.b*nodes(self.N, self.quadrature),
    
    def __eq__(self, mesh):
        if id(self) == id(mesh):
            return True
        if not isinstance(mesh, Legendre):
            return False
        return (mesh.N == self.N and mesh.endpoints == self.endpoints
                and mesh.quadrature == self.quadrature)
        
    def __ne__(self, mesh):
        return not (self == mesh)
    
    def remesh(self, func, mesh, axes):
        if not isinstance(mesh, Legendre):
            raise ValueError("Improper mesh")
        axis, = axes
        coeff = legcoeff(func, self.quadrature, axis)
        coeff2 = np.zeros((mesh.N,) + coeff.shape[1:], coeff.dtype)
        N = min(self.N, mesh.N)
        coeff2[:N] = coeff[:N]
        func = legval(nodes(mesh.N, self.quadrature), coeff2)
        return np.moveaxis(func, 0, axis)
    
    def coeff(self, func, axes):
        axis, = axes
        coeff = legcoeff(func, axis) 
        return np.moveaxis(coeff, 0, axis)
    
    def weights(self):
        return self.b*weights(self.N, self.quadrature)       
        
    def diff(self, func, axis, dim, bval):
        "Calculate a derivative along a given axis"
        if dim != 0:
            raise ValueError("Legendre is one-dimensional")
        D = diff_matrix(self.N, self.quadrature)
        func = np.moveaxis(func, axis, 0)
        res = np.tensordot(D, func, (-1, 0))/self.b
        if bval is not None:
            w = self.weights()
            (b0, b1), = bval
            res[0] += (b0 + func[0])/w[0]
            if self.quadrature == 'Lobatto':
                res[-1] += (b1 - func[-1])/w[-1]
        return np.moveaxis(res, 0, axis)
    
    def match_domains(self, func, axes, masks):
        "Match mesh domains"
        # there are no domains
        return func
        
    def int(self, func, axes):
        "Calculate a definite integral along a given axis"
        axis, = axes
        if axis is None:
            return func
        w = self.b*weights(self.N, self.quadrature)
        return np.tensordot(func, w, (axis, 0))
    
    def eval(self, func, X, axes):
        "Calculate values for given points X"
        axis, = axes
        x, = X
        if x is None:
            return func
        x_scaled = (np.asarray(x) - self.a)/self.b
        coeff = legcoeff(func, self.quadrature, axis)
        func = legval(x_scaled, coeff)
        return func
    
def memoize(f):
    results = {}
    def helper(*args):
        if args not in results:
            results[args] = f(*args)
        return results[args]
    return helper    
    
@memoize
def nodes(N, quadrature):
    "Get Legendre quadrature points"
    if quadrature == 'Lobatto':
        x = -np.cos(np.pi*np.arange(0, N)/(N - 1));
        dx = 1
        while np.abs(dx).max() > np.finfo(float).eps:
            PN2, PN1 = it.islice(legendre_gen(x), N - 2, N) # P_(N - 2) and P_(N - 1)
            dx = (PN2 - x*PN1)/(N*PN1)
            x += dx
        return x
    if quadrature == 'Radau':
        x = -np.cos(np.pi*np.arange(0, N)/N);
        dx = 1
        while np.abs(dx).max() > np.finfo(float).eps:
            PN1, PN = it.islice(legendre_gen(x), N - 1, N + 1) # P_(N - 1) and P_N
            dx = (1 - x)*(PN + PN1)/N/(PN - PN1)
            x += dx
        return x
        

@memoize
def weights(N, quadrature):
    "Get Legendre qudrature weights"
    x = nodes(N, quadrature)
    if quadrature == 'Lobatto':
        return 2/N/(N - 1)/legendre(N - 1, x)**2
    if quadrature == 'Radau':
        return (1 - x)/N**2/legendre(N - 1, x)**2
        

@memoize
def gamma(N, quadrature):
    "Get Legendre qudrature gamma"
    gamma = 1/(.5 + np.arange(N))
    if quadrature == 'Lobatto':
        gamma[-1] = 2/(N - 1)
    return gamma

@memoize
def diff_matrix(N, quadrature):
    "Return a differentiation matrix"
    x = nodes(N, quadrature)
    P = legendre(N - 1, x)
    if quadrature == 'Lobatto':
        with np.errstate(divide='ignore'):
            res = P[:,None]/P/(x[:,None] - x)
        np.fill_diagonal(res, 0)
        res[0, 0] = -N*(N - 1)/4
        res[-1, -1] = N*(N - 1)/4
    elif quadrature == 'Radau':
        with np.errstate(divide='ignore'):
            Px = P/(1 - x)
            res = Px[:,None]/Px/(x[:,None] - x)
        res[np.diag_indices_from(res)] = 1/2/(1 - x)
        res[0, 0] = -(N - 1)*(N + 1)/4
    return res

#23456789012345678901234567890123456789012345678901234567890123456789012345678

def legendre_gen(x, diff=0):
    "Generator of legendre polynomials at points x"
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
    "Evaluate Legendre polynomial with degree k at points x"
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

def legcoeff(func, quadrature, axis=0):
    """Evaluate Legendre coefficients

    Resulting coefficients appears at the first axis

    """
    func = np.moveaxis(func, axis, 0)
    res = np.zeros_like(func)
    N = len(func)
    g = gamma(N, quadrature)
    x = nodes(N, quadrature)
    w = weights(N, quadrature)
    for i, Pi in zip(range(N), legendre_gen(x)):
        res[i] = np.tensordot(func, w*Pi/g[i], (0, 0))
    return res
