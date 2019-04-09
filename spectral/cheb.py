import numpy as np
import scipy as sp
import scipy.fftpack
import itertools as it



class Cheb:
    # Hight priority to overcome numpy arrays
    __array_priority__ = 1
    # set optimal strategy for differentiation
    diff_threshold = 700
    
    def __init__(self, X1, X2, func=None, coeff=None):
        try:
            self.X1 = tuple(X1)
        except TypeError:
            self.X1 = X1,
        try:
            self.X2 = tuple(X2)
        except TypeError:
            self.X2 = X2,
        self.L = tuple(x2 - x1 for x1, x2 in zip(self.X1, self.X2))
        self.cd = len(self.L) # number of Chebyshev dimensions
        if func is not None:
            self.func = np.asarray(func)
        elif coeff is not None:
            res = np.asarray(coeff)
            norm = 1
            for axis in range(-self.cd, 0):
                res[(..., [0, -1]) + (slice(None),)*(-1 - axis)] *= 2
                res = sp.fftpack.dct(res, type=1, axis=axis)
                norm *= 2
            res = res[(...,) + (slice(None, None, -1),)*self.cd]
            self.func = res/norm             
        else:
            raise('One of `func` and `coeff` is expected')
            
        shape = self.func.shape
        self.cs = shape[-self.cd:] # Chebyshev shape
        self.shape = shape[:-self.cd]
        self.ndim = len(shape) - self.cd
        
    def coeff(self):
        "Return Chebyshev coefficients"
        norm = 1
        res = self.func[(...,) + (slice(None, None, -1),)*self.cd]
        for axis in range(-self.cd, 0):
            res = scipy.fftpack.dct(res, type=1, axis=axis)
            res[(..., [0, -1]) + (slice(None),)*(-1 - axis)] /= 2
            norm *= self.func.shape[axis] - 1
        return res/norm
        
    def __len__(self):
        return self.shape[0]
    
    def __getitem__(self, keys):
        func = self.func[keys]
        return Cheb(self.X1, self.X2, func)
    
    def __setitem__(self, keys, val):
        if isinstance(val, Cheb):
            if self.X1 != val.X1 or self.X2 != val.X2:
                raise ValueError('Inconsistent Chebyshev series')
            self.func[keys] = val.func
        else:
            self.func[keys] = val[(...,) + (None,)*self.cd]
            
        
    def reshape(self, *cs):
        "Reshape Chebyshev coefficients"
        if len(cs) == 1:
            try:
                cs = tuple(cs[0])
            except TypeError:
                pass        
        new_shape = self.shape + cs
        coeff = self.coeff()
        coeff2 = np.zeros(new_shape, coeff.dtype)
        idx = tuple(slice(min(s1, s2)) 
                    for s1, s2 in zip(coeff.shape, new_shape))
        coeff2[idx] = coeff[idx]
        return Cheb(self.X1, self.X2, coeff=coeff2)  
    
        
    def mesh(self, *cs):
        "Return a mesh of collocation points"
        if len(cs) == 0:
            cs = self.cs
        elif len(cs) == 1:
            try:
                cs = tuple(cs[0])
            except TypeError:
                pass
        return [(x1 + x2)/2 + (x1 - x2)/2*np.cos(np.pi*np.arange(s)/(s - 1)) 
               for x1, x2, s in zip(self.X1, self.X2, cs)]
    
    
    def __call__(self, *X):
        "Return function values at certain points"
        axis = -1
        coeff = self.coeff()
        X = X + (None,)*(self.cd - len(X))
        X1, X2 = (), ()
        ndim_new = self.ndim
        for x, x1, x2, s in reversed(list(zip(X, self.X1, self.X2, self.cs))):
            if x is None:
                axis -= 1
                X1 = (x1,) + X1
                X2 = (x2,) + X2
            else:
                phi = np.arccos((2*np.asarray(x) - x1 - x2)/(x2 - x1))
                c = np.cos(phi[..., None]*np.arange(s))
                coeff = np.tensordot(c, coeff, (-1, axis))
                ndim_new += c.ndim - 1
        tr = (tuple(range(ndim_new - self.ndim, ndim_new)) + 
              tuple(range(ndim_new - self.ndim)) + 
              tuple(range(ndim_new, coeff.ndim)))
        coeff = coeff.transpose(tr)
        if len(X1) == 0:
            return coeff
        return Cheb(X1, X2, coeff=coeff)

    def __mul__(self, x):
        "Multiply Chebyshev series by x"
        if isinstance(x, Cheb):
            if self.X1 != x.X1 or self.X2 != x.X2:
                raise ValueError('Inconsistent Chebyshev series')
            return Cheb(self.X1, self.X2, func=self.func*x.func)
        else:
            x = np.asarray(x)
            func = self.func*x[(...,) + (None,)*self.cd]
            return Cheb(self.X1, self.X2, func)
        
    def __rmul__(self, x):
        "Multiply Chebyshev series by x"
        return self*x
    
    def __pow__(self, n):
        "Raise to the power"
        return Cheb(self.X1, self.X2, self.func**n)
    
    def __truediv__(self, x):
        "Divide Chebyshev series by x"    
        if isinstance(x, Cheb):
            if self.X1 != x.X1 or self.X2 != x.X2:
                raise ValueError('Inconsistent Chebyshev series')
            return Cheb(self.X1, self.X2, func=self.func/x.func)
        else:
            x = np.asarray(x)
            func = self.func/x[(...,) + (None,)*self.cd]
            return Cheb(self.X1, self.X2, func)
    
    def __matmul__(self, x):
        if not isinstance(x, Cheb):
            x = np.asarray(x)[(...,) + (None,)*self.cd]
            x = Cheb(self.X1, self.X2, x)
        if self.X1 != x.X1 or self.X2 != x.X2:
            raise ValueError('Inconsistent Chebyshev series')
        f1 = self.func
        f2 = x.func
        if self.ndim == 0 or x.ndim == 0:
            raise ValueError('The number of tensor dimensions of both arrays '
                             'shold be nonzero')
        if x.ndim > 1:
            f1 = np.expand_dims(f1, -1 - self.cd)
            f2 = np.expand_dims(f2, -3 - self.cd)
        if f1.shape[self.ndim - 1] != f2.shape[x.ndim - 1]:
            raise ValueError('Inconsistent tensor dimensions')
        func = sum(f1[(slice(None),)*(self.ndim - 1) + (i,)]*
                   f2[(slice(None),)*(x.ndim - 1) + (i,)]
                   for i in range(f1.shape[self.ndim - 1]))
        return Cheb(self.X1, self.X2, func)
    
    def __add__(self, x):
        "Add x to Chebyshev series"
        if isinstance(x, Cheb):
            if self.X1 != x.X1 or self.X2 != x.X2:
                raise ValueError('Inconsistent Chebyshev series')
            return Cheb(self.X1, self.X2, self.func + x.func)
        else:
            x = np.asarray(x)
            func = self.func + x[(...,) + (None,)*self.cd]
            return Cheb(self.X1, self.X2, func)
        
    def __radd__(self, x):
        "Add x to Chebyshev series"
        return self + x
        
    def __sub__(self, x):
        "Subtract x from Chebyshev series"
        return self + (-x)
    
    def __rsub__(self, x):
        "Subtract x from Chebyshev series"
        return self + (-x)
    
    def __neg__(self):
        "Negate Chebyshev series"
        return Cheb(self.X1, self.X2, -self.func)
    
    def conj(self):
        "Conjugate Chebyshev series"
        return Cheb(self.X1, self.X2, self.func.conj())
    
    @property
    def real(self):
        "Get real part of Chebyshev series"
        return Cheb(self.X1, self.X2, self.func.real)    

    @property
    def imag(self):
        "Get imaginary part of Chebyshev series"
        return Cheb(self.X1, self.X2, self.func.imag)
    
    def int(self, axes=None):
        "Calculate a definite integral along gives axes"
        if axes is None:
            axes = range(self.cd)
        axes = tuple(axis - self.cd if axis >= 0 else axis for axis in axes)
        coeff = self.coeff()
        X1, X2 = (), ()
        for axis in range(-self.cd, 0):
            if axis in axes:
                c = np.zeros(self.cs[axis])
                c[::2] = self.L[axis]/(1 - np.arange(0, self.cs[axis], 2)**2)
                coeff = np.tensordot(coeff, c, (axis, 0))
            else:
                X1 = (self.X1[axis],) + X1
                X2 = (self.X2[axis],) + X2
        if len(X1) == 0:
            return coeff                
        return Cheb(X1, X2, coeff=coeff)
    
    def intx(self, axis=0):
        "Calculate an indefinite integral along a given axis"
        axis = axis - self.cd if axis >= 0 else axis
        tmp = np.moveaxis(self.coeff(), axis, 0)
        coeff = np.zeros((len(tmp) + 1,) + tmp.shape[1:], tmp.dtype)
        coeff[1:] = tmp
        coeff[1:-2] -= tmp[2:]
        coeff[1] += tmp[0]
        coeff.T[...,1:] *= self.L[axis]/4/np.arange(1, len(coeff))
        return Cheb(self.X1, self.X2, coeff=np.moveaxis(coeff, 0, axis))
        
    def diff(self, axis=0, order=1, method=None):
        "Calculate a derivative along a given axis"
        if order == 0:
            return self
        axis = axis - self.cd if axis >= 0 else axis
        if method is None:
            if self.cs[axis] > self.diff_threshold:
                method = 'fourier' 
            else:
                method = 'matrix'
        if method == 'fourier':
            res = np.moveaxis(self.func, axis, -1)/(2*self.cs[axis] - 2)
            res = scipy.fftpack.dct(res, type=1, overwrite_x=True)
            c = -4/self.L[axis]*np.arange(self.cs[axis])
            c[-1] /= 2
            for i in range(order):
                tmp = res*c
                res.T[0:-1:2] = np.cumsum(tmp.T[1::2][::-1], 0)[::-1]
                res.T[1:-1:2] = np.cumsum(tmp.T[2::2][::-1], 0)[::-1]
                res.T[-1] = 0
            res = sp.fftpack.dct(res, type=1, overwrite_x=True)
            res = np.moveaxis(res, -1, axis)
            return Cheb(self.X1, self.X2, res)
        elif method == 'matrix':
            D = self.diff_matrix(self.cs[axis])
            res = np.tensordot(D, self.func, (-1, axis))
            for _ in range(order - 1):
                res = np.tensordot(D, res, (-1, 0))
            res = (2/self.L[axis])**order*np.moveaxis(res, 0, axis)
            return Cheb(self.X1, self.X2, res)
        else:
            raise ValueError('Unknown method: %s' % method)
    
    __diff_matrix_cache = {}
    @classmethod
    def diff_matrix(cls, n):
        "Return a differentiation matrix"
        if n not in cls.__diff_matrix_cache:
            res = np.zeros((n, n))
            for k, func in enumerate(np.eye(n)):
                x = Cheb(-1, 1, func)
                res[:,k] = x.diff(method='fourier').func
            cls.__diff_matrix_cache[n] = res
        return cls.__diff_matrix_cache[n] 
    
    def boundary_response(self, axis=0):
        "Return a boundary responce for the differentiation"
        n = self.cs[axis]
        L = self.L[axis]
        a = (2*n*n - 4*n + 3)/3/L
        b = (-1)**n/L
        return np.array(((-a, b), (-b, a)))  
    
    def boundary_func(self, axis=0):
        "Return boundary function values"
        axis = axis - self.cd if axis >= 0 else axis
        return np.moveaxis(self.func, axis, 0)[[0,-1]]
    
    def boundary_add(self, val, axis=0):
        "Add to boundary function values"
        axis = axis - self.cd if axis >= 0 else axis
        np.moveaxis(self.func, axis, 0)[[0,-1]] += val
        
    def grad(self, wavenumber=None):
        "Calculate a gradient"
        funcs = (self.diff(axis).func for axis in range(self.cd))
        if wavenumber is not None:
            funcs = it.chain([1j*wavenumber*self.funcs], funcs)
        return Cheb(self.X1, self.X2, np.stack(funcs))
    
    def div(self, wavenumber=None):
        "Calculate a divergence"
        if wavenumber is None:
            func = sum(self[axis].diff(axis).func 
                       for axis in range(self.cd))
        else:
            func = sum(self[axis + 1].diff(axis).func 
                       for axis in range(self.cd))
            func += 1j*wavenumber*self.coeff[0]
        return Cheb(self.X1, self.X2, func)
    
    def laplacian(self, wavenumber=None):
        "Calculate a Laplacian"
        func = sum(self.diff(axis, 2).func for axis in range(self.cd))
        if wavenumber is not None:
            func -= wavenumber**2*self.func
        return Cheb(self.X1, self.X2, func)
        
    def trace(self, offset=0, axis1=0, axis2=1):
        "Calculate a trace"
        axis1 = axis1 + self.ndim if axis1 < 0 else axis1
        axis2 = axis2 + self.ndim if axis2 < 0 else axis2
        if axis1 < 0  or axis1 >= self.ndim:
            raise ('Improper value of `axis1`.')
        if axis2 < 0  or axis2 >= self.ndim:
            raise ('Improper value of `axis2`.')
        func = self.func.trace(offset=offset, axis1=axis1, axis2=axis2)
        return Cheb(self.X1, self.X2, func)
        
    def transpose(self, axes=None):
        "Transpose tensor axes"
        if axes is None:
            axes = reversed(range(self.ndim))
        axes = tuple(axes) + tuple(range(self.ndim, self.ndim + self.cd))
        func = self.func.transpose(axes)
        return Cheb(self.X1, self.X2, func)

    @property
    def T(self):
        return self.transpose()
    
    def det(self, axis1=0, axis2=1):
        "Calculate a determinant"
        axis1 = axis1 + self.ndim if axis1 < 0 else axis1
        axis2 = axis2 + self.ndim if axis2 < 0 else axis2
        axis1, axis2 = min(axis1, axis2), max(axis1, axis2)
        if axis1 == axis2:
            raise ValueError('Axes should not be the same.')
        if axis1 < 0  or axis1 >= self.ndim:
            raise ('Improper value of `axis1`.')
        if axis2 < 0  or axis2 >= self.ndim:
            raise ('Improper value of `axis2`.')
        if self.shape[axis1] != self.shape[axis2]:
            raise ValueError('Dimensions over specified axes are unequal.')
        func = self.func
        func = np.moveaxis(func, axis2, -1)
        func = np.moveaxis(func, axis1, -2)
        func = np.linalg.det(func)
        return Cheb(self.X1, self.X2, func)
        
    
    def cofactor_matrix(self, axis1=0, axis2=1):
        "Calculate a cofactor matrix"
        axis1 = axis1 + self.ndim if axis1 < 0 else axis1
        axis2 = axis2 + self.ndim if axis2 < 0 else axis2
        axis1, axis2 = min(axis1, axis2), max(axis1, axis2)
        if axis1 == axis2:
            raise ValueError('Axes numbers should not be the same.')
        if axis1 < 0  or axis2 >= self.ndim:
            raise ('Improper value of axis.')
        if self.shape[axis1] != self.shape[axis2]:
            raise ValueError('Dimensions over specified axes are unequal.')
        n = self.shape[axis1]
        func = self.func
        res_ = np.empty_like(func)
        res = np.moveaxis(res_, (axis1, axis2), (0, 1))
        func = np.moveaxis(func, (axis1, axis2), (0, 1))
        if n == 1:
            res[0, 0] = 1
        elif n == 2:
            res[0, 0] =  func[1, 1]
            res[0, 1] = -func[1, 0]
            res[1, 0] = -func[0, 1]
            res[1, 1] =  func[0, 0]
        elif n == 3:
            res[0, 0] =  func[1, 1]*func[2, 2] - func[1, 2]*func[2, 1]
            res[0, 1] = -func[1, 0]*func[2, 2] + func[1, 2]*func[2, 0]
            res[0, 2] =  func[1, 0]*func[2, 1] - func[1, 1]*func[2, 0]
            
            res[1, 0] = -func[0, 1]*func[2, 2] + func[0, 2]*func[2, 1]
            res[1, 1] =  func[0, 0]*func[2, 2] - func[0, 2]*func[2, 0]
            res[1, 2] = -func[0, 0]*func[2, 1] + func[0, 1]*func[2, 0]
            
            res[2, 0] =  func[0, 1]*func[1, 2] - func[0, 2]*func[1, 1]
            res[2, 1] = -func[0, 0]*func[1, 2] + func[0, 2]*func[1, 0]
            res[2, 2] =  func[0, 0]*func[1, 1] - func[0, 1]*func[1, 0]
        else:
            func = np.moveaxis(func, (0, 1), (-2, -1))            
            for i in range(n):
                func1 = np.delete(func, i, -2)
                for j in range(n):
                    func2 = np.delete(func1, j, -1)
                    cofactor = (-1)**(i + j)*np.linalg.det(func2)
                    res[i, j] = cofactor
        return Cheb(self.X1, self.X2, func=res_)        
        
    

        
        
