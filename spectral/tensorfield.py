import numpy as np
from .multidomain import *
from .mesh import *
import pickle 

class TensorField:
    # Hight priority to overcome numpy arrays
    __array_priority__ = 1
    
    def __init__(self, mesh, func):
        """ Make a tensor field on a given spectral mesh.
        
        Input
        -----
        mesh: Mesh
        func: array-like with the following shape:
           tensor shape, spatial shape
        """
        self.mesh = mesh
        self.func = np.asarray(func)
        self.shape = self.func.shape[:-mesh.ndim]
        self.ndim = len(self.shape)
        
    def __len__(self):
        return self.shape[0]
    
    def __getitem__(self, keys):
        func = self.func[keys]
        return TensorField(self.mesh, func)
    
    def __setitem__(self, keys, val):
        if isinstance(val, TensorField):
            if self.mesh != val.mesh:
                raise ValueError('Inconsistent meshes')
            self.func[keys] = val.func
        else:
            self.func[keys] = val[(...,) + (None,)*self.mesh.ndim]        
        
#23456789012345678901234567890123456789012345678901234567890123456789012345678            
    def __call__(self, *X):
        "Return function values at certain points"
        X += (None,)*(self.mesh.ndim - len(X))
        axes = range(-self.mesh.ndim, 0)
        func = self.mesh.eval(self.func, X, axes)
        mask = tuple(x is not None for x in X)
        mesh = self.mesh.remove_dims(mask)
        if mesh.ndim == 0:
            return func
        return TensorField(mesh, func)
    
    def remesh(self, mesh):
        axes = range(-self.mesh.ndim, 0)
        func = self.mesh.remesh(self.func, mesh, axes)
        return TensorField(mesh, func)
    
    def coeff(self):
        axes = range(-self.mesh.ndim, 0)
        return self.mesh.coeff(self.func, axes)
        
#23456789012345678901234567890123456789012345678901234567890123456789012345678 
    def int(self, *dims, coord='cartesian'):
        "Calculate a definite integral along given dimensions"
        if len(dims) == 0:
            dims = range(self.mesh.ndim)
        axes = tuple(d - self.mesh.ndim if d in dims else None 
                     for d in range(self.mesh.ndim))
        if coord == 'cartesian':
            func = self.mesh.int(self.func, axes)
        elif coord == 'cylindrical':
            if self.mesh.ndim != 3:
                raise ValueError('Three dimensions expected.')
            r = self.mesh.grid()[1][:,None]
            func = self.mesh.int(r*self.func, axes)
        else:
            raise ValueError(f'Unknown coordinates: {coord}')
        
        mask = tuple(axis is not None for axis in axes)
        mesh = self.mesh.remove_dims(mask)
        if mesh.ndim == 0:
            return func
        return TensorField(mesh, func)

    def diff(self, dim=0, bval=None):
        "Calculate a derivative along a given dimension"
        axis = dim - self.mesh.ndim  if dim >= 0 else dim
        func = self.mesh.diff(self.func, axis, dim, bval)
        return TensorField(self.mesh, func)

    def match_domains(self, *dims, masks=None):
        "Match mesh domains"
        if len(dims) == 0:
            dims = range(self.mesh.ndim)
        if masks is None:
            masks = [()] * len(dims)
        axes = tuple(d - self.mesh.ndim if d in dims else None 
                     for d in range(self.mesh.ndim))
        func = self.mesh.match_domains(self.func, axes, masks)
        return TensorField(self.mesh, func)    
    
    def grad(self, bval=None, coord='cartesian', rank=None):
        "Calculate a gradient"
        if rank is None:
            rank = self.ndim
        d_func = [self.mesh.diff(self.func, axis, dim, bval)
                  for dim, axis in enumerate(range(-self.mesh.ndim, 0))]
        #print(type(d_func))
        if coord == 'cartesian':
            func = np.stack(d_func)
        elif coord == 'cylindrical':
            if self.mesh.ndim != 3:
                raise ValueError('Three dimensions expected.')
            r = self.mesh.grid()[1][:,None]
            U_x, U_r, U_φ = d_func
            if rank == 0: # scalar
                func = np.stack((U_x, U_r, U_φ/r))
            elif rank == 1: # vector
                Ux, Ur, Uφ = self.func
                Ux_x, Ur_x, Uφ_x = U_x
                Ux_r, Ur_r, Uφ_r = U_r
                Ux_φ, Ur_φ, Uφ_φ = U_φ
                func = np.array([[Ux_x, Ux_r, Ux_φ/r],
                                 [Ur_x, Ur_r, (Ur_φ - Uφ)/r],
                                 [Uφ_x, Uφ_r, (Uφ_φ + Ur)/r]])
            else:
                raise ValueError(f'Improper tensor rank: {rank}')
        else:
            raise ValueError(f'Unknown coordinates: {coord}')
        return TensorField(self.mesh, func)
            

    def div(self, bval=None, coord='cartesian', rank=None):
        "Calculate a divergence"
        if rank is None:
            rank = self.ndim
        d_func = (self.mesh.diff(self.func[axis], axis, dim, bval)
                  for dim, axis in enumerate(range(-self.mesh.ndim, 0)))
        if coord == 'cartesian':
            func = sum(d_func)
        elif coord == 'cylindrical':
            if self.mesh.ndim != 3:
                raise ValueError('Three dimensions expected.')
            if rank == 0:
                raise ValueError('Divergence of scalar function is undefined.')
            r = self.mesh.grid()[1][:,None]
            if rank == 1:
                Ux, Ur, Uφ = self.func
                Ux_x, Ur_r, Uφ_φ = d_func
                func = Ux_x + Ur_r + Ur/r + Uφ_φ/r
            elif rank == 2:
                (Uxx, Uxr, Uxφ), (Urx, Urr, Urφ), (Uφx, Uφr, Uφφ) = self.func
                (Uxx_x, Uxr_x, Uxφ_x), (Urx_r, Urr_r, Urφ_r), (Uφx_φ, Uφr_φ, Uφφ_φ) = d_func
                func = np.array([Uxx_x + Urx_r + Urx/r + Uφx_φ/r,
                                 Urr_r + Uxr_x + Uφr_φ/r + (Urr - Uφφ)/r,
                                 Uφφ_φ/r + Uxφ_x + Urφ_r + (Urφ + Uφr)/r])
            else:
                raise ValueError(f'Improper tensor rank: {rank}')
        else:
            raise ValueError(f'Unknown coordinates: {coord}')
                
        return TensorField(self.mesh, func)  
    
    def curl(self, bval=None, coord='cartesian', rank=None):
        "Calculate a curl"
        if rank is None:
            rank = self.ndim
        d_func = [self.mesh.diff(self.func, axis, dim, bval)
                  for dim, axis in enumerate(range(-self.mesh.ndim, 0))]
        if coord == 'cartesian':
            if rank == 0:
                raise ValueError('Curl of scalar function is undefined.')
            if rank == 1:
                if self.mesh.ndim == 2:
                    (Ux_x, Ux_y), (Uy_x, Uy_y) = d_func
                    func = Uy_x - Ux_y
                else:
                    raise NotImplementedError()
            if rank == 2:
                raise NotImplementedError('Curl of a tensor is not implemented yet')
        elif coord == 'cylindrical':
            raise NotImplementedError('Curl in cylindrical coordinates is not implemented yet')
        else:
            raise ValueError(f'Unknown coordinates: {coord}')
                
        return TensorField(self.mesh, func)
    
    def laplacian(self, bval=None):
        "Calculate a Laplacian"
        res = 0
        for dim, axis in enumerate(range(-self.mesh.ndim, 0)):
            df = self.mesh.diff(self.func, axis, dim, bval)
            res += self.mesh.diff(df, axis, dim, bval)
        return TensorField(self.mesh, res)         
            
    def trace(self, offset=0, axis1=0, axis2=1):
        "Calculate a trace"
        axis1 = axis1 + self.ndim if axis1 < 0 else axis1
        axis2 = axis2 + self.ndim if axis2 < 0 else axis2
        if axis1 < 0  or axis1 >= self.ndim:
            raise ('Improper value of `axis1`.')
        if axis2 < 0  or axis2 >= self.ndim:
            raise ('Improper value of `axis2`.')
        func = self.func.trace(offset=offset, axis1=axis1, axis2=axis2)
        return TensorField(self.mesh, func)
        
    def transpose(self, axes=None):
        "Transpose tensor axes"
        if axes is None:
            axes = tuple(reversed(range(self.ndim)))
        axes = tuple(axes) + tuple(range(len(axes), self.func.ndim))
        func = self.func.transpose(axes)
        return TensorField(self.mesh, func)

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
            raise ValueError('Specified tensor dimensions are unequal.')
        func = self.func
        func = np.moveaxis(func, axis2, -1)
        func = np.moveaxis(func, axis1, -2)
        func = np.linalg.det(func)
        return TensorField(self.mesh, func)
        
    
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
        return TensorField(self.mesh, func=res_)  
    
        
    def __matmul__(self, val):
        if not isinstance(val, TensorField):
            val = np.asarray(val)[(...,) + (None,)*self.mesh.ndim]
            val = TensorField(self.mesh, val)
        if self.mesh != val.mesh:
            raise ValueError('Inconsistent meshes')
        f1 = self.func
        f2 = val.func
        if self.ndim == 0 or val.ndim == 0:
            raise ValueError('The number of tensor dimensions of both arrays '
                             'shold be nonzero')
        if val.ndim > 1:
            f1 = np.expand_dims(f1, -1 - self.mesh.ndim)
            f2 = np.expand_dims(f2, -3 - self.mesh.ndim)
        if f1.shape[self.ndim - 1] != f2.shape[val.ndim - 1]:
            raise ValueError('Inconsistent tensor shapes')
        func = sum(f1[(slice(None),)*(self.ndim - 1) + (i,)]*
                   f2[(slice(None),)*(val.ndim - 1) + (i,)]
                   for i in range(f1.shape[self.ndim - 1]))
        return TensorField(self.mesh, func)    
    
    
    def __mul__(self, val):
        if isinstance(val, TensorField):
            if self.mesh != val.mesh:
                raise ValueError('Inconsistent meshes')
            return TensorField(self.mesh, self.func*val.func)
        else:
            val = np.asarray(val)
            func = self.func*val[(...,) + (None,)*self.mesh.ndim]
            return TensorField(self.mesh, func)
        
    def __rmul__(self, val):
        return self*val
    
    def __pow__(self, n):
        return TensorField(self.mesh, self.func**n)
    
    def __truediv__(self, val):
        if isinstance(val, TensorField):
            if self.mesh != val.mesh:
                raise ValueError('Inconsistent meshes')
            return TensorField(self.mesh, func=self.func/val.func)
        else:
            val = np.asarray(val)
            func = self.func/val[(...,) + (None,)*self.mesh.ndim]
            return TensorField(self.mesh, func)
    
    
    
    
    def __add__(self, val):
        if isinstance(val, TensorField):
            if self.mesh != val.mesh:
                raise ValueError('Inconsistent meshes')
            return TensorField(self.mesh, self.func + val.func)
        else:
            val = np.asarray(val)
            func = self.func + val[(...,) + (None,)*self.mesh.ndim]
            return TensorField(self.mesh, func)
        
    def __radd__(self, val):
        return self + val
        
    def __sub__(self, val):
        return self + (-val)
    
    def __rsub__(self, val):
        return self + (-val)
    
    def __neg__(self):
        return TensorField(self.mesh, -self.func)
    
    
    
    
    def conj(self):
        return TensorField(self.mesh, self.func.conj())
    
    @property
    def real(self):
        return TensorField(self.mesh, self.func.real)    

    @property
    def imag(self):
        return TensorField(self.mesh, self.func.imag)          
