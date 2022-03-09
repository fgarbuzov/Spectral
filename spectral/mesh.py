import numpy as np

class Mesh:
    def remesh(self, func, mesh, axes):
        raise NotImplementedError("This subclass of Mesh did not "
                                  "implement remeshing")
        
    def diff(self, func, axis=0, dim=0, bval=None):
        raise NotImplementedError("This subclass of Mesh did not "
                                  "implement differentiation")
        
    def int(self, func, axes):
        raise NotImplementedError("This subclass of Mesh did not "
                                  "implement integration")
    
    def eval(self, func, X, axes):
        raise NotImplementedError("This subclass of Mesh did not "
                                  "implement evaluation")
        
    def __mul__(self, mesh):
        if not isinstance(mesh, Mesh):
            raise ValueError("Could not multiply by not a Mesh object")
        return MeshProduct(self, mesh)

        
class Mesh0D(Mesh):
    "A dummy 0d mesh"
    ndim = 0
    shape = ()

class Mesh1D(Mesh):
    ndim = 1
    def remove_dims(self, mask):
        "Remove dimensions from the mesh accordind to mask"
        remove, = mask
        if remove:
            return Mesh0D()
        else:
            return self


class MeshProduct(Mesh):
    def __init__(self, mesh1, mesh2):
        if not isinstance(mesh1, Mesh):
            raise ValueError("mesh1 is not a mesh")
        if not isinstance(mesh2, Mesh):
            raise ValueError("mesh2 is not a mesh")
        self.mesh1 = mesh1
        self.mesh2 = mesh2
        self.ndim = mesh1.ndim + mesh2.ndim
        self.shape = mesh1.shape + mesh2.shape
        self.dd = self.mesh1.ndim # dividing dimension
        
    def grid(self):
        return self.mesh1.grid() + self.mesh2.grid()

    def __eq__(self, mesh):
        if id(self) == id(mesh):
            return True
        if not isinstance(mesh, MeshProduct):
            return False
        return self.mesh1 == mesh.mesh1 and self.mesh2 == mesh.mesh2
    
    def __ne__(self, mesh):
        return not (self == mesh)
    
    def remove_dims(self, mask):
        "Remove some dimensions from the mesh"
        mesh1 = self.mesh1.remove_dims(mask[:self.dd])
        mesh2 = self.mesh2.remove_dims(mask[self.dd:])
        if mesh1.ndim == 0:
            return mesh2
        if mesh2.ndim == 0:
            return mesh1
        return MeshProduct(mesh1, mesh2)
    
    
    def remesh(self, func, mesh, axes):
        if not isinstance(mesh, MeshProduct):
            raise ValueError("Improper mesh")
        if self.ndim != mesh.ndim:
            raise ValueError("Could not change the number of dimensions")
        func = self.mesh1.remesh(func, mesh.mesh1, axes[:self.dd])
        func = self.mesh2.remesh(func, mesh.mesh2, axes[self.dd:])
        return func
    
    def coeff(self, func, axes):
        res = func
        res = self.mesh1.coeff(res, axes[:self.dd])
        res = self.mesh2.coeff(res, axes[self.dd:])
        return res
    
#23456789012345678901234567890123456789012345678901234567890123456789012345678    
    def diff(self, func, axis, dim, bval):
        "Calculate a derivative along a given axis"
        if not (0 <= dim < self.ndim):
            raise ValueError("Impossible dimensional number")
        if dim < self.mesh1.ndim:
            if bval is not None:
                bval = bval[:self.dd]
            return self.mesh1.diff(func, axis, dim, bval)
        else:
            if bval is not None:
                bval = bval[self.dd:]
            return self.mesh2.diff(func, axis, dim - self.dd, bval)
        
    def match_domains(self, func, axes, masks):
        "Match mesh domains"
        func = self.mesh1.match_domains(func, axes[:self.dd], masks[:self.dd])
        func = self.mesh2.match_domains(func, axes[self.dd:], masks[self.dd:])
        return func
        
    def int(self, func, axes):
        "Calculate a definite integral along a given axis"
        func = self.mesh1.int(func, axes[:self.dd])
        func = self.mesh2.int(func, axes[self.dd:])
        return func

    def eval(self, func, X, axes):
        "Calculate values for given points X"
        axes1 = axes[:self.dd]
        axes2 = axes[self.dd:]
        func = self.mesh2.eval(func, X[self.dd:], axes2)
        removed = sum(x is not None for x in X[self.dd:])
        axes1 = tuple(axis + removed for axis in axes1)
        func = self.mesh1.eval(func, X[:self.dd], axes1)
        return func
        
        
        
