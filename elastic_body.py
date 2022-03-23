import numpy as np

from spectral import *
from utils import *


class ElasticBody:

    def __init__(self, moduli_dict):
        self.rho = moduli_dict[RHO]
        if YOUNG in moduli_dict:
            self.young = moduli_dict[YOUNG]
            self.poiss = moduli_dict[POISS]
            self.lam, self.mu = young2lame(self.young, self.poiss)
        else:
            self.lam = moduli_dict[LAMBDA]
            self.mu = moduli_dict[MU]
            self.young, self.poiss = lame2young(self.lam, self.mu)

        self.c0 = np.sqrt(self.young/self.rho)
        self.cp = np.sqrt((self.lam + 2*self.mu)/self.rho)
        self.cs = np.sqrt(self.mu/self.rho)

        self.material_type = moduli_dict[MATERIAL_TYPE]
        self.nonlin_elast = ' '.join([NONLINEAR, ELASTICITY]) in self.material_type
        self.retarded = RETARDED in self.material_type
        self.viscous = VISCOSITY in self.material_type
        
        if self.nonlin_elast:
            self.l = moduli_dict[MURN_L]
            self.m = moduli_dict[MURN_N]
            self.n = moduli_dict[MURN_M]
        
        if self.viscous:
            self.xi = moduli_dict[VISC_XI]
            self.eta = moduli_dict[VISC_ETA]
        
        if self.retarded:
            self.tau = moduli_dict[TAU]
    

    def grad(self, u):
        "Gradient of displacement"
        raise NotImplementedError("Gradient of displacement has to be "
                                  "implemented in a subclass")

    def green_lagrange(self, u):
        "Green-Lagrange finite strain tensor"
        d = self.grad(u)
        return (d.T + d + d.T@d) / 2
    
    def pk1(self, u):
        "First Piola-Kirchhoff stress tensor"
        strain = self.green_lagrange(u)
        I1 = strain.trace()
        I2 = 1/2*(strain.trace()**2 - (strain@strain.T).trace())
        dI1 = np.eye(3)
        dI2 = dI1*strain.trace() - strain
        dI3 = strain.cofactor_matrix()
        return (np.eye(3) + self.grad(u))@((self.lam + 2*self.mu)*I1*dI1 
                    - 2*self.mu*dI2 + (self.l + 2*self.m)*I1**2*dI1 
                    - 2*self.m*(dI1*I2 + I1*dI2) + self.n*dI3)
    
    def strain_lin(self, u):
        d = self.grad(u)
        return 1/2*(d.T + d)
    
    def sigma(self, u, lam, mu):
        "Linear stress tensor"
        eps = self.strain_lin(u)
        return lam*eps.trace()*np.eye(3) + 2*mu*eps
    
    def stress_visc(self, v):
        q = self.strain_lin(v)
        stress = self.xi * (q.trace() * np.eye(3))
        stress += 2 * self.eta * q
        return stress
    
    def stress_visc_ret(self, ret):
        stress = 0
        for i in range(len(self.tau)):
            q = self.strain_lin(ret[i])
            stress += self.xi[i] * (q.trace() * np.eye(3))
            stress += 2 * self.eta[i] * q
        return stress
    
    def stress_elast(self, u):
        if self.nonlin_elast:
            return self.pk1(u)
        return self.sigma(u, self.lam, self.mu)

    def kin_energy(self, v):
        "Kinetic energy"
        raise NotImplementedError("Kinetic energy function has to be "
                                  "implemented in a subclass")

    def pot_energy(self, u):
        "Potential energy of elastic deformation"
        raise NotImplementedError("Potential energy function has to be "
                                  "implemented in a subclass")

    def energy(self, u, v):
        "Full mechanical energy"
        return self.kin_energy(v) + self.pot_energy(u)


class RectBar(ElasticBody):
    # def __init__(self, geom_dict, moduli_dict):
    #     self.L = geom_dict[LENGTH]
    #     self.Hy = geom_dict[H_Y]
    #     self.Hz = geom_dict[H_Z]
    #     self.ndim = 3
    #     super().__init__(moduli_dict)
    
    def __init__(self, params_dict):
        self.L = params_dict[LENGTH]
        self.Hy = params_dict[H_Y]
        self.Hz = params_dict[H_Z]
        self.ndim = 3
        super().__init__(params_dict)
    
    def grad(self, u):
        return u.grad()
    
    def kin_energy(self, v):
        return self.rho*(v@v).int()/2

    def pot_energy(self, u):
        CG = self.green_lagrange(u)
        I1 = CG.trace()
        I2 = 1/2*(CG.trace()**2 - (CG@CG.T).trace())
        I3 = CG.det()
        pot = (self.lam/2 + self.mu)*I1**2 - 2*self.mu*I2 
        if self.nonlin_elast:
            pot += (self.l + 2*self.m)/3*I1**3 - 2*self.m*I1*I2 + self.n*I3
        return pot.int()
    
    def derivative(self, u, v, ret, bval):
        stress = self.stress_elast(u)
        dret_dt = ret
        if self.viscous:
            if self.retarded:
                stress += self.stress_visc_ret(ret)
                dret_dt = v - (ret.T/self.tau).T
            else:
                stress += self.stress_visc(v)
        
        #stress = stress.match_domains()
        #bval = (f(t), 0), (0, 0), (0, 0)
        F = stress.div(bval)
        F = F.match_domains()
        
        du_dt = v
        dv_dt = F/self.rho

        return du_dt, dv_dt, dret_dt
