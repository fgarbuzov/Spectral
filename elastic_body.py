import numpy as np
from scipy.integrate import quad
from scipy.special import polygamma, hyp2f1

from spectral import *
from util_names import *


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

        # wave velocities: longitudinal in a rod, pressure, shear, Rayleigh (approx.)
        self.c0 = np.sqrt(self.young/self.rho)
        self.cp = np.sqrt((self.lam + 2*self.mu)/self.rho)
        self.cs = np.sqrt(self.mu/self.rho)
        self.cr = self.cs*(0.862 + 1.14*self.poiss)/(1 + self.poiss)
        # dispersion parameter for longitudinal waves in a rod
        self.q0 = self.poiss**2 #* (1 - self.cs**2 / self.c0**2)
        
        self.material_type = moduli_dict[MATERIAL_TYPE]
        self.is_retarded = RETARDED in self.material_type # retarded viscoelasticity
        self.is_nonlinear = NONLINEAR in self.material_type
        
        # if some values are missing in the dict, these values are set to 0
        # third-order elastic constants (TOEC)
        self.l = moduli_dict.get(MURN_L, 0)
        self.m = moduli_dict.get(MURN_M, 0)
        self.n = moduli_dict.get(MURN_N, 0)
        # fourth-order elastic constants (FOEC)
        self.nu1 = moduli_dict.get(MURN_NU1, 0)
        self.nu2 = moduli_dict.get(MURN_NU2, 0)
        self.nu3 = moduli_dict.get(MURN_NU3, 0)
        self.nu4 = moduli_dict.get(MURN_NU4, 0)
        
        # linear viscoelastic moduli
        self.xi = moduli_dict.get(VISC_XI, np.array([0]))
        self.eta = moduli_dict.get(VISC_ETA, np.array([0]))
        self.tau = moduli_dict.get(TAU, np.array([]))
        self.ret_num = len(self.tau)
        if self.is_retarded and self.ret_num == 0:
            raise ValueError("If the material is retarded relaxation times must be specified")
        
        # if a single number is provided for viscous moduli, turn it into an array
        if len(self.xi) == 1 and len(self.eta) == 1:
            self.xi  = np.array([self.xi[0], ] * self.ret_num)
            self.eta = np.array([self.eta[0],] * self.ret_num)
        
        # final checks
        if (self.ret_num != len(self.xi)) or (self.ret_num != len(self.eta)):
            raise ValueError("Inconsistent sizes of arrays of linear viscoelastic moduli \
                              and relaxation times")
        
        # nonlinear viscoelasticity
        self.nonlin_tau = moduli_dict.get(NONLIN_TAU, np.array([]))
        # mask which tells which 'linear' relaxation times are alse 'nonlinear'
        if len(self.nonlin_tau) == 0:
            self.nonlin_tau_mask = np.zeros_like(self.tau, dtype=bool)
        else:
            self.nonlin_tau_mask = np.sum([np.abs(self.tau/nl_tau - 1) < 1e-14 
                                           for nl_tau in self.nonlin_tau], 
                                           axis=0, dtype=bool)
        nl_tau_num = len(self.nonlin_tau)
        self.l1 = moduli_dict.get(MURN_L1, np.zeros((nl_tau_num + 1,)*2))
        self.m1 = moduli_dict.get(MURN_M1, np.zeros_like(self.l1))
        self.n1 = moduli_dict.get(MURN_N1, np.zeros_like(self.l1))
        self.h1 = moduli_dict.get(MURN_H1, np.zeros_like(self.l1))
        # check the shapes
        if self.l1.shape != (nl_tau_num + 1,)*2:
            raise ValueError("'l1' size must be consistent with the number of \
                              'nonlinear' relaxation times")
        if self.m1.shape != (nl_tau_num + 1,)*2:
            raise ValueError("'m1' size must be consistent with the number of \
                              'nonlinear' relaxation times")
        if self.n1.shape != (nl_tau_num + 1,)*2:
            raise ValueError("'n1' size must be consistent with the number of \
                              'nonlinear' relaxation times")
        if self.h1.shape != (nl_tau_num + 1,)*2:
            raise ValueError("'h1' size must be consistent with the number of \
                              'nonlinear' relaxation times")
        
        # ensure that the quasi-static moduli in l1, m1, and n1 matrices 
        # and the values of the Murnaghan moduli l, m, and n are equal
        self.l1[0,0] = self.l if self.l1[0,0] == 0 else self.l1[0,0]
        self.m1[0,0] = self.m if self.m1[0,0] == 0 else self.m1[0,0]
        self.n1[0,0] = self.n if self.n1[0,0] == 0 else self.n1[0,0]
        self.l = self.l1[0,0]
        self.m = self.m1[0,0]
        self.n = self.n1[0,0]
        
        # quasistatic nonlinear parameters
        # quadratic nonlinearity (KdV, eKdV, Gardner)
        nu = self.poiss
        E = self.young
        self.beta1 = (self.l*(1 - 2*nu)**3 + 2*self.m*(1 + nu)**2*(1 - 2*nu) 
                      + 3*self.n*nu**2) / E
        self.beta2 = (1/E * (1 + nu) * (1 - 2*nu) *
                      (self.l*(1 - 2*nu)**2 + 2*self.m*nu*(1 + nu) - self.n*nu))
        self.beta = self.beta1 + 3/2
        # cubic nonlinearity (eKdV, Gardner)
        self.gamma = (1/2 + 2*self.beta1 - 4/(1 + nu)/(1 - 2*nu)*self.beta2**2
                      + 4*(self.nu1*(1 - 2*nu)**4 
                           - self.nu2*(2 - nu)*nu*(1 - 2*nu)**2 
                           + self.nu3*(1 - 2*nu)*nu**2 
                           + self.nu4*(2 - nu)**2*nu**2)/E)
        
        self.coord = None # to be specified in a subclass
    
    # Solitary wave solution
    def soliton_kdv_velocity_delta(self, A):
        """ Delta between the KdV soliton and linear velocities """
        return A*self.beta*self.c0/3
    
    def soliton_gardner_velocity_delta(self, A, gamma):
        """ Delta between the Gardner soliton and linear velocities """
        return self.soliton_kdv_velocity_delta(A) + A**2*gamma*self.c0/4
    
    def soliton_kdv_ampl(self, L):
        """ KdV soliton amplitude from width parameter L """
        return 6*self.q0*self.Rg**2 / (self.beta*L**2)
    
    def soliton_kdv_width(self, A):
        """ KdV soliton width from amplitude A """
        return self.Rg*np.sqrt(6*self.q0/(A*self.beta))
    
    
    def soliton_kdv(self, A, xi, tau, extended=False):
        """KdV soliton in the form of sech^2((xi - V*tau) / L)
        
        Args:
            A (float): signed amplitude
            xi (float): coordinate
            tau (float): time
            extended (bool, Optional): whether or not to return the extended set of values 

        Returns:
            extended set: (soliton, list of its derivatives with respect to x, 
                           integral over x)
            basic set: (soliton, integral over x)"""
        
        V = self.soliton_kdv_velocity_delta(A)
        B = self.soliton_kdv_width(A)
        arg = (xi - V*tau)/B
        v = A*np.cosh(arg)**(-2)
        if not extended:
            return v, A*B*np.tanh(arg)
        
        ders = [-2*v/B*np.tanh(arg), (-2*v**2/A + 4*v*np.tanh(arg)**2)/B**2]
        return v, ders, A*B*np.tanh(arg)
    
    
    # Stress/strain
    def Strain(self, grad_u):
        "Strain tensor: linear (infinitesimal) or Green-Lagrange (finite)"
        e = (grad_u.T + grad_u) / 2
        if self.is_nonlinear:
            e += grad_u.T @ grad_u / 2
        return e
        
    def kin_energy(self, v):
        "Total kinetic energy"
        return self.rho*(v@v).int(coord=self.coord)/2

    def pot_energy(self, grad_u):
        "Total quasistatic potential energy of elastic deformation (elastic strain energy)"
        strain = self.Strain(grad_u)
        I1 = strain.trace()
        I2 = 1/2*(I1**2 - (strain@strain.T).trace())
        I3 = strain.det()
        pot = ((self.lam/2 + self.mu)*I1**2 - 2*self.mu*I2 
               + (self.l + 2*self.m)/3*I1**3 - 2*self.m*I1*I2 + self.n*I3
               + self.nu1*I1**4 + self.nu2*I1**2*I2 + self.nu3*I1*I3 
               + self.nu4*I2**2)
        return pot.int(coord=self.coord)

    def energy(self, u, v):
        "Full quasistatic mechanical energy"
        return self.kin_energy(v) + self.pot_energy(u.grad(coord=self.coord))
    
    def PK2(self, grad_u):
        "Quasistatic second Piola-Kirchhoff stress tensor (includes FOEC)"
        strain = self.Strain(grad_u)
        strain_sqr = strain@strain.T
        I1 = strain.trace()
        I2 = 1/2*(I1**2 - strain_sqr.trace())
        # calculate third invariant if needed
        I3 = strain.det() if self.nu3 else 0.0
        dI1 = np.eye(self.ndim)
        dI2 = I1*dI1 - strain
        dI3 = strain.cofactor_matrix()
        pk2 = ((self.lam + 2*self.mu)*I1*dI1 - 2*self.mu*dI2 
               + (self.l + 2*self.m)*I1**2*dI1 - 2*self.m*(dI1*I2 + I1*dI2) 
               + self.n*dI3
               + 4*self.nu1*I1**3*dI1 + self.nu2*(2*I1*dI1*I2 + I1**2*dI2)
               + self.nu3*(dI1*I3 + I1*dI3) + 2*self.nu4*I2*dI2)
        return pk2
    
    def PK2_ret(self, grad_u, ret):
        "Retarded second Piola-Kirchhoff stress tensor (only TOEC)"
        strain = self.Strain(grad_u)
        I = np.eye(self.ndim)
        pk2 = self.lam*strain.trace()*I + 2*self.mu*strain
        for i in range(self.ret_num + 1):
            qi = strain if i == 0 else ret[i-1]
            # linear terms (i = 0 - elastic, i > 0 - viscous)
            if i > 0: # the case of i=0 was calculated before the for-loop
                pk2 += self.xi[i-1]*qi.trace()*I + 2*self.eta[i-1]*qi
            # nonlinear terms
            if self.is_nonlinear:
                for j in range(self.ret_num + 1):
                    if ((i == 0 or self.nonlin_tau_mask[i-1]) and 
                        (j == 0 or self.nonlin_tau_mask[j-1])):
                        qj = strain if j == 0 else ret[j-1]
                        i1 = sum(self.nonlin_tau_mask[:i])
                        j1 = sum(self.nonlin_tau_mask[:j])
                        pk2 += ((self.l1[i1,j1] - self.m1[i1,j1] + self.n1[i1,j1]/2)*qi.trace()*qj.trace()*I
                                + (self.m1[i1,j1] - self.n1[i1,j1]/2)*(qi@qj.T).trace()*I
                                + 2*(self.m1[i1,j1] - self.n1[i1,j1]/2 + self.h1[i1,j1])*qi*qj.trace()
                                + self.n1[i1,j1]/2*(qi@qj + qj@qi))
        return pk2
    
    def derivative_nonlin(self, bval, u, v, ret, sponge=0):
        """ Function to pass into an ODE solver. 
            Accounts for nonlinear viscosity. """
        grad_u = u.grad(coord=self.coord)
        grad_v = v.grad(coord=self.coord)
        
        stress = self.PK2_ret(grad_u, ret)
        if self.is_nonlinear:
            stress = (np.eye(self.ndim) + grad_u) @ stress # PK2 to PK1
        
        F = stress.div(bval, coord=self.coord)
        F -= sponge*v
        F = F.match_domains()
        
        du_dt = v
        dv_dt = F/self.rho
        
        if self.is_retarded:
            dret_dt = ((grad_v.T + grad_v + grad_v.T @ grad_u + grad_u.T @ grad_v) / 2 
                       - (ret.T/self.tau).T)
            return [du_dt, dv_dt, dret_dt]
        return [du_dt, dv_dt]



class RectBar(ElasticBody):
    def __init__(self, params_dict):
        self.L = params_dict[LENGTH]
        self.Hy = params_dict[H_Y]
        self.Hz = params_dict[H_Z]
        self.Rg = np.sqrt((self.Hy**2 + self.Hz**2) / 12)
        self.ndim = 3
        super().__init__(params_dict)
        self.coord = COORD_CART


class CylindricalRod(ElasticBody):
    def __init__(self, params_dict):
        self.L = params_dict[LENGTH]
        self.R = params_dict[RADIUS]
        self.Rg = self.R/np.sqrt(2)
        self.ndim = 3
        super().__init__(params_dict)
        self.coord = COORD_CYL
    


# Special functions for the soliton decay theory

def I1lin(x: float):
    return 8/x**4 * (x + x**2/2 + x**3/6 - polygamma(1, 1/x))
def I2lin(x: float):
    return 2/x**2 * (1 + x + x**2/2 + polygamma(2, 1/x) / x**2)

def J(x:float, z:float):
    return -((2 * np.exp(-2*z) * hyp2f1(1, 1+1/2/x, 2+1/2/x, 
                                        -np.exp(-2*z))) / (1 + 2*x) 
             - 1 + np.tanh(z)) / x - np.cosh(z)**(-2)

def I1nl(x:float, y:float):
    return quad(lambda z, x, y: 2*np.cosh(z)**(-2) * np.tanh(z) * J(x,z) * J(y,z),
                -100, 100, args=(x, y))[0]
def I2nl(x:float, y:float):
    return quad(lambda z, x, y: 2*np.cosh(z)**(-2) * (z*np.tanh(z) - 1) * J(x,z) * J(y,z),
                -100, 100, args=(x, y))[0]

def der_lin(t, y, body):
    L, x = y[0], y[1]
    gamma = lame2young(body.xi, body.eta)[0] / body.young
    dL_dt = body.c0/2 * gamma@I1lin(2*body.c0*body.tau/L)
    dx_dt = body.c0 * (1 + 2*body.q0*body.Rg**2/L**2 + 
                       1/2 * gamma@I2lin(2*body.c0*body.tau/L))
    return np.array([dL_dt, dx_dt])

def der_nl(t, y, body):
    L, x = y[0], y[1]
    nu = body.poiss
    beta_su = ((body.l1*(1 - 2*nu)**3 + 2*body.m1*(1 + nu)**2*(1 - 2*nu) 
                + 3*body.n1*nu**2) / body.young) * 3 / body.beta
    beta_su[0, 0] = 0
    tau = np.append([body.tau.max()**3], body.nonlin_tau) # the first one is 'infinity'
    theta, eta = np.meshgrid(body.c0*tau/L, body.c0*tau/L)
    dL_dt = (body.c0*body.q0*body.Rg**2/2/L**2 * 
             beta_su.ravel()@np.vectorize(I1nl)(theta, eta).ravel())
    dx_dt = (body.c0*body.q0*body.Rg**2/2/L**2 * 
             beta_su.ravel()@np.vectorize(I2nl)(theta, eta).ravel())
    return der_lin(t, y, body) + np.array([dL_dt, dx_dt])


# Auxiliary fuctions

def young2lame(young, poiss):
    lam = young*poiss/(1 + poiss)/(1 - 2*poiss)
    mu = young/2/(1 + poiss)
    return lam, mu

def lame2young(lam, mu):
    young = mu*(3*lam + 2*mu)/(lam + mu)
    poiss = lam/2/(lam + mu)
    return young, poiss
