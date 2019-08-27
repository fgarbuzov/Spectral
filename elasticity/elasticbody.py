import numpy as np

class ElasticBody:
    def set_constants(self, type, moduli):
        self.rho = moduli[0]
        self.young = moduli[1]
        self.c = np.sqrt(self.young/self.rho)
        self.nu = moduli[2]
        self.lam = self.young*self.nu/(1 + self.nu)/(1 - 2*self.nu)
        self.mu = self.young/2/(1 + self.nu)
        self.type = type
        if (type == 'murnaghan'):
            self.l = moduli[3]
            self.m = moduli[4]
            self.n = moduli[5]
            self.g1 = moduli[6]
            self.g2 = moduli[7]
            self.g3 = moduli[8]
            self.g4 = moduli[9]
        elif (type == 'landau'):
            self.A = moduli[3]
            self.B = moduli[4]
            self.C = moduli[5]
            self.D = moduli[6]
            self.F = moduli[7]
            self.G = moduli[8]
            self.H = moduli[9]
        else:
            ValueError("Unknown material type")
    
    def grad(self, u):
        "Gradient of displacement"
        raise NotImplementedError("Gradient of displacement has to be "
                                  "implemented in a subclass")

    def cauchy_green(self, u):
        "Cauchy-Green (Green-Lagrange) strain tensor"
        d = self.grad(u)
        return 1/2*(d.T + d + d.T@d)

    def pk1(self, u):
        "First Piola-Kirchhoff stress tensor"
        CG = self.cauchy_green(u)
        if (self.type == 'murnaghan'):
            I1 = CG.trace()
            I2 = 1/2*(CG.trace()**2 - (CG@CG.T).trace())
            dI1 = np.eye(3)
            dI2 = dI1*CG.trace() - CG
            dI3 = CG.cofactor_matrix()
            return (np.eye(3) + self.grad(u))@((self.lam + 2*self.mu)*I1*dI1 
                        - 2*self.mu*dI2 + (self.l + 2*self.m)*I1**2*dI1 
                        - 2*self.m*(dI1*I2 + I1*dI2) + self.n*dI3)
        if (self.type == 'landau'):
            CG2 = CG@CG.T
            I = np.eye(3)
            tr1 = CG.trace()
            tr2 = (CG@CG.T).trace()
            tr3 = (CG@CG.T@CG.T).trace()
            return (np.eye(3) + self.grad(u))@(self.lam*tr1*I + 2*self.mu*CG 
                        + self.A*CG2 + self.B*(tr2*I + 2*tr1*CG) + self.C*tr1**2*I 
                        + self.D*(tr3*I + 3*tr1*CG2) + 2*tr1*self.F*(tr2*I + tr1*CG) 
                        + 4*self.G*tr2*CG + 4*self.H*tr1**3*I)

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


class CircularRod(ElasticBody):
    def __init__(self, L, R, type, moduli):
        self.L = L
        self.R = R
        self.set_constants(type, moduli)

        # coefficients in Boussinesq and Gardner equations
        young = self.young; nu = self.nu
        A = self.A; B = self.B; C = self.C; D = self.D; F = self.F; G = self.G; H = self.H
        self.alpha1 = (1 + nu)/4
        self.alpha2 = -(1 + nu + nu**2)/2
        self.alpha3 = self.alpha1
        self.beta1 = -(3.0/2*young + A*(1 - 2*nu**3)
                       + 3*B*(1 + 2*nu**2)*(1 - 2*nu) + C*(1 - 2*nu)**3)
        self.beta2 = -2*(1 + nu)*(2*B*(1 - 2*nu)*(1 - 2*nu + 6*nu**2) + 2*C*(1 - 2*nu)**3
                                  + nu*(young + 2*A*nu*(1 - 2*nu)))
        self.beta4 = (4*(B + C)**2 - 2*young*(A + 3*B + C + 2*(D + F + G + H)) - young**2/2
                      + 4*nu*(-5*B**2 - 14*B*C - 9*C**2 + young*(3*B + 3*C + 2*D + 4*F + 8*H))
                      + 4*nu**2*(18*B**2 + 44*B*C + 30*C**2 + 2*A*(B + C)
                                 - young*(3*B + 6*C + 6*F + 4*G + 24*H))
                      + 4*nu**3*(-32*B**2 - 76*B*C - 40*C**2 - 6*A*B - 10*A*C
                                 + young*(A + 6*B + 4*C + 2*D + 8*F + 32*H))
                      + 4*nu**4*(A**2 + 28*B**2 + 40*B*C + 12*A*(B + C) - 4*young*(D + 2*F + G + 4*H))
                      - 4*nu**5*(A**2 + 4*A*(B - 2*C) - 4*(3*B**2 + 20*B*C + 12*C**2))
                      - 8*nu**6*(A + 6*B + 4*C)**2)

        self.q4 = nu**2*(7 - 4*nu - 20*nu**2 + 4*nu**3 + 12*nu**4)/48/(1 - nu**2)
        self.q3 = nu*(2*A*(1 - 4*nu**2 + 2*nu**3 + 8*nu**4) + 2*B*(-1 + 6*nu - 18*nu**2 - 4*nu**3 + 48*nu**4)
                      - 2*C*(1 - 2*nu)**3*(3 + 4*nu) + young*(3 - 2*nu - 4*nu**2))/2
        self.q2 = nu*(-4*C*(1 + nu)*(1 - 2*nu)**3 + 4*B*(-1 + 3*nu - 6*nu**2 + 2*nu**3 + 12*nu**4)
                      - nu*(young*(1 + 2*nu) + 4*A*nu*(1 + nu)*(1 - 2*nu)))/2
        self.q1 = -nu**2/2

        self.a2 = 1.0/6 + 2*self.q4/3/self.q1**2 - 2*(2*self.alpha1 + self.alpha2)/3/self.q1
        self.a3 = (self.q3 + (self.q1*(1 - 4*self.a2)
                              - 2*(2*self.alpha1 + self.alpha2))*self.beta1)/3/self.q1/young
        self.a1 = (-2*self.q2 + 3*(6*self.a2 - 1)*self.q1*self.beta1
                   + 12*(2*self.alpha1 + self.alpha2)*self.beta1 + 6*self.a3*self.q1*young)/8/self.beta1

        self.beta4hat = self.beta4 + self.beta1*(self.beta1*(1 - 2*self.a2) - self.a3*young)/3
    
    def grad(self, u):
        return u.grad(coord='cylindrical')

    def kin_energy(self, v):
        return self.rho*(v@v).int(coord='cylindrical')/2

    def pot_energy(self, u):
        CG = self.cauchy_green(u)
        I1 = CG.trace()
        I2 = 1/2*(CG.trace()**2 - (CG@CG.T).trace())
        I3 = CG.det()
        pot = ((self.lam/2 + self.mu)*I1**2 - 2*self.mu*I2 
            + (self.l + 2*self.m)/3*I1**3 - 2*self.m*I1*I2 + self.n*I3)
        return pot.int(coord='cylindrical')
    
    def sol_gardner_params(self, A):
        B = np.sqrt(1 + 3*self.beta4hat*A/2/self.beta1/self.young)
        F = np.sqrt(self.beta1*A/3/self.young/self.R**2/self.q1)
        c = np.sqrt(self.young/self.rho)
        v = -self.beta1*A*c/6/self.young
        return B, F, v

    def sol_gardner_strain(self, A, x, t):
        B, F, v = self.sol_gardner_params(A)
        return A/(1 + B*np.cosh(F*(x - v*t)))

    def sol_gardner_displ(self, A, x, t):
        B, F, v = self.sol_gardner_params(A)
        if (B < 1.0) and (B > -1.0):
            return 2*A*np.arctanh(np.sqrt((1 - B)/(1 + B))*np.tanh(F*(x - v*t)/2))/F/np.sqrt(1 - B**2)
        if B > 1.0:
            return 2*A*np.arctan(np.sqrt((B - 1)/(B + 1))*np.tanh(F*(x - v*t)/2))/F/np.sqrt(B**2 - 1)
        #res += (np.max(res) - np.min(res))/2

    def sol_kdv_strain(self, A, x, t):
        F = np.sqrt(self.beta1*2*A/3/self.young/self.R**2/self.q1)
        v = -self.beta1*A*self.c/3/self.young
        return 2*A/(1 + np.cosh(F*(x - v*t)))
    
    def sol_ekdv_strain(self, u, A, x, t):
        v = self.sol_gardner_params(A)[2]
        c = np.sqrt(self.young/self.rho)
        return (u(x) - (self.R**2*self.a1*u.diff().diff()(x) + self.a2*x*v/c*u.diff()(x)
                        + np.abs(self.a3*u.diff()(x)*self.sol_gardner_displ(A, x, t))))


class RectBar(ElasticBody):
    def __init__(self, L, Hy, Hz, type, moduli):
        self.L = L
        self.Hy = Hy
        self.Hz = Hz
        self.set_constants(type, moduli)
    
    def grad(self, u):
        return u.grad()
