def landau2murn(A, B, C):
    l = B + C
    m = A/2 + B
    n = A
    return [l, m, n]

def murn2landau(l, m, n):
    A = n
    B = m - n/2
    C = l - m + n/2
    return [A, B, C]


def get_constants(type, material):
    if (type == 'murnaghan'):
        if material == 'ps':
            rho = 1.06; young = 3.7; nu = 0.34
            l = -18.9; m = -13.3; n = -10.0
            return [rho, young, nu, l, m, n, 0, 0, 0, 0]
        if material == 'pmma':
            rho = 1.16; young = 4.92; nu = 0.34
            l = -10.9; m = -7.7; n = -1.4
            return [rho, young, nu, l, m, n, 0, 0, 0, 0]
    
    if (type == 'landau'):
        if material == 'pmma':
            rho = 1.16; young = 4.92; nu = 0.34
            A = -1.41; B = -7.02; C = -3.91
            D = 1000; F = 1000; G = -225.8; H = 1000
            return [rho, young, nu, A, B, C, D, F, G, H]
        if material == 'pmma2':
            rho = 1.16; young = 4.92; nu = 0.34
            A = -1.41; B = -7.02; C = -3.91
            D = 1000; F = 1000; G = 225.8; H = 1000
            return [rho, young, nu, A, B, C, D, F, G, H]
        if material == 'mat1':
            rho = 1; young = 5; nu = 0.34
            A = -5.85; D = -1.35
            return [rho, young, nu, A, A, A, D, D, D, D]
        if material == 'mat2':
            rho = 1; young = 5; nu = 0.34
            A = -5.85; D = 15.76
            return [rho, young, nu, A, A, A, D, D, D, D]
        if material == 'mat3':
            rho = 1; young = 5; nu = 0.34
            A = -1.17; D = -8.97
            return [rho, young, nu, A, A, A, D, D, D, D]
        if material == 'mat4':
            rho = 1; young = 5; nu = 0.34
            A = -1.17; D = 8.15
            return [rho, young, nu, A, A, A, D, D, D, D]
    
    raise ValueError('No ' + type + ' constants for ' + material)

