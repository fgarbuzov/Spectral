def landau2murn(A, B, C):
    l = B + C
    m = A/2 + B
    n = A
    return l, m, n

def murn2landau(l, m, n):
    A = n
    B = m - n/2
    C = l - m + n/2
    return A, B, C


def get_constants(type, material):
    if (type == 'murnaghan'):
        if material == 'ps':
            rho = 1.06; young = 3.7; nu = 0.34
            l = -18.9; m = -13.3; n = -10.0
        if material == 'pmma':
            rho = 1.16; young = 4.92; nu = 0.34
            l = -10.93; m = -7.73; n = -1.41
        return [rho, young, nu, l, m, n, 0, 0, 0, 0]
    
    if (type == 'landau'):
        if material == 'pmma':
            rho = 1.16; young = 4.92; nu = 0.34
            A, B, C = murn2landau(-10.93, -7.73, -1.41)
            D = 1000; F = 1000; G = -225.8; H = 1000
        if material == 'ps':
            rho = 1.06; young = 3.7; nu = 0.34
            A, B, C = murn2landau(-18.9, -13.3, -10.0)
            D, F, G, H = 0, 0, 0, 0
        if material == 'pmma2':
            rho = 1.16; young = 4.92; nu = 0.34
            A = -1.41; B = -7.02; C = -3.91
            D = 1000; F = 1000; G = 225.8; H = 1000
        if material == 'mat1':
            rho = 1; young = 5; nu = 0.34
            A = -5.85; D = -2.93
            B, C = A, A; F, G, H = D, D, D
        if material == 'mat2':
            rho = 1; young = 5; nu = 0.34
            A = -5.85; D = 14.18
            B, C = A, A; F, G, H = D, D, D
        if material == 'mat3':
            rho = 1; young = 5; nu = 0.34
            A = -1.17; D = -8.46
            B, C = A, A; F, G, H = D, D, D
        if material == 'mat4':
            rho = 1; young = 5; nu = 0.34
            A = -1.17; D = 8.656
            B, C = A, A; F, G, H = D, D, D
        if material == 'mat5':
            rho = 1; young = 5; nu = 0.34
            A = -3.52; D = 2.4
            B, C = A, A; F, G, H = D, D, D
        if material == 'mat6':
            rho = 1; young = 5; nu = 0.34
            A = -3.52; D = 2.45
            B, C = A, A; F, G, H = D, D, D
        return [rho, young, nu, A, B, C, D, F, G, H]
    
    raise ValueError('No ' + type + ' constants for ' + material)

