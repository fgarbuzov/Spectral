import pathlib

# assuming that this file is in the project root
PROJ_PATH = str(pathlib.Path(__file__).parent.resolve())
# subfolders
PARAMS_DIR = 'params'
RESULTS_DIR = 'simulation_results'
# filenames
PARAMS_FILENAME = 'params'
U_FILENAME = 'U'
V_FILENAME = 'V'
R_FILENAME = 'R'
# separators
ELEM_SEP = ','
VAL_SEP = '='
ROW_SEP = ';'

# density
RHO = 'rho'
# second-order moduli
YOUNG = 'young'
POISS = 'poiss'
LAMBDA = 'lam'
MU = 'mu'
# third-order moduli
MURN_L = 'l'
MURN_M = 'm'
MURN_N = 'n'
# fourth-order moduli
MURN_NU1 = 'nu1'
MURN_NU2 = 'nu2'
MURN_NU3 = 'nu3'
MURN_NU4 = 'nu4'
# viscous moduli
TAU = 'tau'
VISC_XI = 'xi'
VISC_ETA = 'eta'
# viscous nonlinear moduli (see Garbuzov & Beltukov 2024)
MURN_L1 = 'l1'
MURN_M1 = 'm1'
MURN_N1 = 'n1'
MURN_H1 = 'h1'
NONLIN_TAU = 'nonlin_tau'

# param fields: impact
IMPACT_AMPL = 'force_amplitude'
IMPACT_TIME = 'width'
# param fields: material
ELASTICITY = 'elasticity'
VISCOSITY = 'viscosity'
LINEAR = 'linear'
NONLINEAR = 'nonlinear'
RETARDED = 'retarded'
INSTANT = 'instant'
MATERIAL_TYPE = 'type'
# param fields: geometry and domain
BODY_TYPE = 'body'
BAR = 'bar'
PLATE = 'plate'
ROD = 'rod'
LENGTH = 'L'
H_Y = 'Hy'
H_Z = 'Hz'
RADIUS = 'R'
DOM_NUM = 'domain_number'
DOM_LEN = 'domain_length'
DOM_PNT = 'domain_points'
DOM_PER = 'domain_period'
DOM_BAS = 'domain_basis'
BAS_LEG = 'Legendre'
BAS_CHEB = 'Chebyshev'
BAS_FOUR = 'Fourier'

# time
STOP_TIME = 'tmax'
TIME_STEP = 'dt'
START_TIME = 't0'


BACKUP_ITER = 'backup_iter'

COORD_CYL  = 'cylindrical'
COORD_CART = 'cartesian'
