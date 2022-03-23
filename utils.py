import datetime
import matplotlib.pyplot as plt
import numpy as np

RESULTS_DIR = 'simulation_results'
PARAMS_FILENAME = 'params'
U_FILENAME = 'U'
V_FILENAME = 'V'
ELEM_SEP = ','
VAL_SEP = '='

RHO = 'rho'
YOUNG = 'young'
POISS = 'poiss'
LAMBDA = 'lam'
MU = 'mu'

MURN_L = 'l'
MURN_M = 'm'
MURN_N = 'n'

TAU = 'tau'
VISC_XI = 'xi'
VISC_ETA = 'eta'

IMPACT_AMPL = 'force_amplitude'
IMPACT_TIME = 'width'

ELASTICITY = 'elasticity'
VISCOSITY = 'viscosity'
LINEAR = 'linear'
NONLINEAR = 'nonlinear'
RETARDED = 'retarded'
INSTANT = 'instant'
MATERIAL_TYPE = 'type'

BODY_TYPE = 'body'
BAR = 'bar'
PLATE = 'plate'
LENGTH = 'L'
H_Y = 'Hy'
H_Z = 'Hz'
RADIUS = 'R'
DOM_NUM = 'domain_number'
DOM_LEN = 'domain_length'
DOM_PNT = 'domain_points'
DOM_PER = 'domain_period'

STOP_TIME = 'tmax'
TIME_STEP = 'dt'
START_TIME = 't0'

BACKUP_ITER = 'backup_iter'

#RETARDED_LIN_VISC = 'retarded linear ' + VISCOSITY
#INSTANT_LIN_VISC = 'instant linear ' + VISCOSITY

def young2lame(young, poiss):
    lam = young*poiss/(1 + poiss)/(1 - 2*poiss)
    mu = young/2/(1 + poiss)
    return lam, mu

def lame2young(lam, mu):
    young = mu*(3*lam+2*mu) / (lam+mu)
    poiss = lam / 2 / (lam+mu)
    return young, poiss

def create_simulation_id():
    return datetime.datetime.now().strftime("%Y%b%d_%H-%M-%S")

# Print iterations progress, source: https://stackoverflow.com/questions/3173320/text-progress-bar-in-terminal-with-block-characters
def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
    # Print New Line on Complete
    if iteration == total: 
        print()

def plot_deformations_many_scales(U, T, times_to_plot, x=None):
    fig, ax = plt.subplots(len(times_to_plot), 1, figsize=(13,.5+.6*len(times_to_plot)), sharex=True)

    x_mesh, y_mesh = U.mesh.grid()[0:2]
    if not isinstance(x, np.ndarray):
        x = np.linspace(x_mesh[0], x_mesh[-1], num=1001, endpoint=True)
    y = np.linspace(y_mesh[0], y_mesh[-1], num=101, endpoint=True)
    u = -U[:, 0].diff(0).real
    
    for i in range(len(times_to_plot)):
        im = ax[i].imshow(np.flip(u[times_to_plot[i]](x,y,0).T, axis=0),
                          extent=(x.min(), x.max(), y.min(), y.max()))
        ax[i].set_aspect('equal')
        ax[i].set_ylabel('y, mm')
        #ax[i].set_title('t = %.1f $\mu$s' % T[t[i]], fontsize=11)
        fig.colorbar(im, ax=ax[i], #label=r'$-\partial u_1/\partial x}$', 
                     aspect=5, pad=0.015)
        ax[i].text(0.875, 0.6, '('+'abcdefgh'[i]+') ' + f't = {int(T[times_to_plot[i]])} $\mu$s', c='w', 
                   transform=ax[i].transAxes, fontsize=12)
    
    ax[-1].set_xlabel('x, mm')
    #plt.tight_layout()
    
    return fig, ax