from elastic_body import *
from spectral import *
from utils import *

import numpy as np
import scipy as sp
import scipy.integrate

import sys
import os
import pathlib


# assuming the structure PROJ_DIR/simulation.py
PROJ_DIR = str(pathlib.Path(__file__).parent.resolve())

def parse_line(line):
    l = line.split(VAL_SEP)
    if len(l) < 2:
        err_str = "incorrect line '{}' in the input file".format(line)
        raise ValueError(err_str)
    name = l[0].strip()
    value = l[1].strip()
    return name, value

def is_comment_or_empty_line(line):
    return line.startswith(('#', '\n'))

def parse_input_file(file):
    param_dict = {}
    number_vals_list = [RHO, LAMBDA, MU, YOUNG, POISS, MURN_L, MURN_M, MURN_N]
    type = ''
    for line in file:
        if is_comment_or_empty_line(line):
            continue
        name, value = parse_line(line)
        # material file
        if name == MATERIAL_TYPE:
            type = value
        elif name in number_vals_list:
            value = float(value)
        elif name == TAU:
            if not RETARDED in type:
                raise ValueError('tau is for the retarded viscosity')
            arr = value.split(ELEM_SEP)
            value = np.array([float(el.strip()) for el in arr])
        elif name in [VISC_ETA, VISC_XI]:
            if RETARDED in type:
                arr = value.split(ELEM_SEP)
                value = np.array([float(el.strip()) for el in arr])
            else:
                value = float(value)
        # body file
        elif name == BODY_TYPE:
            #value = float(value)
            pass
        elif name in [LENGTH, H_Y, H_Z, RADIUS]:
            value = float(value)
        elif name in [DOM_NUM, DOM_PNT]:
            arr = value.split(ELEM_SEP)
            value = np.array([int(el.strip()) for el in arr])
        elif name == DOM_PER:
            arr = value.split(ELEM_SEP)
            value = np.array([el.lower().strip() == 'true' for el in arr])
        # impact file
        elif name in [IMPACT_AMPL, IMPACT_TIME]:
            value = float(value)
        # time
        elif name in [START_TIME, TIME_STEP, STOP_TIME]:
            value = float(value)
        # backup
        elif name == BACKUP_ITER:
            value = int(value)
        # error            
        else:
            err_str = "unknown parameter '{}' in the input file '{}'".format(name, file.name)
            raise ValueError(err_str)
        # save parameter
        param_dict[name] = value
    return param_dict

def compress_ret(u, v, retarded):
    return np.stack((u.func, v.func, *retarded.func)).ravel().view(float)

def compress(u, v):
    return np.stack((u.func, v.func)).ravel().view(float)

def decompress_ret(y):
    Y = y.reshape(2 + len(body.tau), 3, *mesh.shape)
    u = TensorField(mesh, Y[0])
    v = TensorField(mesh, Y[1])
    retarded = TensorField(mesh, Y[2:])
    return u, v, retarded

def decompress(y):
    Y = y.reshape(2, 3, *mesh.shape)
    u = TensorField(mesh, Y[0])
    v = TensorField(mesh, Y[1])
    return u, v

def derivative(t, y):
    if body.retarded:
        u, v, ret = decompress_ret(y) # ret is for 'retarded'
    else:
        u, v = decompress(y) # ret is for 'retarded'
        ret = None
    
    bval = [(f(t), 0),] + [(0, 0),] * (body.ndim-1)
    bval = tuple(bval)
    du_dt, dv_dt, dret_dt = body.derivative(u, v, ret, bval)  
    
    if body.retarded:
        return compress_ret(du_dt, dv_dt, dret_dt)
    return compress(du_dt, dv_dt)

def f(t):
    if t/w < 20:
        F = ampl*np.cosh(t/w)**(-2), 0*t, 0*t
    else:
        F = 0*t, 0*t, 0*t
    return np.asarray(F)[:,None,None]


def create_body_and_mesh(params_dict):
    # create body
    if BAR in params_dict[BODY_TYPE] or PLATE in params_dict[BODY_TYPE]:
        body = RectBar(params_dict)
        L, Hy, Hz = params_dict[LENGTH], params_dict[H_Y], params_dict[H_Z]
        body_dims = [[0, L], [-Hy/2, Hy/2], [-Hz/2, Hz/2]]
        quadrature = ['Lobatto',]*3
    else:
        raise NotImplementedError('not implemented for other bodies')
    # create mesh
    phys_dim = len(params_dict[DOM_NUM])
    mesh = None
    for dim in range(phys_dim):
        period = params_dict[DOM_PER]
        if np.diff(body_dims[dim]) == 0 or params_dict[DOM_PNT][dim] == 1:
            mesh_i = Constant()
        elif params_dict[DOM_NUM][dim] == 1:
            mesh_i = Legendre(params_dict[DOM_PNT][dim], endpoints=body_dims[dim], quadrature=quadrature[dim])
        else:
            domain_borders = np.linspace(body_dims[dim][0], body_dims[dim][1], params_dict[DOM_NUM][dim] + 1)
            mesh_i = Multidomain(Legendre(params_dict[DOM_PNT][dim], quadrature=quadrature[dim]), domain_borders, periodic=period[dim])
        if not isinstance(mesh, Mesh):
            mesh = mesh_i
        else: 
            mesh *= mesh_i
    return body, mesh


def print_help(argnum):
    print('List of {} arguments: material file (elastic moduli), body file (geometry and mesh),'.format(argnum), 
          'impact file (amplitude of impact and time-width), output filename, max. time and dt.')


if __name__ == "__main__":
    argnum = 7
    if 'help' in sys.argv[1]:
        print_help(argnum)
    if len(sys.argv) < argnum:
        print("Error: not enough args.")
        sys.exit()
    if len(sys.argv) > argnum:
        print("Warning: too many arguments, using only the first two of them.")

    # parse material (moduli)
    material_file = open(sys.argv[1])
    moduli_dict = parse_input_file(material_file)
    material_file.close()
    
    # parse body (geometry and mesh)
    body_file = open(sys.argv[2])
    geometry_dict = parse_input_file(body_file)
    body_file.close()
    
    # parse impact
    impact_file = open(sys.argv[3])
    impact_dict = parse_input_file(impact_file)
    impact_file.close()
    ampl = impact_dict[IMPACT_AMPL]
    w = impact_dict[IMPACT_TIME]

    # output files prefix
    simulation_name = sys.argv[4].strip()
    output_prefix = create_simulation_id() + '_' + simulation_name + '_'

    print('Simulation 3D:', simulation_name.replace('_', ' ').strip())

    # simulation times
    tmax = float(sys.argv[5])
    dt = float(sys.argv[6])
    t0 = -8*w

    # create body and mesh
    params_dict = {**moduli_dict, **geometry_dict, **impact_dict}
    body, mesh = create_body_and_mesh(params_dict)
    print('body and mesh created')

    # create output parameters file
    of = open(os.path.join(PROJ_DIR, RESULTS_DIR, output_prefix + PARAMS_FILENAME), 'w')
    params_dict[STOP_TIME] = tmax
    params_dict[TIME_STEP] = dt
    params_dict[START_TIME] = t0
    for key, value in params_dict.items():
        s = key + ' ' + VAL_SEP + ' '
        if isinstance(value, (list, np.ndarray)):
            value = list(map(str, value))
            s += (ELEM_SEP + ' ').join(value)
        else:
            s += str(value)
        s += '\n'
        of.write(s)
    of.close()
    print('input parameters saved, starting simulation...')

    # create arrays
    T = np.arange(t0, tmax + dt, dt)
    u0 = TensorField(mesh, np.zeros((3,) + mesh.shape))
    v0 = TensorField(mesh, np.zeros((3,) + mesh.shape))
    U = TensorField(mesh, np.zeros((len(T), 3) + mesh.shape))
    V = TensorField(mesh, np.zeros((len(T), 3) + mesh.shape))

    # prepare simulation
    r = sp.integrate.ode(derivative).set_integrator('dop853', rtol=1e-10, atol=1e-10, nsteps=1e6)
    if body.retarded:
        r0 = TensorField(mesh, np.zeros((len(body.tau), 3) + mesh.shape))
        r.set_initial_value(compress_ret(u0, v0, r0), t=t0)
    else:
        r.set_initial_value(compress(u0, v0), t=t0)
    
    # simulate
    #printProgressBar(0, len(T), prefix='Progress:', suffix='Complete', length=50)
    for k, t in enumerate(T):
        printProgressBar(k, len(T)-1, prefix='Progress:', suffix='Complete', length=50)
        if t > r.t:
            r.integrate(t)
        if body.retarded:
            U[k], V[k] = decompress_ret(r.y)[0:2]
        else:
            U[k], V[k] = decompress(r.y)
        # backup
        if (k > 0) and (BACKUP_ITER in params_dict) and not(k % params_dict[BACKUP_ITER]):
            np.save(os.path.join(PROJ_DIR, RESULTS_DIR, output_prefix + U_FILENAME), U.func)
            np.save(os.path.join(PROJ_DIR, RESULTS_DIR, output_prefix + V_FILENAME), V.func)
    
    # save simulation
    np.save(os.path.join(PROJ_DIR, RESULTS_DIR, output_prefix + U_FILENAME), U.func)
    np.save(os.path.join(PROJ_DIR, RESULTS_DIR, output_prefix + V_FILENAME), V.func)

    print('simulation saved, id (prefix): {}'.format(output_prefix))
