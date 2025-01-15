import datetime
import os
import numpy as np

from elastic_body import *
from util_names import *

def parse_line(line):
    l = line.split(VAL_SEP)
    if len(l) < 2:
        err_str = "Incorrect line '{}' in the input file".format(line)
        raise ValueError(err_str)
    name = l[0].strip()
    value = l[1].strip()
    return name, value

def is_comment_or_empty_line(line):
    return line.startswith(('#', '\n'))

def parse_input_file(file):
    param_dict = {}
    number_vals_list = [RHO, LAMBDA, MU, YOUNG, POISS, MURN_L, MURN_M, MURN_N,
                        MURN_NU1, MURN_NU2, MURN_NU3, MURN_NU4]
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
        elif name in [TAU, NONLIN_TAU]:
            if not RETARDED in type:
                raise ValueError("'tau' is for the retarded viscosity")
            arr = value.split(ELEM_SEP)
            value = np.array([float(el.strip()) for el in arr])
        elif name in [VISC_ETA, VISC_XI]:
            if RETARDED in type:
                arr = value.split(ELEM_SEP)
                value = np.array([float(el.strip()) for el in arr])
            else:
                value = float(value)
        elif name in [MURN_L1, MURN_M1, MURN_N1, MURN_H1]:
            rows = value.split(ROW_SEP)
            value = []
            for row in rows:
                arr = row.split(ELEM_SEP)
                value.append([float(el.strip()) for el in arr])
            value = np.array(value)
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
            #value = np.array([el.lower().strip() == 'true' for el in arr])
            value = []
            for el in arr:
                if el.lower().strip() == 'true':
                    value.append(True)
                elif el.lower().strip() == 'false':
                    value.append(False)
                else:
                    err_str = "Unknown bool value '{}' in the input file '{}'".format(el, file.name)
                    raise ValueError(err_str)
            value = np.asarray(value)
        elif name == DOM_BAS:
            arr = value.split(ELEM_SEP)
            value = []
            for el in arr:
                if el.lower().strip() == str(Legendre(1)):
                    value.append(Legendre(1))
                elif el.lower().strip() == str(Fourier(1)):
                    value.append(Fourier(1))
                else:
                    err_str = "Unknown basis '{}' in the input file '{}'".format(el, file.name)
                    raise ValueError(err_str)
            value = np.asarray(value)
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
            err_str = "Unknown parameter '{}' in the input file '{}'".format(name, file.name)
            raise ValueError(err_str)
        # save parameter
        param_dict[name] = value
    return param_dict

def save_params(file, d):
    if isinstance(file, str):
        file = open(file, 'w')
    for key, value in d.items():
        s = key + ' ' + VAL_SEP + ' '
        if isinstance(value, (list, np.ndarray)):
            if len(np.asarray(value).shape) == 1:
                value = list(map(str, value))
                s += (ELEM_SEP + ' ').join(value)
            elif len(value.shape) == 2:
                for val in value:
                    s += (ELEM_SEP + ' ').join(list(map(str, val)))
                    s += ROW_SEP + ' '
                s = s[:-(len(ROW_SEP)+1)]
            else:
                raise ValueError(f'The dimention of the {key} is greater than 2.')
        else:
            s += str(value)
        s += '\n'
        file.write(s)
    file.close()


def compress(*args):
    u, v = args[:2]
    if len(args) > 2:
        ret = args[2]
        return np.stack((u.func, v.func, *ret.func)).ravel().view(float)
    return np.stack((u.func, v.func)).ravel().view(float)

def decompress(y, mesh, ret_num=0):
    Y = y.reshape(2 + ret_num, 3, *mesh.shape)
    u = TensorField(mesh, Y[0])
    v = TensorField(mesh, Y[1])
    if ret_num > 0:
        retarded = TensorField(mesh, Y[2:])
        return u, v, retarded
    return u, v

def compress_tens_ret(u, v, ret):
    """Same as decompress function but treats retarded variables as tensors"""
    ret_func_resh = ret.func.reshape(-1, *ret.func.shape[2:])
    return np.stack((u.func, v.func, *ret_func_resh)).ravel()

def decompress_tens_ret(y, mesh, ret_num: int):
    """Same as decompress function but treats retarded variables as tensors"""
    if ret_num < 1:
        raise ValueError(f"No retarded variables: {ret_num}")
    Y = y.reshape((2 + ret_num*3, 3) + mesh.shape)
    u = TensorField(mesh, Y[0])
    v = TensorField(mesh, Y[1])
    retarded = TensorField(mesh, Y[2:].reshape((ret_num, 3, 3) + mesh.shape))
    return u, v, retarded


def create_body_and_mesh(params_dict):
    # create body
    if BAR in params_dict[BODY_TYPE]:
        body = RectBar(params_dict)
        L, Hy, Hz = params_dict[LENGTH], params_dict[H_Y], params_dict[H_Z]
        body_dims = [[0, L], [-Hy/2, Hy/2], [-Hz/2, Hz/2]]
        quadrature = ['Lobatto',]*3
    elif ROD in params_dict[BODY_TYPE]:
        body = CylindricalRod(params_dict)
        L, R = params_dict[LENGTH], params_dict[RADIUS]
        body_dims = [[0, L], [R, 0], [0, 2*np.pi]]
        quadrature = ['Lobatto', 'Radau', 'Lobatto']
    else:
        err_str = "Body '{}' is not implemented".format(params_dict[BODY_TYPE])
        raise NotImplementedError(err_str)
    # create mesh
    phys_dim = len(params_dict[DOM_NUM])
    bases = (params_dict[DOM_BAS] if DOM_BAS in params_dict 
             else [Legendre(2),]*phys_dim) # Legendre is the default basis
    mesh = None
    for dim in range(phys_dim):
        period = params_dict[DOM_PER]
        basis = type(bases[dim])
        if np.diff(body_dims[dim]) == 0 or params_dict[DOM_PNT][dim] == 1:
            mesh_i = Constant()
        elif params_dict[DOM_NUM][dim] == 1:
            mesh_i = basis(params_dict[DOM_PNT][dim], endpoints=body_dims[dim], 
                           quadrature=quadrature[dim])
        else:
            domain_borders = np.linspace(body_dims[dim][0], body_dims[dim][1], 
                                         params_dict[DOM_NUM][dim] + 1)
            mesh_i = Multidomain(basis(params_dict[DOM_PNT][dim], 
                                       quadrature=quadrature[dim]), 
                                 domain_borders, periodic=period[dim])
        if not isinstance(mesh, Mesh):
            mesh = mesh_i
        else: 
            mesh *= mesh_i
    return body, mesh


def create_simulation_id():
    return datetime.datetime.now().strftime("%Y%b%d_%H-%M-%S")


# mask mesh points
def make_mask_from_bounds(bounds: list | tuple, 
                          mesh: mesh.MeshProduct | mesh.Mesh1D) -> np.ndarray:
    if len(bounds) != mesh.ndim:
        raise ValueError("Length of 'bounds' does not match the space dimentions.")
    mask = np.zeros(mesh.shape)
    coords = mesh.grid()
    cond = True
    for d in range(mesh.ndim):
        if isinstance(bounds[d], (int, float)):
            b0, b1, = bounds[d], bounds[d]
        elif isinstance(bounds[d], (list, tuple)) and len(bounds[d]) == 2:
            b0, b1, = bounds[d]
        else:
            raise ValueError(f"Unsupported object in 'bound' at position {d}")
        new_cond = (coords[d]>=b0) & (coords[d]<=b1)
        cond = np.multiply.outer(cond, new_cond)
    mask[cond] += 1
    return mask


def load_res(pref):
    path_prefix = os.path.join(PROJ_PATH, RESULTS_DIR, pref)
    f = open(path_prefix + PARAMS_FILENAME, 'r')
    params_dict = parse_input_file(f)
    f.close()

    body, mesh = create_body_and_mesh(params_dict)
    t0, dt, tmax = params_dict[START_TIME], params_dict[TIME_STEP], \
        params_dict[STOP_TIME]
    T = np.arange(t0, tmax + dt/2, dt)
    U_func = np.load(path_prefix + U_FILENAME + '.npy')
    U = TensorField(mesh, U_func)
    return body, mesh, U, T