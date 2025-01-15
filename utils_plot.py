import matplotlib.pyplot as plt
import numpy as np

from util_names import *

def plot_deformations(fig, ax, U, T, times_to_plot, diff=None, x=None,
                      vmin=None, vmax=None):
    if isinstance(times_to_plot, int):
        times_to_plot = [times_to_plot]
    if len(times_to_plot) == 1:
        ax = [ax]

    x_mesh, y_mesh = U.mesh.grid()[0:2]
    if not isinstance(x, np.ndarray):
        x = np.linspace(x_mesh[0], x_mesh[-1], num=1001, endpoint=True)
    y = np.linspace(y_mesh[0], y_mesh[-1], num=101, endpoint=True)
    if diff or diff==0:
        u = U.diff(diff).real
    else:
        u = U
    
    for i in range(len(times_to_plot)):
        im = ax[i].imshow(np.flip(u[times_to_plot[i]](x,y,0).T, axis=0),
                          extent=(x.min(), x.max(), y.min(), y.max()), 
                          vmin=vmin, vmax=vmax)
        ax[i].set_aspect('equal')
        ax[i].set_ylabel('y, mm')
        #ax[i].set_title('t = %.1f $\mu$s' % T[t[i]], fontsize=11)
        fig.colorbar(im, ax=ax[i], aspect=5, pad=0.015)
        ax[i].text(0.875, 0.6, ('(' + 'abcdefgh'[i] + ') ' 
                                + f't = {int(T[times_to_plot[i]])} $\mu$s'), 
                   c='w', transform=ax[i].transAxes, fontsize=12)
    
    ax[-1].set_xlabel('x, mm')
    fig.set_tight_layout(True)