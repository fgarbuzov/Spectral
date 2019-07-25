import matplotlib.pyplot as plt
from matplotlib.image import NonUniformImage
from matplotlib.colors import Normalize
import numpy as np

__all__ = ['error_plot', 'heatmap']

def error_plot(x, y, dy, color=None, sigma=2, fill_alpha=0.3, **kwargs):
    """Plot line with an error as a shaded region.
    
    Parameters
    ----------
    x: array
        An array of x coordinates
    y: array
        An array of y coordinates
    dy: array
        An array of the standard deviations of `y`
    color: matplotlib color
        Color of the plot. The default is None, which means the default
        matplotlib color
    sigma: float
        How much standard deviations show. The default value is 2.
    fill_alpha: float
        Alpha of the filling.
    All other options will pe passed to `matplotlib.pyplot.plot`.
    """
    line, = plt.plot(x, y, color=color, **kwargs)
    if color is None:
        color = line.get_color()
    plt.fill_between(x, y - sigma*dy, y + sigma*dy, 
                     color=color, alpha=fill_alpha)
    
    

def heatmap(X, Y, V, xlim=None, ylim=None, vmin=None, vmax=None, cmap=None,
            interpolation='bilinear'):
    """Plot heatmap.
    
    Parameters
    ----------
    X: array
        ...
    Y: array
        ...
    V: array
        ...
    xlim: (float or None, float or None)
        ...
    ylim: (float or None, float or None)
        ...
    vmin: float or None
        ...
    vmax: float or None
        ...
    cmap: matplotlib colormap
        ...
    
    """
    ax = plt.gca()
    if xlim is None:
        xlim = None, None
    if ylim is None:
        ylim = None, None
    minmax = np.min, np.max
    xlim = tuple(f(X) if l is None else l for f, l in zip(minmax, xlim))
    ylim = tuple(f(Y) if l is None else l for f, l in zip(minmax, ylim))
    plt.xlim(xlim)
    plt.ylim(ylim)
    if X.ndim == 2 and Y.ndim == 2:
        if X.shape[1] == 1 and Y.shape[0] == 1:
            X = X.ravel()
            Y = Y.ravel()
        elif X.shape[0] == 1 and Y.shape[1] == 1:
            X = X.ravel()
            Y = Y.ravel()
            V = V.T
    if X.ndim == 1 and Y.ndim == 1:
        if (xlim[0] < xlim[1]) != (X[0] < X[-1]):
            X = X[::-1]
            V = V[::-1]
        if (ylim[0] < ylim[1]) != (Y[0] < Y[-1]):
            Y = Y[::-1]
            V = V[:,::-1]
        im = NonUniformImage(ax, interpolation=interpolation, cmap=cmap,
                             norm=Normalize(vmin, vmax),
                             extent=xlim + ylim)
        im.set_data(X, Y, V.T)
        ax.images.append(im)
        plt.sci(im)
    else:
        X, Y = np.broadcast_arrays(X, Y)
        if interpolation == 'nearest':
            shading = 'flat'
        if interpolation == 'bilinear':
            shading = 'gouraud'
        plt.pcolormesh(X, Y, V, vmin=vmin, vmax=vmax, cmap=cmap,
                       shading=shading, rasterized=True)
