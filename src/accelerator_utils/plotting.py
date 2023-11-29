import matplotlib.pyplot as plt
import matplotlib.animation
from scipy import interpolate
import numpy as np

from .file_interfaces import get_density_series_bounding_box, pixel_coords_from_row


def make_density_animation(tables, files, output="", idx=0, bound_scale=1, screen_table='screens', param_table='data', timelike_key='spos', coords=['r', 'vr'], dpi=300, titlesize=16):
    """
    Generates an animation from matplotlib of density in the tables / files along the timelike coordinate.

    Args:
        tables (dict): the loaded tables
        files (dict): the loaded files
        output (str, optional): A location to save the animation as a file. Defaults to "".
        idx (int, optional): The index of the point the screen outputs are associated with. Defaults to 0.
        bound_scale (float, optional): Scaling factor for bounding box. Defaults to 1.
        screen_table (str, optional): name of the table containing the screen objects with densities. Defaults to 'screens'.
        param_table (str, optional): name of the table containing parameters associated with the screens. Defaults to 'data'.
        timelike_key (str, optional): name of the timelike coordinate in the screen table. Defaults to 'spos'.
        coords (list, optional): list of the coordinate names. Defaults to ['r', 'vr'].
        dpi (int, optional): dpi of the saved animation. Defaults to 300.
        titlesize (float, optional): font size of the title. Defaults to 16.

    Returns:
        animation object: the matplotlib animation
    """
    df = tables[screen_table][tables[screen_table][f'{param_table}_idx'] == idx].sort_values(timelike_key)

    # Setup the plot
    fig, ax = plt.subplots(facecolor='w', layout='constrained')
    p = [ax.contourf(np.zeros((2,2)), np.zeros((2,2)), np.zeros((2,2)), antialiased=True)]
    p[0].set_edgecolor("face")
    plt.gca().set_aspect('equal')
    
    # Make labels
    labels = {
        'r': 'Radius (mm)',
        'vr':'Radial Velocity (mrad)',
        'x': 'x (mm)',
        'y': 'y (mm)',
        'vx': 'x Velocity (mrad)',
        'vy': 'y Velocity (mrad)',
        'z': 'z (mm)'
    }
    plt.xlabel(labels.get(coords[0], ""))
    plt.ylabel(labels.get(coords[1], ""))
    
    # Find the bounding box for the images
    mins, maxes = get_density_series_bounding_box(df)
    bounds = {k: bound_scale*max((abs(mins[k]), abs(maxes[k]))) for k in mins}

    # The callback function for matplotlib
    def update(idx):
        p[0].remove()
        row = df.iloc[idx]
        x, y = pixel_coords_from_row(row, files)
        xinterp, yinterp = np.mgrid[-bounds['r']:bounds['r']:256j, -bounds['vr']:bounds['vr']:256j]
        ypred = interpolate.griddata(np.vstack((x.ravel(), y.ravel())).T, files[row['rho']].ravel(), np.vstack((xinterp.ravel(), yinterp.ravel())).T, method='linear', fill_value=0.0)
        ypred = np.reshape(ypred, xinterp.shape)
        p[0] = ax.contourf(1e3*xinterp, 1e3*yinterp, ypred)
        ax.set_title('s = %.2f m' % row['spos'], fontdict={'size': titlesize})
        return p

    ani = matplotlib.animation.FuncAnimation(fig, update, frames=len(df), interval=1/7*1000, blit=True, repeat=True)
    if output:
        ani.save(output, dpi=dpi)
    return ani
