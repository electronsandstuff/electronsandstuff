from pmd_beamphysics import ParticleGroup
import matplotlib.pyplot as plt
import numpy as np
from KDEpy import FFTKDE


def phase_space_diff(beam_a: ParticleGroup, beam_b: ParticleGroup):
    pass


def plot_density_contour(
    beam: ParticleGroup,
    var_x: str,
    var_y: str,
    fig=None,
    ax=None,
    grid_size=100,
    bw="scott",
):
    """
    Plot 2D density contours of two variables from a beam object using KDE.

    Parameters
    ----------
    beam : ParticleGroup
        Beam object to plot
    var_x : str
        Variable name for x-axis
    var_y : str
        Variable name for y-axis
    fig : matplotlib.figure.Figure, optional
        Figure to plot on. If None, a new figure is created.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, a new axes is created.
    grid_size : int, optional
        Size of the grid for KDE computation. Default is 100.
    bw : float, str
        Bandwidth for KDE. If float, will use that value directly.
        If 'scott', uses Scott's rule for 2D data.
        If 'silverman', uses Silverman's rule for 2D data.
        If another string, will pass to KDEpy (but note these are optimized for 1D data).

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure containing the plot
    ax : matplotlib.axes.Axes
        Axes containing the plot
    """
    # Create figure and axes if not provided
    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))

    # Get data from beam
    x_data = beam[var_x]
    y_data = beam[var_y]

    # Calculate mean and standard deviation for standardization
    x_mean, x_std = np.mean(x_data), np.std(x_data)
    y_mean, y_std = np.mean(y_data), np.std(y_data)

    # Standardize data to have std=1 in each coordinate
    x_data_std = (x_data - x_mean) / x_std
    y_data_std = (y_data - y_mean) / y_std

    # Combine standardized data for KDE
    data_std = np.vstack([x_data_std, y_data_std]).T

    # Determine data range with 5% expansion for original data (for display)
    x_min, x_max = np.min(x_data), np.max(x_data)
    y_min, y_max = np.min(y_data), np.max(y_data)

    x_range = x_max - x_min
    y_range = y_max - y_min

    x_min -= 0.05 * x_range
    x_max += 0.05 * x_range
    y_min -= 0.05 * y_range
    y_max += 0.05 * y_range

    # Compute bandwidth if using Scott's or Silverman's rule
    if isinstance(bw, str) and bw.lower() in ["scott", "silverman"]:
        # Get number of data points
        n = len(data_std)

        # Scott's and Silverman's rules for 2D data
        # Both are n^(-1/6) * sigma for 2D data
        factor = n ** (-1 / 6)

        # For standardized data, sigma = 1
        sigma = 1.0

        # Calculate bandwidth
        bw = factor * sigma

    # Use the provided bandwidth or KDEpy's method
    kde = FFTKDE(bw=bw)

    # Fit the KDE model to the standardized data
    grid_std, points = kde.fit(data_std, beam.weight).evaluate(grid_size)

    # The grid is of shape (obs, dims), points are of shape (obs, 1)
    x_grid_std, y_grid_std = np.unique(grid_std[:, 0]), np.unique(grid_std[:, 1])

    # Transform grid points back to original scale
    x = (x_grid_std * x_std) + x_mean
    y = (y_grid_std * y_std) + y_mean

    # Reshape points for contour plotting
    z = points.reshape(grid_size, grid_size).T

    # Plot contours
    ax.contour(x, y, z, levels=10)

    # Set labels
    ax.set_xlabel(var_x)
    ax.set_ylabel(var_y)

    return fig, ax


def plot_marginal(
    beams: list[ParticleGroup],
    var: str,
    fig=None,
    ax=None,
    bins=50,
    alpha=0.7,
):
    """
    Plot histograms of a variable from multiple beam objects.

    Parameters
    ----------
    beams : list[ParticleGroup]
        List of beam objects to plot
    var : str
        Variable name to plot
    fig : matplotlib.figure.Figure, optional
        Figure to plot on. If None, a new figure is created.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, a new axes is created.
    bins : int or sequence, optional
        Number of bins or bin edges for the histograms. Default is 50.
    alpha : float, optional
        Transparency of the histograms. Default is 0.7.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure containing the plot
    ax : matplotlib.axes.Axes
        Axes containing the plot
    """
    # Create figure and axes if not provided
    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))

    # Get all data from beams
    all_data = [beam[var] for beam in beams]

    # Determine common bin range for all histograms
    min_val = min(np.min(data) for data in all_data)
    max_val = max(np.max(data) for data in all_data)

    # Expand range by 5% on both sides
    range_val = max_val - min_val
    expansion = 0.05 * range_val
    min_val -= expansion
    max_val += expansion

    # Process each beam
    fill_between_objects = []
    for i, data in enumerate(all_data):
        # Create the histogram using numpy.hist
        hist, bin_edges = np.histogram(data, bins=bins, range=(min_val, max_val))

        # Rescale histogram so it peaks at 1.0
        if np.max(hist) > 0:
            hist = hist / np.max(hist)

        # Calculate bin centers for step plotting
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        # Plot filled histogram with alpha (let matplotlib assign default color)
        fill_obj = ax.fill_between(bin_centers, hist, step="mid", alpha=alpha)
        fill_between_objects.append(fill_obj)

        # Get the color that was assigned to the fill_between object
        fill_color = fill_obj.get_facecolor()[0]  # Get the first face color

        # Plot solid line on top with the same color
        ax.step(bin_centers, hist, where="mid", color=fill_color, linewidth=1.5)

    # Set labels
    ax.set_xlabel(var)
    ax.set_ylabel("Normalized Count")

    return fig, ax
