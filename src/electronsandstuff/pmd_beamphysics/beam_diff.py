from pmd_beamphysics import ParticleGroup
import matplotlib.pyplot as plt
import numpy as np
from KDEpy import FFTKDE


def phase_space_diff(
    beam_a: ParticleGroup,
    beam_b: ParticleGroup,
    var_x: str = "x",
    var_y: str = "px",
    grid_size: int = 100,
    bw: str = "scott",
    figsize: tuple = (6, 4),
):
    """
    Plot phase space difference between two beams with contours and marginals.

    Parameters
    ----------
    beam_a : ParticleGroup
        First beam object to plot
    beam_b : ParticleGroup
        Second beam object to plot
    var_x : str, optional
        Variable name for x-axis. Default is "x".
    var_y : str, optional
        Variable name for y-axis. Default is "px".
    grid_size : int, optional
        Size of the grid for KDE computation. Default is 100.
    bw : str or float, optional
        Bandwidth for KDE. Default is "scott".
    figsize : tuple, optional
        Figure size (width, height). Default is (10, 8).

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure containing the plot
    axes : dict
        Dictionary of axes containing the plots
    """
    # Create figure and gridspec for layout
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(
        2, 2, width_ratios=[4, 1], height_ratios=[1, 4], hspace=0.05, wspace=0.05
    )

    # Create axes
    ax_joint = fig.add_subplot(gs[1, 0])  # Main plot (bottom-left)
    ax_marg_x = fig.add_subplot(gs[0, 0], sharex=ax_joint)  # Top marginal
    ax_marg_y = fig.add_subplot(gs[1, 1], sharey=ax_joint)  # Right marginal

    # Turn off tick labels on marginals
    plt.setp(ax_marg_x.get_xticklabels(), visible=False)
    plt.setp(ax_marg_y.get_yticklabels(), visible=False)

    # Calculate a common grid for both beams with 5% expansion
    # Get data from both beams
    x_data_a = beam_a[var_x]
    y_data_a = beam_a[var_y]
    x_data_b = beam_b[var_x]
    y_data_b = beam_b[var_y]

    # Find global min/max for both variables across both beams
    x_min_global = min(np.min(x_data_a), np.min(x_data_b))
    x_max_global = max(np.max(x_data_a), np.max(x_data_b))
    y_min_global = min(np.min(y_data_a), np.min(y_data_b))
    y_max_global = max(np.max(y_data_a), np.max(y_data_b))

    # Add 5% expansion to the bounding box
    x_range_global = x_max_global - x_min_global
    y_range_global = y_max_global - y_min_global
    x_min_expanded = x_min_global - 0.05 * x_range_global
    x_max_expanded = x_max_global + 0.05 * x_range_global
    y_min_expanded = y_min_global - 0.05 * y_range_global
    y_max_expanded = y_max_global + 0.05 * y_range_global

    # Create common grid
    x_grid_common = np.linspace(x_min_expanded, x_max_expanded, grid_size)
    y_grid_common = np.linspace(y_min_expanded, y_max_expanded, grid_size)

    # Create grid points in the required order for FFTKDE
    # The grid must be sorted dimension-by-dimension (x_1, x_2, ..., x_D)
    # This creates points like: (0,0), (0,1), (0,2), (1,0), (1,1), (1,2), ...
    grid_points_common = np.array(
        [(x, y) for x in x_grid_common for y in y_grid_common]
    )

    # Calculate density for both beams using KDE
    densities = {}

    for i, beam in enumerate([beam_a, beam_b]):
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

        # Compute bandwidth if using Scott's or Silverman's rule
        if isinstance(bw, str) and bw.lower() in ["scott", "silverman"]:
            # Get number of data points
            n_points = len(data_std)

            # Scott's and Silverman's rules for 2D data
            # Both are n^(-1/6) * sigma for 2D data
            factor = n_points ** (-1 / 6)

            # For standardized data, sigma = 1
            sigma = 1.0

            # Calculate bandwidth
            bandwidth = factor * sigma
        else:
            bandwidth = bw

        # Use KDE to estimate density
        kde = FFTKDE(bw=bandwidth)

        # Scale the grid points to standardized space using broadcasting
        # Create a copy of grid_points_common and transform it
        grid_points_std = grid_points_common.copy()
        grid_points_std[:, 0] = (grid_points_std[:, 0] - x_mean) / x_std
        grid_points_std[:, 1] = (grid_points_std[:, 1] - y_mean) / y_std

        # Evaluate on our standardized grid
        density_values = kde.fit(data_std, beam.weight).evaluate(grid_points_std)

        # Reshape points for plotting
        density_grid = density_values.reshape(grid_size, grid_size)

        # Store results
        densities[i] = density_grid

    # Calculate density difference (beam_a - beam_b)
    diff_density = densities[0] - densities[1]

    # Plot density difference using pcolormesh
    # Create meshgrid for plotting
    x_grid_mesh, y_grid_mesh = np.meshgrid(x_grid_common, y_grid_common)
    ax_joint.pcolormesh(
        x_grid_mesh, y_grid_mesh, diff_density.T, cmap="coolwarm", shading="auto"
    )

    # Plot contours for both beams
    plot_density_contour(
        beam_a,
        var_x,
        var_y,
        fig=fig,
        ax=ax_joint,
        grid_size=grid_size,
        bw=bw,
        color="C0",
    )
    plot_density_contour(
        beam_b,
        var_x,
        var_y,
        fig=fig,
        ax=ax_joint,
        grid_size=grid_size,
        bw=bw,
        color="C1",
    )

    # Plot marginals
    plot_marginal([beam_a, beam_b], var_x, fig=fig, ax=ax_marg_x)
    ax_marg_x.set_xlabel("")
    ax_marg_x.set_ylabel("")
    ax_marg_x.tick_params("y", which="both", left=False, right=False, labelleft=False)

    # For y-marginal, we need to rotate the plot
    plot_marginal([beam_a, beam_b], var_y, fig=fig, ax=ax_marg_y, flip=True)
    ax_marg_x.tick_params("y", which="both", bottom=False, top=False, labelbottom=False)
    ax_marg_y.set_xlabel("")
    ax_marg_y.set_ylabel("")

    # Set main plot labels
    ax_joint.set_xlabel(var_x)
    ax_joint.set_ylabel(var_y)

    # Create a dictionary of axes for return
    axes = {"joint": ax_joint, "marginal_x": ax_marg_x, "marginal_y": ax_marg_y}

    return fig, axes


def plot_density_contour(
    beam: ParticleGroup,
    var_x: str,
    var_y: str,
    fig=None,
    ax=None,
    grid_size=100,
    bw="scott",
    color=None,
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
    color : str, tuple, or None, optional
        Color for the contour lines. If None, uses the next color from the current color cycle.

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

    # Plot contours with a single color (from cycler if not specified)
    if color is None:
        # Get the next color from the current color cycle
        # Create a dummy line to get its color, then remove it
        (line,) = ax.plot([], [])
        color = line.get_color()
        line.remove()

    ax.contour(x, y, z, levels=10, colors=color)

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
    flip=False,
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
    flip : bool, optional
        Flips x and y axes

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
        if flip:
            # For flipped axes, swap x and y in the plotting
            fill_obj = ax.fill_betweenx(bin_centers, hist, step="mid", alpha=alpha)
            fill_between_objects.append(fill_obj)

            # Get the color that was assigned to the fill_between object
            fill_color = fill_obj.get_facecolor()[0]  # Get the first face color

            # Plot solid line on top with the same color
            ax.step(hist, bin_centers, where="mid", color=fill_color, linewidth=1.5)
        else:
            # Normal orientation
            fill_obj = ax.fill_between(bin_centers, hist, step="mid", alpha=alpha)
            fill_between_objects.append(fill_obj)

            # Get the color that was assigned to the fill_between object
            fill_color = fill_obj.get_facecolor()[0]  # Get the first face color

            # Plot solid line on top with the same color
            ax.step(bin_centers, hist, where="mid", color=fill_color, linewidth=1.5)

    # Set labels
    if flip:
        ax.set_ylabel(var)
        ax.set_xlabel("Normalized Count")
    else:
        ax.set_xlabel(var)
        ax.set_ylabel("Normalized Count")

    return fig, ax
