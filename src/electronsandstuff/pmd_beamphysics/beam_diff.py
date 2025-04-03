from pmd_beamphysics import ParticleGroup
import matplotlib.pyplot as plt
import numpy as np
from KDEpy import FFTKDE
from typing import Union, Tuple, Optional


def calculate_density_kde(
    beam: ParticleGroup,
    var_x: str,
    var_y: str,
    grid_size: Union[int, Tuple[int, int]] = 100,
    bw: Union[str, float] = "scott",
    grid_points: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate 2D density of two variables from a beam object using KDE.

    Parameters
    ----------
    beam : ParticleGroup
        Beam object to calculate density for
    var_x : str
        Variable name for x-axis
    var_y : str
        Variable name for y-axis
    grid_size : int or tuple, optional
        Size of the grid for KDE computation. If int, creates a square grid.
        If tuple, creates a grid with (x_size, y_size). Default is 100.
    bw : str or float, optional
        Bandwidth for KDE. If float, will use that value directly.
        If 'scott', uses Scott's rule for 2D data.
        If 'silverman', uses Silverman's rule for 2D data.
        If another string, will pass to KDEpy.
    grid_points : np.ndarray, optional
        Custom grid points to evaluate density on. If provided, grid_size is ignored.
        Should be of shape (n_points, 2) where each row is (x, y).

    Returns
    -------
    x : np.ndarray
        Grid points for x-axis in original scale
    y : np.ndarray
        Grid points for y-axis in original scale
    z : np.ndarray
        Density values on the grid, shape matches the grid
    """
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

    # Fit the KDE model to the standardized data
    kde_fitted = kde.fit(data_std, beam.weight)

    # Handle different grid evaluation options
    if grid_points is not None:
        # Transform provided grid points to standardized space
        grid_points_std = grid_points.copy()
        grid_points_std[:, 0] = (grid_points_std[:, 0] - x_mean) / x_std
        grid_points_std[:, 1] = (grid_points_std[:, 1] - y_mean) / y_std

        # Evaluate on the provided grid
        density_values = kde_fitted.evaluate(grid_points_std)

        # Extract unique x and y values from the grid
        x = np.unique(grid_points[:, 0])
        y = np.unique(grid_points[:, 1])

        # Reshape density values to match grid dimensions
        z = density_values.reshape(len(x), len(y)).T
    else:
        # Handle grid_size as int or tuple
        if isinstance(grid_size, int):
            x_grid_size = y_grid_size = grid_size
        else:
            x_grid_size, y_grid_size = grid_size

        # Evaluate on a regular grid
        grid_std, points = kde_fitted.evaluate((x_grid_size, y_grid_size))

        # Extract unique x and y values from the standardized grid
        x_grid_std = np.unique(grid_std[:, 0])
        y_grid_std = np.unique(grid_std[:, 1])

        # Transform grid points back to original scale
        x = (x_grid_std * x_std) + x_mean
        y = (y_grid_std * y_std) + y_mean

        # Reshape points for plotting
        z = points.reshape(x_grid_size, y_grid_size).T

    return x, y, z


def joint_and_marginal_diff(
    beam_a: ParticleGroup,
    beam_b: ParticleGroup,
    var_x: str = "x",
    var_y: str = "px",
    grid_size: int = 100,
    bw: str = "scott",
    figsize: tuple = (6, 4),
    label_a: str = "",
    label_b: str = "",
    fig: Optional[plt.Figure] = None,
    axes: Optional[dict] = None,
):
    """
    Plot joint and marginal difference between two variables in two beams.

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
        Figure size (width, height). Default is (6, 4).
    label_a : str, optional
        Label associated with beam_a for legend
    label_b : str, optional
        Label associated with beam_b for legend
    fig : matplotlib.figure.Figure, optional
        Figure to plot on. If None, a new figure is created.
    axes : dict, optional
        Dictionary of axes to plot on. If None, new axes are created.
        Should contain keys 'joint', 'marginal_x', and 'marginal_y'.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure containing the plot
    axes : dict
        Dictionary of axes containing the plots
    """
    # Create figure and axes if not provided
    if fig is None or axes is None:
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

        # Create a dictionary of axes for return
        axes = {"joint": ax_joint, "marginal_x": ax_marg_x, "marginal_y": ax_marg_y}
    else:
        # Use provided axes
        ax_joint = axes["joint"]
        ax_marg_x = axes["marginal_x"]
        ax_marg_y = axes["marginal_y"]

    # Find global min/max for both variables across both beams
    x_min = min(np.min(beam_a[var_x]), np.min(beam_b[var_x]))
    x_max = max(np.max(beam_a[var_x]), np.max(beam_b[var_x]))
    y_min = min(np.min(beam_a[var_y]), np.min(beam_b[var_y]))
    y_max = max(np.max(beam_a[var_y]), np.max(beam_b[var_y]))

    # Add 5% expansion to the bounding box
    tx = x_max - x_min
    ty = y_max - y_min
    x_min = x_min - 0.05 * tx
    x_max = x_max + 0.05 * tx
    y_min = y_min - 0.05 * ty
    y_max = y_max + 0.05 * ty

    # Create common grid
    x_grid_common = np.linspace(x_min, x_max, grid_size)
    y_grid_common = np.linspace(y_min, y_max, grid_size)

    # Create grid points in the required order for FFTKDE
    # The grid must be sorted dimension-by-dimension (x_1, x_2, ..., x_D)
    # This creates points like: (0,0), (0,1), (0,2), (1,0), (1,1), (1,2), ...
    grid_points_common = np.array(
        [(x, y) for x in x_grid_common for y in y_grid_common]
    )

    # Calculate density for both beams using KDE
    densities = {}

    for i, beam in enumerate([beam_a, beam_b]):
        # Use the common calculate_density_kde function with the common grid
        _, _, density_grid = calculate_density_kde(
            beam=beam, var_x=var_x, var_y=var_y, bw=bw, grid_points=grid_points_common
        )

        # Store results - reshape to match the expected grid size
        densities[i] = density_grid.reshape(grid_size, grid_size)

    # Calculate density difference (beam_a - beam_b)
    diff_density = densities[0] - densities[1]

    # Plot density difference using pcolormesh
    x_grid_mesh, y_grid_mesh = np.meshgrid(x_grid_common, y_grid_common)
    vmax = max(abs(np.min(diff_density)), abs(np.max(diff_density)))
    ax_joint.pcolormesh(
        x_grid_mesh,
        y_grid_mesh,
        diff_density,
        cmap="seismic",
        vmin=-vmax,
        vmax=vmax,
        shading="auto",
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

    # Deal with legend
    if label_a or label_b:
        ax_joint.plot([], [], c="C0", label=label_a)
        ax_joint.plot([], [], c="C1", label=label_b)
        ax_joint.legend()

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

    # Use the common calculate_density_kde function
    x, y, z = calculate_density_kde(
        beam=beam, var_x=var_x, var_y=var_y, grid_size=grid_size, bw=bw
    )

    # Plot contours with a single color (from cycler if not specified)
    if color is None:
        # Get the next color from the current color cycle
        # Create a dummy line to get its color, then remove it
        (line,) = ax.plot([], [])
        color = line.get_color()
        line.remove()

    ax.contour(x, y, z, levels=10, colors=color, linewidths=1, alpha=0.8)

    # Set labels
    ax.set_xlabel(var_x)
    ax.set_ylabel(var_y)

    return fig, ax


def phase_space_diff(
    beam_a: ParticleGroup,
    beam_b: ParticleGroup,
    grid_size: int = 100,
    bw: str = "scott",
    figsize: tuple = (18, 6),
    label_a: str = "",
    label_b: str = "",
):
    """
    Plot three phase space difference plots side by side in one row.
    Each plot includes joint and marginal distributions.

    Parameters
    ----------
    beam_a : ParticleGroup
        First beam object to plot
    beam_b : ParticleGroup
        Second beam object to plot
    grid_size : int, optional
        Size of the grid for KDE computation. Default is 100.
    bw : str or float, optional
        Bandwidth for KDE. Default is "scott".
    figsize : tuple, optional
        Figure size (width, height). Default is (18, 6).
    label_a : str, optional
        Label associated with beam_a for legend
    label_b : str, optional
        Label associated with beam_b for legend

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure containing the plots
    axes : dict
        Dictionary of axes containing the plots
    """
    # Define the variable pairs for each subplot
    var_pairs = [("x", "px"), ("y", "py"), ("delta_t", "delta_energy")]

    # Create a figure
    fig = plt.figure(figsize=figsize)

    # Create a top-level gridspec with 3 columns (one for each variable pair)
    # with wider spacing between pairs
    top_gs = fig.add_gridspec(1, 3, wspace=0.25)

    # Create a dictionary to store all axes
    all_axes = {}

    # Create three joint+marginal plots
    for i, (var_x, var_y) in enumerate(var_pairs):
        # Create a nested gridspec for each variable pair with tighter spacing
        # between the density plot and its marginal
        nested_gs = top_gs[0, i].subgridspec(
            2,
            2,
            width_ratios=[4, 1],  # Main plot, y-marginal
            height_ratios=[1, 4],  # x-marginal, main plot
            hspace=0.05,  # Tight spacing between x-marginal and joint plot
            wspace=0.05,  # Tight spacing between joint plot and y-marginal
        )

        # Create axes for this plot
        ax_joint = fig.add_subplot(nested_gs[1, 0])  # Main plot
        ax_marg_x = fig.add_subplot(nested_gs[0, 0], sharex=ax_joint)  # Top marginal
        ax_marg_y = fig.add_subplot(nested_gs[1, 1], sharey=ax_joint)  # Right marginal

        # Turn off tick labels on marginals
        plt.setp(ax_marg_x.get_xticklabels(), visible=False)
        plt.setp(ax_marg_y.get_yticklabels(), visible=False)

        # Store axes in dictionary
        plot_axes = {
            "joint": ax_joint,
            "marginal_x": ax_marg_x,
            "marginal_y": ax_marg_y,
        }

        # Add these axes to the all_axes dictionary
        all_axes[f"{var_x}_{var_y}"] = plot_axes

        # Call joint_and_marginal_diff with these axes
        # Only add legend to the first plot
        use_label_a = label_a if i == 0 else ""
        use_label_b = label_b if i == 0 else ""

        # Call joint_and_marginal_diff with the prepared axes
        _, _ = joint_and_marginal_diff(
            beam_a=beam_a,
            beam_b=beam_b,
            var_x=var_x,
            var_y=var_y,
            grid_size=grid_size,
            bw=bw,
            label_a=use_label_a,
            label_b=use_label_b,
            fig=fig,
            axes=plot_axes,
        )

    # Adjust layout
    fig.tight_layout()

    return fig, all_axes


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
