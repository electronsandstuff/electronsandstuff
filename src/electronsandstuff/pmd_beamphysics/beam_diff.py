from pmd_beamphysics import ParticleGroup
import matplotlib.pyplot as plt
import numpy as np


def phase_space_diff(beam_a: ParticleGroup, beam_b: ParticleGroup):
    pass


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
