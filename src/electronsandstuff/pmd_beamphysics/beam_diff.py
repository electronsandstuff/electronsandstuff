from pmd_beamphysics import ParticleGroup
import matplotlib.pyplot as plt
import numpy as np


def phase_space_diff(beam_a: ParticleGroup, beam_b: ParticleGroup):
    pass


def plot_marginal(
    beam_a: ParticleGroup,
    beam_b: ParticleGroup,
    var: str,
    fig=None,
    ax=None,
    bins=50,
    alpha=0.7,
):
    """
    Plot histograms of a variable from two beam objects.

    Parameters
    ----------
    beam_a : ParticleGroup
        First beam to plot
    beam_b : ParticleGroup
        Second beam to plot
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

    # Get the data
    data_a = beam_a[var]
    data_b = beam_b[var]

    # Determine common bin range for both histograms
    min_val = min(np.min(data_a), np.min(data_b))
    max_val = max(np.max(data_a), np.max(data_b))

    # Expand range by 5% on both sides
    range_val = max_val - min_val
    expansion = 0.05 * range_val
    min_val -= expansion
    max_val += expansion

    # Create the histograms using numpy.hist
    hist_a, bin_edges_a = np.histogram(data_a, bins=bins, range=(min_val, max_val))
    hist_b, bin_edges_b = np.histogram(data_b, bins=bins, range=(min_val, max_val))

    # Rescale histograms so both peak at 1.0
    if np.max(hist_a) > 0:
        hist_a = hist_a / np.max(hist_a)
    if np.max(hist_b) > 0:
        hist_b = hist_b / np.max(hist_b)

    # Calculate bin centers for step plotting
    bin_centers_a = (bin_edges_a[:-1] + bin_edges_a[1:]) / 2
    bin_centers_b = (bin_edges_b[:-1] + bin_edges_b[1:]) / 2

    # Plot filled histograms with alpha
    ax.fill_between(bin_centers_a, hist_a, step="mid", alpha=alpha, color="C0")
    ax.fill_between(bin_centers_b, hist_b, step="mid", alpha=alpha, color="C1")

    # Plot solid lines on top
    ax.step(bin_centers_a, hist_a, where="mid", color="C0", linewidth=1.5)
    ax.step(bin_centers_b, hist_b, where="mid", color="C1", linewidth=1.5)

    # Set labels
    ax.set_xlabel(var)
    ax.set_ylabel("Normalized Count")

    return fig, ax
