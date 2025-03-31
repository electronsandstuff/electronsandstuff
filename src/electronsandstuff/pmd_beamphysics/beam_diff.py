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

    # Create the histograms
    ax.hist(
        data_a,
        bins=bins,
        range=(min_val, max_val),
        color="C0",
        alpha=alpha,
        edgecolor="C0",
    )
    ax.hist(
        data_b,
        bins=bins,
        range=(min_val, max_val),
        color="C1",
        alpha=alpha,
        edgecolor="C1",
    )

    # Set labels
    ax.set_xlabel(var)
    ax.set_ylabel("Count")

    return fig, ax
