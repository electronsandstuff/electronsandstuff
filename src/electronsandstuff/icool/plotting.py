import matplotlib.pyplot as plt
import matplotlib.patches as patches
import logging
import numpy as np
from dataclasses import dataclass
from typing import Tuple

from .fields import FieldAccel2, FieldAccel10
from .region_commands import SRegion, Cell, Repeat

logger = logging.getLogger(__name__)

# Define material colors
MATERIAL_COLORS = {
    # Vacuum
    "VAC": "white",
    # Gases
    "GH": "lightpink",  # Gaseous hydrogen
    "GHE": "lightskyblue",  # Gaseous helium
    # Liquids
    "LH": "pink",  # Liquid hydrogen
    "LHE": "skyblue",  # Liquid helium
    "LI": "lightcoral",  # Lithium
    # Metals - various shades of gray
    "BE": "#D3D3D3",  # Beryllium - light gray
    "B": "#C0C0C0",  # Boron - silver
    "C": "#A9A9A9",  # Carbon - dark gray
    "AL": "#A8A8A8",  # Aluminum - gray
    "TI": "#909090",  # Titanium - darker gray
    "FE": "#808080",  # Iron - gray
    "CU": "#CD7F32",  # Copper - copper color
    "W": "#696969",  # Tungsten - dim gray
    "HG": "#A9A9A9",  # Mercury - dark gray
    "PB": "#778899",  # Lead - light slate gray
    "AM": "#B0B0B0",  # AlBeMet - light gray
    # Compounds
    "LIH": "lightpink",  # Lithium hydride
    "CH2": "#90EE90",  # Polyethylene - light green
    "SS": "#708090",  # Stainless steel - slate gray
}


@dataclass
class BoundingBox:
    lower_left: Tuple[float, float]
    upper_right: Tuple[float, float]

    def __add__(self, other: "BoundingBox"):
        return BoundingBox(
            lower_left=(
                min(self.lower_left[0], other.lower_left[0]),
                min(self.lower_left[1], other.lower_left[1]),
            ),
            upper_right=(
                max(self.upper_right[0], other.upper_right[0]),
                max(self.upper_right[1], other.upper_right[1]),
            ),
        )

    @property
    def width(self):
        return self.upper_right[0] - self.lower_left[0]

    @property
    def height(self):
        return self.upper_right[1] - self.lower_left[1]


def plot_icool_input(
    icool_input,
    fig=None,
    ax=None,
    figsize=(6, 4),
    show_labels=True,
    rotate_labels=False,
    expand_repeats=False,
    expand_cells=False,
):
    """
    Plot the ICOOL input file elements as boxes.

    Parameters
    ----------
    icool_input : ICoolInput
        The ICoolInput object to plot.
    fig : matplotlib.figure.Figure, optional
        Figure to plot on. If None, a new figure is created.
    ax : matplotlib.axes.Axes, optional
        Matplotlib axis to plot on. If None, a new figure is created.
    figsize : tuple of float, optional
        Figure size if creating a new figure, default (6, 4).
    show_labels : bool, optional
        Whether to show labels for repeats and cells, default True.
    rotate_labels : bool, optional
        Whether to rotate labels 90 degrees, default False.
    expand_repeats : bool, optional
        Whether to expand repeat sections, plotting all repeats subsequently instead of a single cell, default False.
    expand_cells : bool, optional
        Whether to expand cells, plotting all cells subsequently instead of a single cell, default False.

    Returns
    -------
    matplotlib.figure.Figure, matplotlib.axes.Axes
        The matplotlib figure and axis objects.
    """
    # Create figure if needed
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    # If there are substitutions, resolve them first
    if icool_input.has_substitutions:
        resolved_obj = icool_input.perform_substitutions()
        return plot_icool_input(
            resolved_obj,
            fig=fig,
            ax=ax,
            figsize=figsize,
            show_labels=show_labels,
            rotate_labels=rotate_labels,
            expand_repeats=expand_repeats,
            expand_cells=expand_cells,
        )

    if icool_input.cooling_section is None:
        ax.text(
            0.5,
            0.5,
            "No cooling section defined",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        return ax

    # Plot the cooling section
    bbox = plot_commands(
        ax,
        icool_input.cooling_section.commands,
        0,
        0,
        show_labels,
        rotate_labels,
        expand_repeats,
        expand_cells,
    )

    # Set axis properties
    ax.set_xlabel(r"$z$ Position (m)")
    ax.set_ylabel(r"$r$ Position (m)")
    ax.set_title(f"ICOOL Layout: {icool_input.title}")
    ax.grid(True, linestyle="--", alpha=0.7)

    t = bbox.upper_right[0] - bbox.lower_left[0]
    ax.set_xlim(bbox.lower_left[0] - 0.05 * t, bbox.upper_right[0] + 0.05 * t)
    t = bbox.upper_right[1] - bbox.lower_left[1]
    ax.set_ylim(bbox.lower_left[1] - 0.05 * t, bbox.upper_right[1] + 0.05 * t)

    return fig, ax


def plot_commands(
    ax,
    commands,
    z_start,
    level,
    show_labels,
    rotate_labels=False,
    expand_repeats=False,
    expand_cells=False,
):
    """
    Recursively plot commands.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Matplotlib axis to plot on.
    commands : list
        List of commands to plot.
    z_start : float
        Starting z position.
    level : int
        Nesting level for indentation.
    show_labels : bool
        Whether to show labels.
    rotate_labels : bool, optional
        Whether to rotate labels 90 degrees, default False.
    expand_repeats : bool, optional
        Whether to expand repeat sections, default False.
    expand_cells : bool, optional
        Whether to expand cells, default False.

    Returns
    -------
    BoundingBox
        Bounding box containing the plotted commands.
    """
    bbox = BoundingBox(lower_left=(z_start, 0), upper_right=(z_start, 0))
    for cmd in commands:
        if isinstance(cmd, SRegion):
            sub_bbox = plot_sregion(ax, cmd, bbox.upper_right[0], level)
        elif isinstance(cmd, Cell):
            sub_bbox = plot_cell(
                ax,
                cmd,
                bbox.upper_right[0],
                level,
                show_labels,
                rotate_labels,
                expand_cells,
                expand_repeats,
            )
        elif isinstance(cmd, Repeat):
            sub_bbox = plot_repeat(
                ax,
                cmd,
                bbox.upper_right[0],
                level,
                show_labels,
                rotate_labels,
                expand_repeats,
                expand_cells,
            )
        else:
            # Skip other command types for now
            sub_bbox = BoundingBox(
                lower_left=(bbox.upper_right[0], 0),
                upper_right=(
                    bbox.upper_right[0] + cmd.get_length(check_substitutions=False),
                    0,
                ),
            )
        bbox = bbox + sub_bbox

    return bbox


def plot_sregion(ax, sregion, z_start, level):
    """
    Plot an SRegion as a box.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Matplotlib axis to plot on.
    sregion : SRegion
        The SRegion to plot.
    z_start : float
        Starting z position.
    level : int
        Nesting level for indentation.

    Returns
    -------
    BoundingBox
        Bounding box containing the plotted SRegion.
    """
    z_length = sregion.slen
    z_end = z_start + z_length

    # Plot each subregion
    r_max = 0
    for subregion in sregion.subregions:
        r_low = subregion.rlow
        r_high = subregion.rhigh
        r_max = max(r_high, r_max)

        # Determine the color based on field type and material
        color = MATERIAL_COLORS.get(subregion.mtag, "gray")

        # Override color for accelerating cavities
        if isinstance(subregion.field, (FieldAccel2, FieldAccel10)):
            color = "maroon"

        # If r_low is close to 0, plot one merged box instead of two separate boxes
        if np.isclose(r_low, 0):
            # Create a single rectangle from -r_high to r_high
            rect = patches.Rectangle(
                (z_start, -r_high),
                z_length,
                2 * r_high,  # Total height is 2*r_high
                linewidth=1,
                edgecolor="black",
                facecolor=color,
                alpha=0.7,
                transform=ax.transData,
            )
            ax.add_patch(rect)
        else:
            # Create two separate rectangles as before
            rect = patches.Rectangle(
                (z_start, r_low),
                z_length,
                r_high - r_low,
                linewidth=1,
                edgecolor="black",
                facecolor=color,
                alpha=0.7,
                transform=ax.transData,
            )
            ax.add_patch(rect)

            rect = patches.Rectangle(
                (z_start, -r_high),
                z_length,
                r_high - r_low,
                linewidth=1,
                edgecolor="black",
                facecolor=color,
                alpha=0.7,
                transform=ax.transData,
            )
            ax.add_patch(rect)

    return BoundingBox(lower_left=(z_start, -r_max), upper_right=(z_end, r_max))


def plot_cell(
    ax,
    cell,
    z_start,
    level,
    show_labels,
    rotate_labels=False,
    expand_cells=False,
    expand_repeats=False,
):
    """
    Plot a Cell as a rectangle that encompasses its commands.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Matplotlib axis to plot on.
    cell : Cell
        The Cell to plot.
    z_start : float
        Starting z position.
    level : int
        Nesting level for indentation.
    show_labels : bool
        Whether to show labels.
    rotate_labels : bool, optional
        Whether to rotate labels 90 degrees, default False.
    expand_cells : bool, optional
        Whether to expand cells, plotting all cells subsequently instead of a single cell, default False.
    expand_repeats : bool, optional
        Whether to expand repeat sections, default False.

    Returns
    -------
    BoundingBox
        Bounding box containing the plotted Cell.
    """
    # First, calculate the total length of one cell
    cell_length = sum(
        cmd.get_length(check_substitutions=False) for cmd in cell.commands
    )

    if not expand_cells or cell.n_cells <= 1:
        # Plot the commands for the first cell
        bbox = plot_commands(
            ax,
            cell.commands,
            z_start,
            level + 1,
            show_labels,
            rotate_labels,
            expand_repeats,
            expand_cells,
        )
    else:
        # Plot all cells subsequently
        bbox = BoundingBox(lower_left=(z_start, 0), upper_right=(z_start, 0))
        for i in range(cell.n_cells):
            cell_start = z_start + i * cell_length
            cell_bbox = plot_commands(
                ax,
                cell.commands,
                cell_start,
                level + 1,
                show_labels,
                rotate_labels,
                expand_repeats,
                expand_cells,
            )
            bbox = bbox + cell_bbox

    # Expand the box
    t1 = bbox.upper_right[1] - bbox.lower_left[1]
    bbox.lower_left = (bbox.lower_left[0], bbox.lower_left[1] - 0.05 * t1)
    bbox.upper_right = (bbox.upper_right[0], bbox.upper_right[1] + 0.05 * t1)

    # Draw a rectangle around the cell
    rect = patches.Rectangle(
        bbox.lower_left,
        bbox.width,
        bbox.height,
        linewidth=1.5,
        edgecolor="blue",
        facecolor="none",
        linestyle="--",
        transform=ax.transData,
    )
    ax.add_patch(rect)

    # Add label for number of cells if requested and not expanded
    if show_labels and cell.n_cells > 1 and not expand_cells:
        label_text = f"{cell.n_cells} cells"
        # Vertical label (rotated 90 degrees)
        ax.text(
            bbox.upper_right[0] - 0.05 * bbox.width,
            bbox.upper_right[1] - 0.01 * bbox.height,
            label_text,
            ha="right",
            va="top",
            color="blue",
            fontsize=9,
            bbox=dict(facecolor="white", alpha=0.7, pad=2),
            rotation=-90 if rotate_labels else 0,
        )

    return bbox


def plot_repeat(
    ax,
    repeat,
    z_start,
    level,
    show_labels,
    rotate_labels=False,
    expand_repeats=False,
    expand_cells=False,
):
    """
    Plot a Repeat section as a rectangle that encompasses its commands.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Matplotlib axis to plot on.
    repeat : Repeat
        The Repeat section to plot.
    z_start : float
        Starting z position.
    level : int
        Nesting level for indentation.
    show_labels : bool
        Whether to show labels.
    rotate_labels : bool, optional
        Whether to rotate labels 90 degrees, default False.
    expand_repeats : bool, optional
        Whether to expand repeat sections, plotting all repeats subsequently instead of a single repeat, default False.
    expand_cells : bool, optional
        Whether to expand cells, default False.

    Returns
    -------
    BoundingBox
        Bounding box containing the plotted Repeat section.
    """
    # First, calculate the total length of one repeat
    repeat_length = sum(
        cmd.get_length(check_substitutions=False) for cmd in repeat.commands
    )

    if not expand_repeats or repeat.n_repeat <= 1:
        # Plot the commands for the first repeat
        bbox = plot_commands(
            ax,
            repeat.commands,
            z_start,
            level + 1,
            show_labels,
            rotate_labels,
            expand_repeats,
            expand_cells,
        )
    else:
        # Plot all repeats subsequently
        bbox = BoundingBox(lower_left=(z_start, 0), upper_right=(z_start, 0))
        for i in range(repeat.n_repeat):
            repeat_start = z_start + i * repeat_length
            repeat_bbox = plot_commands(
                ax,
                repeat.commands,
                repeat_start,
                level + 1,
                show_labels,
                rotate_labels,
                expand_repeats,
                expand_cells,
            )
            bbox = bbox + repeat_bbox

    # Expand the box
    t1 = bbox.upper_right[1] - bbox.lower_left[1]
    bbox.lower_left = (bbox.lower_left[0], bbox.lower_left[1] - 0.05 * t1)
    bbox.upper_right = (bbox.upper_right[0], bbox.upper_right[1] + 0.05 * t1)

    # Draw a rectangle around the repeat section only if not expanded
    if not expand_repeats or repeat.n_repeat <= 1:
        rect = patches.Rectangle(
            bbox.lower_left,
            bbox.width,
            bbox.height,
            linewidth=1.5,
            edgecolor="green",
            facecolor="none",
            linestyle="-.",
            transform=ax.transData,
        )
        ax.add_patch(rect)

    # Add label for number of repeats if requested and not expanded
    if show_labels and repeat.n_repeat > 1 and not expand_repeats:
        label_text = f"{repeat.n_repeat} repeats"
        ax.text(
            bbox.upper_right[0] - 0.02 * bbox.width,
            bbox.upper_right[1] - 0.02 * bbox.height,
            label_text,
            ha="right",
            va="top",
            color="green",
            fontsize=9,
            bbox=dict(facecolor="white", alpha=0.7, pad=2),
            rotation=-90 if rotate_labels else 0,
        )

    return bbox
