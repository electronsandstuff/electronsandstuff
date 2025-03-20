import matplotlib.pyplot as plt
import matplotlib.patches as patches
import logging
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


def plot_icool_input(icool_input, fig=None, ax=None, figsize=(6, 4), show_labels=True):
    """
    Plot the ICOOL input file elements as boxes.

    Args:
        icool_input: The ICoolInput object to plot.
        ax: Optional matplotlib axis to plot on. If None, a new figure is created.
        figsize: Figure size if creating a new figure.
        show_labels: Whether to show labels for repeats and cells.

    Returns:
        The matplotlib axis object.
    """
    # Create figure if needed
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    # If there are substitutions, resolve them first
    if icool_input.has_substitutions:
        resolved_obj = icool_input.perform_substitutions()
        return plot_icool_input(
            resolved_obj, fig=fig, ax=ax, figsize=figsize, show_labels=show_labels
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
    bbox = plot_commands(ax, icool_input.cooling_section.commands, 0, 0, show_labels)

    # Set axis properties
    ax.set_xlabel("z position (m)")
    ax.set_ylabel("r position (m)")
    ax.set_title(f"ICOOL Layout: {icool_input.title}")
    ax.grid(True, linestyle="--", alpha=0.7)

    t = bbox.upper_right[0] - bbox.lower_left[0]
    ax.set_xlim(bbox.lower_left[0] - 0.05 * t, bbox.upper_right[0] + 0.05 * t)
    t = bbox.upper_right[1] - bbox.lower_left[1]
    ax.set_ylim(bbox.lower_left[1] - 0.05 * t, bbox.upper_right[1] + 0.05 * t)

    return fig, ax


def plot_commands(ax, commands, z_start, level, show_labels):
    """
    Recursively plot commands.

    Args:
        ax: Matplotlib axis to plot on.
        commands: List of commands to plot.
        z_start: Starting z position.
        level: Nesting level for indentation.
        show_labels: Whether to show labels.

    Returns:
        The ending z position.
    """
    bbox = BoundingBox(lower_left=(z_start, 0), upper_right=(z_start, 0))
    for cmd in commands:
        if isinstance(cmd, SRegion):
            sub_bbox = plot_sregion(ax, cmd, bbox.upper_right[0], level)
        elif isinstance(cmd, Cell):
            sub_bbox = plot_cell(ax, cmd, bbox.upper_right[0], level, show_labels)
        elif isinstance(cmd, Repeat):
            sub_bbox = plot_repeat(ax, cmd, bbox.upper_right[0], level, show_labels)
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

    Args:
        ax: Matplotlib axis to plot on.
        sregion: The SRegion to plot.
        z_start: Starting z position.
        level: Nesting level for indentation.

    Returns:
        The ending z position.
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

        # Create rectangle
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


def plot_cell(ax, cell, z_start, level, show_labels):
    """
    Plot a Cell as a rectangle that encompasses its commands.

    Args:
        ax: Matplotlib axis to plot on.
        cell: The Cell to plot.
        z_start: Starting z position.
        level: Nesting level for indentation.
        show_labels: Whether to show labels.

    Returns:
        The ending z position.
    """
    # First, calculate the total length of one cell
    sum(cmd.get_length(check_substitutions=False) for cmd in cell.commands)

    # Plot the commands for the first cell
    bbox = plot_commands(ax, cell.commands, z_start, level + 1, show_labels)

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

    return bbox


def plot_repeat(ax, repeat, z_start, level, show_labels):
    """
    Plot a Repeat section as a rectangle that encompasses its commands.

    Args:
        ax: Matplotlib axis to plot on.
        repeat: The Repeat section to plot.
        z_start: Starting z position.
        level: Nesting level for indentation.
        show_labels: Whether to show labels.

    Returns:
        The ending z position.
    """
    # First, calculate the total length of one repeat
    sum(cmd.get_length(check_substitutions=False) for cmd in repeat.commands)

    # Plot the commands for the first repeat
    bbox = plot_commands(ax, repeat.commands, z_start, level + 1, show_labels)

    # Expand the box
    t1 = bbox.upper_right[1] - bbox.lower_left[1]
    bbox.lower_left = (bbox.lower_left[0], bbox.lower_left[1] - 0.05 * t1)
    bbox.upper_right = (bbox.upper_right[0], bbox.upper_right[1] + 0.05 * t1)

    # Draw a rectangle around the repeat section
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

    return bbox
