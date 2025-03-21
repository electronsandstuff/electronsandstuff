from pydantic import Field
from typing import Optional, Dict, Any, Union, TextIO
import re
import logging

from .base import ICoolBase
from .plotting import plot_icool_input
from .substitution import Substitution
from .region_commands import CoolingSection
from .utils import stripped_no_comment_str


logger = logging.getLogger(__name__)


class ICoolInput(ICoolBase):
    """Represents an ICOOL input file."""

    title: str = Field(default="", description="Title of the input file (max 79 chars)")
    substitutions: Dict[str, Substitution] = Field(
        default_factory=dict, description="Substitution variables defined with &SUB"
    )
    cooling_section: Optional[CoolingSection] = Field(
        default=None, description="Cooling section between SECTION and ENDSECTION"
    )

    def perform_substitutions(
        self, substitutions: Optional[Dict[str, Any]] = None
    ) -> "ICoolInput":
        """
        Create a new object with substitutions applied to all member variables.

        Parameters
        ----------
        substitutions : dict, optional
            A dictionary mapping substitution keys to their values.
            If None, constructs a dictionary from self.substitutions.

        Returns
        -------
        ICoolInput
            A new instance of ICoolInput with substitutions applied.
        """
        # If substitutions is None, construct it from self.substitutions
        if substitutions is None:
            substitutions = {}
            for name, sub_obj in self.substitutions.items():
                substitutions[name] = sub_obj.value

        # Call the parent class's perform_substitutions method
        return super().perform_substitutions(substitutions)

    @classmethod
    def from_file(cls, file_or_filename: Union[str, TextIO]) -> "ICoolInput":
        """
        Load an ICOOL input file and parse it into a pydantic structure.

        Parameters
        ----------
        file_or_filename : str or file-like object
            Path to the ICOOL input file to load, or a file-like object
            with a read() method.

        Returns
        -------
        ICoolInput
            A new instance of ICoolInput containing the parsed file content.
        """
        if isinstance(file_or_filename, str):
            # If a string is provided, treat it as a filename
            with open(file_or_filename, "r") as f:
                content = f.read()
        else:
            # Otherwise, assume it's a file-like object with a read method
            content = file_or_filename.read()

        return cls.from_str(content)

    @classmethod
    def from_str(cls, content: str) -> "ICoolInput":
        """
        Load an ICOOL input file from a string and parse it into a pydantic structure.

        Parameters
        ----------
        content : str
            String containing the ICOOL input file content.

        Returns
        -------
        ICoolInput
            A new instance of ICoolInput containing the parsed content.
        """
        lines = content.splitlines()

        # First line is the title (up to 79 characters)
        title = stripped_no_comment_str(lines[0])
        if len(title) > 79:
            title = title[:79]

        # Parse &SUB statements for variable substitutions
        substitutions = {}
        sub_pattern = re.compile(r"&SUB\s+(\w+)\s+(.*?)(?=\s*$|\s*!)")

        idx = 1
        cooling_section = None
        while idx < len(lines):
            line_stripped = stripped_no_comment_str(lines[idx])

            # Skip comments and empty lines when checking for section markers
            if not line_stripped:
                idx += 1
                continue

            # If we see a substitution
            if line_stripped.startswith("&SUB"):
                match = sub_pattern.match(line_stripped)
                if match:
                    var_name, var_value = match.groups()
                    var_value = var_value.strip()

                    substitution = Substitution(name=var_name, value=var_value)
                    substitutions[var_name] = substitution
                    logger.debug(f'Found substitution "{var_name}" -> "{var_value}"')
                idx += 1
                continue

            # Start of the cooling regions section
            if line_stripped == "SECTION":
                logger.debug("Starting to parse cooling section")
                cooling_section, idx = CoolingSection.parse_input_file(lines, idx)
                continue

            idx += 1

        return cls(
            title=title, substitutions=substitutions, cooling_section=cooling_section
        )

    def get_length(self) -> float:
        """
        Calculate the total length of the ICOOL input file.

        If the object contains substitutions, they will be performed first
        and the length will be calculated on the resulting object.

        Returns
        -------
        float
            The total length of all regions in meters.
        """
        if self.has_substitutions:
            # Perform substitutions and get length from the resulting object
            resolved_obj = self.perform_substitutions()
            return resolved_obj.get_length()

        if self.cooling_section is None:
            return 0.0

        return self.cooling_section.get_length(check_substitutions=False)

    def plot_layout(
        self, fig=None, ax=None, figsize=(6, 4), show_labels=True, rotate_labels=False
    ):
        """
        Plot the ICOOL input file elements as boxes.

        Parameters
        ----------
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

        Returns
        -------
        matplotlib.figure.Figure, matplotlib.axes.Axes
            The matplotlib figure and axis objects.
        """
        return plot_icool_input(self, fig, ax, figsize, show_labels, rotate_labels)
