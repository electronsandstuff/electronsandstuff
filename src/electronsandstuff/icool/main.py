from pydantic import BaseModel, Field, validator
from typing import Optional, List, Dict, Union, Any, Literal, Tuple, Annotated
import re
import logging

from .substitution import Substitution
from .region_commands import CoolingSection, parse_region_cmds
from .utils import stripped_no_comment_str


logger = logging.getLogger(__name__)


class ICOOLInput(BaseModel):
    """Represents an ICOOL input file."""
    title: str = Field(default="", description="Title of the input file (max 79 chars)")
    substitutions: Dict[str, Substitution] = Field(
        default_factory=dict,
        description="Substitution variables defined with &SUB"
    )
    cooling_section: Optional[CoolingSection] = Field(
        default=None,
        description="Cooling section between SECTION and ENDSECTION"
    )
    
    @classmethod
    def from_file(cls, filename: str) -> 'ICOOLInput':
        """Load an ICOOL input file and parse it into a pydantic structure."""
        with open(filename, 'r') as f:
            content = f.read()
        
        return cls.from_str(content)
    
    @classmethod
    def from_str(cls, content: str) -> 'ICOOLInput':
        """Load an ICOOL input file from a string and parse it into a pydantic structure."""
        lines = content.splitlines()
        
        # First line is the title (up to 79 characters)
        title = stripped_no_comment_str(lines[0])
        if len(title) > 79:
            title = title[:79]
        
        # Parse &SUB statements for variable substitutions
        substitutions = {}
        sub_pattern = re.compile(r'&SUB\s+(\w+)\s+(.*?)(?=\s*$|\s*!)')
        
        idx = 1
        cooling_section = None
        while idx < len(lines):
            line_stripped = stripped_no_comment_str(lines[idx])

            # Skip comments and empty lines when checking for section markers
            if not line_stripped:
                idx += 1
                continue

            # If we see a substitution
            if line_stripped.startswith('&SUB'):
                match = sub_pattern.match(line_stripped)
                if match:
                    var_name, var_value = match.groups()
                    var_value = var_value.strip()
                    
                    substitution = Substitution(name=var_name, value=var_value)
                    substitutions[var_name] = substitution
                    logger.debug(f"Found substitution \"{var_name}\" -> \"{var_value}\"")
                idx += 1
                continue

            # Start of the cooling regions section
            if line_stripped == "SECTION":
                logger.debug("Starting to parse cooling section")
                cooling_section, idx = CoolingSection.parse_input_file(lines, idx)
                continue
            
            idx += 1
                
        return cls(
            title=title,
            substitutions=substitutions,
            cooling_section=cooling_section
        )