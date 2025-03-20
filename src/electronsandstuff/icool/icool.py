from pydantic import BaseModel, Field, validator
from typing import Optional, List, Dict, Union, Any, Literal, Tuple, Annotated
import re
import logging


logger = logging.getLogger(__name__)


class SubKey(BaseModel):
    """Represents a key to a substitution which will be replaced eventually"""
    key: str


# Helper types for values that can be replaced with substitutions
IntOrSub = Union[SubKey, int]
FloatOrSub = Union[SubKey, float]
BoolOrSub = Union[SubKey, bool]
StrOrSub = Union[SubKey, str]


def to_int_or_sub(val: str) -> IntOrSub:
    val = val.strip()
    if val[0] == "&":
        return SubKey(key=val[1:])
    else:
        return int(val)
    

def to_float_or_sub(val: str) -> IntOrSub:
    val = val.strip()
    if val[0] == "&":
        return SubKey(key=val[1:])
    else:
        return float(val)


def to_bool_or_sub(val: str):
    val = val.strip()
    if val[0] == "&":
        return SubKey(key=val[1:])
    else:
        val = val.lower()
        if val == ".true.":
            return True
        if val == ".false.":
            return False
        raise ValueError(f"Could not process str as fortran logical type: \"{val}\"")


def to_str_or_sub(val: str) -> IntOrSub:
    val = val.strip()
    if val[0] == "&":
        return SubKey(key=val[1:])
    else:
        return val


class ICoolField(BaseModel):
    name: Literal["FIELD"] = "FIELD"

    @classmethod
    def parse_input_file(cls, lines: List[str], start_idx: int) -> Tuple["ICoolField", int]:
        ftag = stripped_no_comment_str(lines[start_idx])
        fparam = stripped_no_comment_str(lines[start_idx+1]).split()
        fparam = [to_float_or_sub(x) for x in fparam]

        if ftag == "NONE":
            obj = FieldNone()
        elif ftag == "STUS":
            obj = FieldSTUS()
        elif ftag == "ACCEL":
            model = int(fparam[0])
            if model == 2:
                obj = FieldAccel2(
                    freq=fparam[1],
                    gradient=fparam[2],
                    phase=fparam[3],
                    rect_param=fparam[4],
                    x_offset=fparam[5],
                    y_offset=fparam[6],
                    long_mode_p=fparam[7],
                )
            else:
                raise ValueError(f"Sorry, but accelerating cavity model {model} is not implemented yet")
        else:
            raise ValueError(f"Unrecognized value for FTAG: \"{ftag}\"")
        return obj, (start_idx + 2)


class FieldSTUS(ICoolField):
    name: Literal['STUS'] = "STUS"


class FieldNone(ICoolField):
    name: Literal['NONE'] = "NONE"


class FieldAccel2(ICoolField):
    name: Literal['ACCEL2'] = "ACCEL2"
    freq: FloatOrSub
    gradient: FloatOrSub
    phase: FloatOrSub
    rect_param: FloatOrSub
    x_offset: FloatOrSub
    y_offset: FloatOrSub
    long_mode_p: IntOrSub


all_fields = Union[FieldSTUS, FieldNone, FieldAccel2]


class ICoolGeometry(BaseModel):
    name: Literal["Geometry"] = "Geometry"

    @classmethod
    def parse_input_file(cls, lines: List[str], start_idx: int) -> Tuple["ICoolGeometry", int]:
        gtag = stripped_no_comment_str(lines[start_idx])
        gparam = stripped_no_comment_str(lines[start_idx+1]).split()
        gparam = [to_float_or_sub(x) for x in gparam]

        if gtag == "NONE":
            obj = GeometryNone()
        elif gtag == "ASPW":
            obj = GeometryASPW(
                z_pos=gparam[0],
                z_offset=gparam[1],
                a0=gparam[2],
                a1=gparam[3],
                a2=gparam[4],
                a3=gparam[5],
            )
        elif gtag == "ASRW":
            obj = GeometryASWR(
                symmetry_dist=gparam[0],
                max_half_thickness=gparam[1],
                a0=gparam[2],
                a1=gparam[3],
                a2=gparam[4],
                a3=gparam[5],
            )
        elif gtag == "CBLOCK":
            obj = GeometryCBlock()
        elif gtag == "HWIN":
            obj = GeometryHWin(
                end_flag=gparam[0],
                r_inner=gparam[1],
                thickness=gparam[2],
                offset=gparam[3],
            )
        elif gtag == "NIA":
            obj = GeometryNIA(
                zv=gparam[0],
                z0=gparam[1],
                z1=gparam[2],
                theta0=gparam[3],
                phi0=gparam[4],
                theta1=gparam[5],
                phi1=gparam[6],
            )
        elif gtag == "PWEDGE":
            obj = GeometryPWedge(
                vert_x=gparam[1],
                vert_z=gparam[2],
                vert_phi=gparam[3],
                width=gparam[4],
                height=gparam[5],
                a0=gparam[6],
                a1=gparam[7],
                a2=gparam[8],
                a3=gparam[0],
            )
        elif gtag == "RING":
            obj = GeometryRing(
                r_inner=gparam[0],
                r_outer=gparam[1]
            )
        elif gtag == "WEDGE":
            obj = GeometryWedge(
                full_angle=gparam[0],
                vert_x=gparam[1],
                vert_z=gparam[2],
                vert_phi=gparam[3],
                width=gparam[4],
                height=gparam[5]
            )
        else:
            raise ValueError(f"Unrecognized value for GTAG: \"{gtag}\"")
        return obj, (start_idx + 2)


class GeometryNone(ICoolGeometry):
    name: Literal["NONE"] = "NONE"


class GeometryASPW(ICoolGeometry):
    name: Literal["ASPW"] = "ASPW"
    z_pos: FloatOrSub
    z_offset: FloatOrSub
    a0: FloatOrSub
    a1: FloatOrSub
    a2: FloatOrSub
    a3: FloatOrSub


class GeometryASWR(ICoolGeometry):
    name: Literal["ASWR"] = "ASWR"
    symmetry_dist: FloatOrSub
    max_half_thickness: FloatOrSub
    a0: FloatOrSub
    a1: FloatOrSub
    a2: FloatOrSub
    a3: FloatOrSub


class GeometryCBlock(ICoolGeometry):
    name: Literal["CBLOCK"] = "CBLOCK"


class GeometryHWin(ICoolGeometry):
    name: Literal["HWIN"] = "HWIN"
    end_flag: FloatOrSub
    r_inner: FloatOrSub
    thickness: FloatOrSub
    offset: FloatOrSub


class GeometryNIA(ICoolGeometry):
    name: Literal["NIA"] = "NIA"
    zv: FloatOrSub
    z0: FloatOrSub
    z1: FloatOrSub
    theta0: FloatOrSub
    phi0: FloatOrSub
    theta1: FloatOrSub
    phi1: FloatOrSub


class GeometryPWedge(ICoolGeometry):
    name: Literal["PWEDGE"] = "PWEDGE"
    vert_x: FloatOrSub
    vert_z: FloatOrSub
    vert_phi: FloatOrSub
    width: FloatOrSub
    height: FloatOrSub
    a0: FloatOrSub
    a1: FloatOrSub
    a2: FloatOrSub
    a3: FloatOrSub


class GeometryRing(ICoolGeometry):
    name: Literal["RING"] = "RING"
    r_inner: FloatOrSub
    r_outer: FloatOrSub


class GeometryWedge(ICoolGeometry):
    name: Literal["WEDGE"] = "WEDGE"
    full_angle: FloatOrSub
    vert_x: FloatOrSub
    vert_z: FloatOrSub
    vert_phi: FloatOrSub
    width: FloatOrSub
    height: FloatOrSub


all_geometry = Union[ICoolGeometry, GeometryCBlock, GeometryNone, GeometryWedge]


class RegionCommand(BaseModel):
    pass

    @classmethod
    def parse_input_file(cls, lines: List[str], start_idx: int) -> Tuple["RegionCommand", int]:
        raise NotImplementedError


class RSubRegion(BaseModel):
    rlow: FloatOrSub
    rhigh: FloatOrSub
    field: Annotated[all_fields, Field(discriminator="name")]
    mtag: StrOrSub
    geometry: Annotated[all_geometry, Field(discriminator="name")]


class SRegion(RegionCommand):
    name: Literal["SREGION"] = "SREGION"
    slen: FloatOrSub
    zstep: FloatOrSub
    subregions: List[RSubRegion]

    @classmethod
    def parse_input_file(cls, lines: List[str], start_idx: int) -> Tuple["RegionCommand", int]:
        slen, nrreg, zstep = stripped_no_comment_str(lines[start_idx+1]).split()
        slen = to_float_or_sub(slen)
        zstep = to_float_or_sub(zstep)

        subregions = []
        end_idx = start_idx + 2
        for reg_idx in range(int(nrreg)):
            irreg, rlow, rhigh = stripped_no_comment_str(lines[start_idx+2+6*reg_idx]).split()
            if int(irreg) != (reg_idx + 1):
                raise ValueError(f"r region index did not match loop index. Something went wrong? (irreg={irreg}, loop_idx={reg_idx})")

            field, _ = ICoolField.parse_input_file(lines, start_idx+3+6*reg_idx)
            mtag = stripped_no_comment_str(lines[ start_idx+5+6*reg_idx])
            geometry, _ = ICoolGeometry.parse_input_file(lines, start_idx+6+6*reg_idx)
            subregions.append(RSubRegion(
                rlow=to_float_or_sub(rlow),
                rhigh=to_float_or_sub(rhigh),
                field=field,
                mtag=to_str_or_sub(mtag),
                geometry=geometry
            ))

            end_idx += 6

        obj = cls(
            slen=slen,
            zstep=zstep,
            subregions=subregions,
        )
        return obj, end_idx


class RefP(RegionCommand):
    name: Literal["REFP"] = "REFP"
    refpar: IntOrSub
    

    @classmethod
    def parse_input_file(cls, lines: List[str], start_idx: int) -> Tuple["RegionCommand", int]:
        # Pull out parameters
        refpar, param_a, param_b, param_c, phmoderef = stripped_no_comment_str(lines[start_idx+1]).split()
        refpar = to_int_or_sub(refpar)
        param_a = to_float_or_sub(param_a)
        param_b = to_float_or_sub(param_b)
        param_c = to_float_or_sub(param_c)

        # Generate the correct chlid class
        phmoderef = int(phmoderef)
        if phmoderef == 2:
            obj = RefP2(refpar=refpar)
        elif phmoderef == 3:
            obj = RefP3(refpar=refpar, pz0=param_a, t0=param_b)
        elif phmoderef == 4:
            obj = RefP4(refpar=refpar, pz0=param_a, t0=param_b, dedz=param_c)
        elif phmoderef == 5:
            obj = RefP5(refpar=refpar, e0=param_a, dedz=param_b, d2edz2=param_c)
        elif phmoderef == 6:
            obj = RefP6(refpar=refpar, e0=param_a, dedz=param_b, d2edz2=param_c)
        else:
            raise ValueError(f"Unrecognized PHMODEREF: {phmoderef}")
        return obj, (start_idx + 2)


class RefP2(RefP):
    name: Literal["REFP2"] = "REFP2"
    pass


class RefP3(RefP):
    name: Literal["REFP3"] = "REFP3"
    pz0: FloatOrSub
    t0: FloatOrSub


class RefP4(RefP):
    name: Literal["REFP4"] = "REFP4"
    pz0: FloatOrSub
    t0: FloatOrSub
    dedz: FloatOrSub


class RefP5(RefP):
    name: Literal["REFP5"] = "REFP5"
    e0: FloatOrSub
    dedz: FloatOrSub
    d2edz2: FloatOrSub


class RefP6(RefP):
    name: Literal["REFP6"] = "REFP6"
    e0: FloatOrSub
    dedz: FloatOrSub
    d2edz2: FloatOrSub


class Grid(RegionCommand):
    name: Literal["GRID"] = "GRID"
    grid_num: IntOrSub
    field_type: StrOrSub
    file_num: IntOrSub
    curvature_flag: IntOrSub
    ref_momentum: FloatOrSub
    field_scale: FloatOrSub
    curvature_sign: FloatOrSub
    file_format: IntOrSub
    long_shift: FloatOrSub

    @classmethod
    def parse_input_file(cls, lines: List[str], start_idx: int) -> Tuple["RegionCommand", int]:
        grid_num = stripped_no_comment_str(lines[start_idx+1])
        field_type = stripped_no_comment_str(lines[start_idx+2])

        # Extract 15 grid parameters
        (_, file_num, curvature_flag, 
         ref_momentum, field_scale, _, _,
         curvature_sign, file_format, 
         long_shift, _, _, _, _, _)  = stripped_no_comment_str(lines[start_idx+3]).split()

        # Construct object
        obj = Grid(
            grid_num=to_int_or_sub(grid_num),
            field_type=to_str_or_sub(field_type),
            file_num=to_int_or_sub(file_num.rstrip(".")),
            curvature_flag=to_int_or_sub(curvature_flag.rstrip(".")),
            ref_momentum=to_float_or_sub(ref_momentum),
            field_scale=to_float_or_sub(field_scale),
            curvature_sign=to_float_or_sub(curvature_sign),
            file_format=to_int_or_sub(file_format.rstrip(".")),
            long_shift=to_float_or_sub(long_shift)
        )
        return obj, (start_idx + 4)


class DVar(RegionCommand):
    name: Literal["DVAR"] = "DVAR"
    var_idx: IntOrSub
    change: FloatOrSub
    apply_to: IntOrSub

    @classmethod
    def parse_input_file(cls, lines: List[str], start_idx: int) -> Tuple["RegionCommand", int]:
        # Grab the parameters
        var_idx, change, apply_to = stripped_no_comment_str(lines[start_idx+1]).split()
        obj = cls(
            var_idx=to_int_or_sub(var_idx), 
            change=to_float_or_sub(change), 
            apply_to=to_int_or_sub(apply_to)
        )
        return obj, (start_idx + 2)
    

class Cell(RegionCommand):
    name: Literal["CELL"] = "CELL"
    n_cells: IntOrSub
    cell_flip: BoolOrSub
    field: Annotated[all_fields, Field(discriminator="name")]
    commands: List[Annotated[
        Union[SRegion, RefP2, RefP3, RefP4, RefP5, RefP6, Grid, DVar, "Repeat"],
        Field(discriminator="name")]] = Field(default_factory=list)


    @classmethod
    def parse_input_file(cls, lines: List[str], start_idx: int) -> Tuple["RegionCommand", int]:
        # Grab parameters from top of cell
        n_cells = to_int_or_sub(stripped_no_comment_str(lines[start_idx+1]))
        cell_flip = to_bool_or_sub(stripped_no_comment_str(lines[start_idx+2]))
        field, _ = ICoolField.parse_input_file(lines, start_idx+3)

        # Process internal commands
        cmds, end_idx = parse_region_cmds(lines, start_idx+5, end_cmd="ENDCELL")

        # Make object
        obj = cls(commands=cmds, n_cells=n_cells, cell_flip=cell_flip, field=field)
        return obj, end_idx


class Repeat(RegionCommand):
    name: Literal["REPEAT"] = "REPEAT"
    n_repeat: IntOrSub
    commands: List[Annotated[
        Union[SRegion, RefP2, RefP3, RefP4, RefP5, RefP6, Grid, DVar],
        Field(discriminator="name")]] = Field(default_factory=list)

    @classmethod
    def parse_input_file(cls, lines: List[str], start_idx: int) -> Tuple["RegionCommand", int]:
        # Get the number of repeats
        n_repeat = to_int_or_sub(stripped_no_comment_str(lines[start_idx+1]))

        # Process commands in the block
        cmds, end_idx = parse_region_cmds(lines, start_idx+2, end_cmd="ENDR")

        # Return the object
        return cls(commands=cmds, n_repeat=n_repeat), end_idx


# The registered commands
registered_commands: List[RegionCommand] = [SRegion, RefP, Grid, DVar, Cell, Repeat]
name_to_command: Dict[str, RegionCommand] = {cmd.model_fields["name"].default: cmd for cmd in registered_commands}


def stripped_no_comment_str(ss):
    if "!" in ss:
        return ss.split("!")[0].strip()
    return ss.strip()


def parse_region_cmds(lines, start_idx, end_cmd=""):
    logger.debug(f"Begining to parse region commands (len(lines)={len(lines)}, start_idx={start_idx}, end_cmd={end_cmd})")
    idx = start_idx
    cmds = []
    while idx < len(lines):
        line_stripped = stripped_no_comment_str(lines[idx])

        # If we see a registered command, parse it and add to list
        if line_stripped in name_to_command:
            logger.debug(f"Found command \"{line_stripped}\"")
            cmd, end_idx = name_to_command[line_stripped].parse_input_file(lines, idx)
            cmds.append(cmd)
            idx = end_idx
            continue

        # If we see the "end command" for this section
        if line_stripped == end_cmd:
            break

        idx += 1

    return cmds, idx


class Substitution(BaseModel):
    """Represents a name substitution defined with &SUB."""
    name: str = Field(..., description="Name of the substitution (max 20 chars)")
    value: Union[float, int, str] = Field(..., description="Value of the substitution (max 30 chars)")
    
    @validator('name')
    def name_length(cls, v):
        if len(v) > 20:
            raise ValueError(f"Substitution name '{v}' exceeds maximum length of 20 characters")
        if v.upper() in ["SUB", "SCL"]:
            raise ValueError(f"Substitution name '{v}' cannot be 'SUB' or 'SCL'")
        return v
    
    @validator('value', pre=True)
    def convert_value(cls, v):
        if isinstance(v, (int, float)):
            return v
        
        # Try to convert string to numeric if possible
        try:
            if '.' in v or 'e' in v.lower() or 'E' in v:
                return float(v)
            else:
                return int(v)
        except (ValueError, TypeError):
            # Keep as string if conversion fails
            if len(str(v)) > 30:
                raise ValueError(f"Substitution value '{v}' exceeds maximum length of 30 characters")
            return v


class CoolingSection(BaseModel):
    """Represents the cooling section of an ICOOL input file."""
    commands: List[Annotated[
        Union[SRegion, RefP2, RefP3, RefP4, RefP5, RefP6, Grid, DVar, Cell, Repeat],
        Field(discriminator="name")]] = Field(default_factory=list, description="Content of the cooling section")


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
            idx += 1

            # Skip comments and empty lines when checking for section markers
            if not line_stripped:
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

            # Start of the cooling regions section
            if line_stripped == "SECTION":
                logger.debug("Starting to parse cooling section")
                cmds, end_idx = parse_region_cmds(lines, idx, end_cmd="ENDSECTION")
                cooling_section = CoolingSection(commands=cmds)
                idx = end_idx
                continue
                
        return cls(
            title=title,
            substitutions=substitutions,
            cooling_section=cooling_section
        )