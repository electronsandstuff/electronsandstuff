from pydantic import BaseModel, Field
from typing import List, Dict, Union, Literal, Tuple, Annotated
import logging

from .substitution import FloatOrSub, StrOrSub, IntOrSub, BoolOrSub, to_bool_or_sub, to_float_or_sub, to_int_or_sub, to_str_or_sub
from .fields import all_fields, ICoolField
from .geometry import all_geometry, ICoolGeometry
from .utils import stripped_no_comment_str


logger = logging.getLogger(__name__)


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


class CoolingSection(RegionCommand):
    """Represents the cooling section of an ICOOL input file."""
    name: Literal["SECTION"] = "SECTION"
    commands: List[Annotated[
        Union[SRegion, RefP2, RefP3, RefP4, RefP5, RefP6, Grid, DVar, Cell, Repeat],
        Field(discriminator="name")]] = Field(default_factory=list, description="Content of the cooling section")

    @classmethod
    def parse_input_file(cls, lines: List[str], start_idx: int) -> Tuple["CoolingSection", int]:
        # Process commands in the block
        cmds, end_idx = parse_region_cmds(lines, start_idx+1, end_cmd="ENDSECTION")

        # Return the object
        return cls(commands=cmds), end_idx
