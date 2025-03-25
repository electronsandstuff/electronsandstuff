from typing import List, Union, Literal, Tuple
import logging

from .base import ICoolBase
from .utils import stripped_no_comment_str
from .substitution import to_float_or_sub, FloatOrSub, IntOrSub


logger = logging.getLogger(__name__)


class ICoolField(ICoolBase):
    name: Literal["FIELD"] = "FIELD"

    @classmethod
    def parse_input_file(
        cls, lines: List[str], start_idx: int
    ) -> Tuple["ICoolField", int]:
        ftag = stripped_no_comment_str(lines[start_idx])
        fparam = stripped_no_comment_str(lines[start_idx + 1]).split()
        fparam = [to_float_or_sub(x) for x in fparam]

        if ftag == "ACCEL":
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
            elif model == 10:
                obj = FieldAccel10(
                    phase=fparam[3],
                    n_wavelen=fparam[4],
                    reset=fparam[5],
                    total_len=fparam[6],
                    g0=fparam[7],
                    g1=fparam[8],
                    g2=fparam[9],
                    phase_model=fparam[11],
                )
            else:
                raise ValueError(
                    f"Sorry, but accelerating cavity model {model} is not implemented yet"
                )
        elif ftag == "NONE":
            obj = FieldNone()
        elif ftag == "SOL":
            obj = FieldSol()
        elif ftag == "STUS":
            obj = FieldSTUS()
        else:
            raise ValueError(f'Unrecognized value for FTAG: "{ftag}"')
        return obj, (start_idx + 2)


class FieldSTUS(ICoolField):
    name: Literal["STUS"] = "STUS"


class FieldNone(ICoolField):
    name: Literal["NONE"] = "NONE"


class FieldAccel2(ICoolField):
    name: Literal["ACCEL2"] = "ACCEL2"
    freq: FloatOrSub
    gradient: FloatOrSub
    phase: FloatOrSub
    rect_param: FloatOrSub
    x_offset: FloatOrSub
    y_offset: FloatOrSub
    long_mode_p: IntOrSub


class FieldAccel10(ICoolField):
    name: Literal["ACCEL10"] = "ACCEL10"
    phase: FloatOrSub
    n_wavelen: FloatOrSub
    reset: FloatOrSub
    total_len: FloatOrSub
    g0: FloatOrSub
    g1: FloatOrSub
    g2: FloatOrSub
    phase_model: IntOrSub


class FieldSol(ICoolField):
    """Placeholder solenoid element"""

    name: Literal["SOL"] = "SOL"


all_fields = Union[FieldSTUS, FieldNone, FieldAccel2, FieldAccel10, FieldSol]
