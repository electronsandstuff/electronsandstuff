from pydantic import BaseModel
from typing import List, Union, Literal, Tuple
import logging

from .utils import stripped_no_comment_str
from .substitution import to_float_or_sub, FloatOrSub, IntOrSub


logger = logging.getLogger(__name__)


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
