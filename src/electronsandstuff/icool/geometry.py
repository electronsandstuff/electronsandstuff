from pydantic import BaseModel, Field, validator
from typing import Optional, List, Dict, Union, Any, Literal, Tuple, Annotated
import logging


from .utils import stripped_no_comment_str
from .substitution import to_float_or_sub, FloatOrSub


logger = logging.getLogger(__name__)


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
