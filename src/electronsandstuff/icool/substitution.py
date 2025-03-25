from pydantic import BaseModel, Field, field_validator
from typing import Union
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
        raise ValueError(f'Could not process str as fortran logical type: "{val}"')


def to_str_or_sub(val: str) -> IntOrSub:
    val = val.strip()
    if val[0] == "&":
        return SubKey(key=val[1:])
    else:
        return val


class Substitution(BaseModel):
    """Represents a name substitution defined with &SUB."""

    name: str = Field(..., description="Name of the substitution (max 20 chars)")
    value: Union[float, int, str] = Field(
        ..., description="Value of the substitution (max 30 chars)"
    )

    @field_validator("name")
    @classmethod
    def name_length(cls, v):
        if len(v) > 20:
            raise ValueError(
                f"Substitution name '{v}' exceeds maximum length of 20 characters"
            )
        if v.upper() in ["SUB", "SCL"]:
            raise ValueError(f"Substitution name '{v}' cannot be 'SUB' or 'SCL'")
        return v

    @field_validator("value", mode="before")
    @classmethod
    def convert_value(cls, v):
        if isinstance(v, (int, float)):
            return v

        # Try to convert string to numeric if possible
        try:
            if "." in v or "e" in v.lower() or "E" in v:
                return float(v)
            else:
                return int(v)
        except (ValueError, TypeError):
            # Keep as string if conversion fails
            if len(str(v)) > 30:
                raise ValueError(
                    f"Substitution value '{v}' exceeds maximum length of 30 characters"
                )
            return v
