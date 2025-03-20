from pydantic import BaseModel, Field, validator
from typing import Optional, List, Dict, Union, Any, Literal, Tuple, Annotated
import re
import logging


logger = logging.getLogger(__name__)


def stripped_no_comment_str(ss):
    if "!" in ss:
        return ss.split("!")[0].strip()
    return ss.strip()

