from pydantic import BaseModel
from typing import Dict, Any
import logging

from .substitution import SubKey

logger = logging.getLogger(__name__)


class ICoolBase(BaseModel):
    """
    Base class for all ICOOL objects that provides substitution functionality.

    This class implements a method to recursively perform substitutions on all
    member variables that are of type SubKey or contain SubKey instances.
    """

    def perform_substitutions(self, substitutions: Dict[str, Any]) -> "ICoolBase":
        """
        Create a new object with substitutions applied to all member variables.

        Args:
            substitutions: A dictionary mapping substitution keys to their values.

        Returns:
            A new instance of the same class with substitutions applied.
        """
        logger.debug(f"Performing substitutions on {self.__class__.__name__}")

        # Create a dictionary to hold the new field values
        new_values = {}

        # Process each field
        for field_name, field_value in self.__dict__.items():
            # Skip private fields
            if field_name.startswith("_"):
                continue

            # Handle SubKey directly
            if isinstance(field_value, SubKey):
                if field_value.key in substitutions:
                    logger.debug(
                        f"Substituting {field_name}: {field_value.key} -> {substitutions[field_value.key]}"
                    )
                    new_values[field_name] = substitutions[field_value.key]
                else:
                    logger.warning(f"No substitution found for key '{field_value.key}'")
                    new_values[field_name] = field_value

            # Handle lists - could contain SubKey or ICoolBase objects
            elif isinstance(field_value, list):
                new_list = []
                for item in field_value:
                    if isinstance(item, SubKey):
                        if item.key in substitutions:
                            logger.debug(
                                f"Substituting list item: {item.key} -> {substitutions[item.key]}"
                            )
                            new_list.append(substitutions[item.key])
                        else:
                            logger.warning(
                                f"No substitution found for key '{item.key}'"
                            )
                            new_list.append(item)
                    elif isinstance(item, ICoolBase):
                        new_list.append(item.perform_substitutions(substitutions))
                    else:
                        new_list.append(item)
                new_values[field_name] = new_list

            # Recursively handle ICoolBase objects
            elif isinstance(field_value, ICoolBase):
                new_values[field_name] = field_value.perform_substitutions(
                    substitutions
                )

            # Copy other values as is
            else:
                new_values[field_name] = field_value

        # Create a new instance with the substituted values
        return self.__class__(**new_values)
