class ICoolException(Exception):
    """Base exception for all ICOOL-related exceptions."""

    pass


class UnresolvedSubstitutionsError(ICoolException):
    """Exception raised when there are unresolved substitutions in an ICOOL object."""

    def __init__(self, field_path, key):
        self.field_path = field_path
        self.key = key
        message = f"Unresolved substitution '{key}' found in field '{field_path}'"
        super().__init__(message)
