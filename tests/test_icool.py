import pytest
import os
import numpy as np
from electronsandstuff.icool import ICoolInput, UnresolvedSubstitutionsError


def test_icool_input_loading():
    """Test loading an ICOOL input file."""
    # Path to the test data file
    data_path = os.path.join(os.path.dirname(__file__), "data", "adib.f01")

    # Test file existence
    assert os.path.exists(data_path), f"Test data file not found: {data_path}"

    # Load the input object from the file
    inp = ICoolInput.from_file(data_path)

    # Check that the title was loaded
    assert isinstance(inp.title, str)
    assert len(inp.title) > 0

    # Check that the cooling section was loaded
    assert inp.cooling_section is not None
    assert len(inp.cooling_section.commands) > 0


def test_icool_substitutions():
    """Test the substitution functionality."""
    # Path to the test data file
    data_path = os.path.join(os.path.dirname(__file__), "data", "adib.f01")

    # Load the input object from the file
    inp = ICoolInput.from_file(data_path)

    # Check if the file has substitutions
    assert inp.has_substitutions

    # Test that assert_no_substitutions raises the expected error
    with pytest.raises(UnresolvedSubstitutionsError):
        inp.assert_no_substitutions()

    # Perform substitutions
    inp_sub = inp.perform_substitutions()

    # Verify substitutions were performed
    assert not inp_sub.has_substitutions

    # Test that assert_no_substitutions no longer raises an error
    try:
        inp_sub.assert_no_substitutions()
    except UnresolvedSubstitutionsError:
        pytest.fail("Substitutions were not properly resolved")


def test_icool_get_length():
    """Test the get_length functionality."""
    # Path to the test data file
    data_path = os.path.join(os.path.dirname(__file__), "data", "adib.f01")

    # Load the input object from the file
    inp = ICoolInput.from_file(data_path)

    # Check that we can get the total length
    total_length = inp.get_length()
    assert isinstance(total_length, float)
    assert total_length > 0

    # Perform substitutions
    inp_sub = inp.perform_substitutions()

    # Check that we can get the length after substitutions
    total_length_sub = inp_sub.get_length()
    assert isinstance(total_length_sub, float)
    assert total_length_sub > 0

    # The lengths should be the same
    assert np.isclose(total_length, total_length_sub)


def test_icool_from_string():
    """Test loading an ICOOL input file from a string."""
    # Path to the test data file
    data_path = os.path.join(os.path.dirname(__file__), "data", "adib.f01")

    # Read the file content
    with open(data_path, "r") as f:
        content = f.read()

    # Load from string
    inp = ICoolInput.from_str(content)

    # Basic checks
    assert inp.cooling_section is not None
    assert len(inp.cooling_section.commands) > 0

    # Check if the file has substitutions
    assert inp.has_substitutions
