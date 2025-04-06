import pytest
from datetime import datetime, timezone, timedelta
from electronsandstuff.xopt.cnsga_to_nsga2 import extract_datetime


@pytest.mark.parametrize(
    "filename,expected_datetime",
    [
        # Format with colons in the time part
        (
            "cnsga_offspring_2024-08-30T14:04:59.094665-05:00.csv",
            datetime(
                2024, 8, 30, 14, 4, 59, 94665, tzinfo=timezone(timedelta(hours=-5))
            ),
        ),
        (
            "cnsga_offspring_2024-08-30T14:09:29.129272-05:00.csv",
            datetime(
                2024, 8, 30, 14, 9, 29, 129272, tzinfo=timezone(timedelta(hours=-5))
            ),
        ),
        (
            "cnsga_offspring_2024-08-30T14:13:17.169997-05:00.csv",
            datetime(
                2024, 8, 30, 14, 13, 17, 169997, tzinfo=timezone(timedelta(hours=-5))
            ),
        ),
        # Format with underscores in the time part
        (
            "cnsga_population_2025-04-05T17_15_05.234149-07_00.csv",
            datetime(
                2025, 4, 5, 17, 15, 5, 234149, tzinfo=timezone(timedelta(hours=-7))
            ),
        ),
        (
            "cnsga_population_2025-03-30T10_45_12.123456-07_00.csv",
            datetime(
                2025, 3, 30, 10, 45, 12, 123456, tzinfo=timezone(timedelta(hours=-7))
            ),
        ),
        (
            "cnsga_population_2025-04-01T08_30_20.987654-07_00.csv",
            datetime(
                2025, 4, 1, 8, 30, 20, 987654, tzinfo=timezone(timedelta(hours=-7))
            ),
        ),
    ],
)
def test_extract_datetime(filename, expected_datetime):
    """Test that extract_datetime correctly parses datetime from different filename formats."""
    result = extract_datetime(filename)
    assert (
        result == expected_datetime
    ), f"Failed to extract correct datetime from {filename}"
