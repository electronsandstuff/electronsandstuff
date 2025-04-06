from xopt.generators.ga.nsga2 import NSGA2Generator
import os
import re
from datetime import datetime
import pandas as pd


# Pattern for resolving dates from the filenames
date_pattern = re.compile(
    r".*_(\d{4}-\d{2}-\d{2}T\d{2}[:_]\d{2}[:_]\d{2}\.\d+-\d{2}[:_]\d{2})\.csv"
)


def extract_datetime(filename: str) -> datetime:
    """
    Load a datetime object from the population files in CNSGA's output path. Raise error on unrecognizeable path.
    Supports both formats:
    - With colons: cnsga_offspring_2024-08-30T14:04:59.094665-05:00.csv
    - With underscores: cnsga_population_2025-04-05T17_15_05.234149-07_00.csv

    Parameters
    ----------
    filename : str
        Filename to load the datetime object from

    Returns
    -------
    datetime
        Date from the filename
    """
    match = date_pattern.match(filename)
    if match:
        # Extract the datetime string
        dt_str = match.group(1)

        # Replace underscores with colons if they exist
        dt_str = dt_str.replace("_", ":")

        # Parse the datetime string
        return datetime.fromisoformat(dt_str)
    raise ValueError(f"Could not find datetime in filename: {filename}")


def nsga2_from_cnsga_output_path(output_path: str, **kwargs) -> NSGA2Generator:
    """
    Create a `NSGA2Generator` from existing populations output from `CNSGAGenerator`. This will import the final population
    from the saved data and try to populate all metadata in the new generator object. All other keyword arguments get passed
    on to the `NSGA2Generator` constructor. Be sure to include the required argument `vocs`.

    Parameters
    ----------
    output_path : str
        The path to CNSGA's output

    Returns
    -------
    NSGA2Generator
        The NSGA2Generator loaded with data from CNSGA's final population
    """
    # Grab the last population
    population_files = sorted(
        [f for f in os.listdir(output_path) if "population" in f], key=extract_datetime
    )
    last_population = population_files[-1]

    # Load the last population and all populations
    df_last = pd.read_csv(os.path.join(output_path, last_population))
    df_all = pd.concat(
        [pd.read_csv(os.path.join(output_path, f)) for f in population_files]
    )

    # Set the NSGA2 population parameter
    kwargs["pop"] = df_last.to_dict("records")

    # Add in special keys for this generator
    for individual in kwargs["pop"]:
        individual["xopt_parent_generation"] = 0
        individual["xopt_candidate_idx"] = individual["xopt_index"]

    # Try to get the number of function evaluations
    kwargs["fevals"] = df_all["xopt_index"].max() + 1
    kwargs["n_candidates"] = df_all["xopt_index"].max() + 1
    kwargs["n_generations"] = len(population_files)

    return NSGA2Generator(**kwargs)
