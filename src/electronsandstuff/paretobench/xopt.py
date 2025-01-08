from datetime import datetime
from functools import partial
from paretobench import Problem, Population, History
from typing import Union, Optional
from xopt import VOCS, Xopt
import numpy as np
import os
import pandas as pd
import re


class XoptProblemWrapper:
    def __init__(self, problem: Problem):
        """
        This class wraps a ParetoBench problem for use with xopt. After creation of the wrapper object from
        the problem, the Xopt VOCS object can be accessed through a class property. The wrapper object is
        also a callable and may be directly passed to the Xopt evaluator object.

        Example
        -------
        > import paretobench as pb
        > from xopt import Xopt, Evaluator
        > from xopt.generators.ga.cnsga import CNSGAGenerator
        >
        > prob = XoptProblemWrapper(pb.Problem.from_line_fmt('WFG1'))
        > population_size = 50
        > ev = Evaluator(function=prob, vectorized=True, max_workers=population_size)
        > X = Xopt(
        >        generator=CNSGAGenerator(vocs=prob.vocs, population_size=population_size),
        >        evaluator=ev,
        >        vocs=prob.vocs,
        >    )


        Parameters
        ----------
        problem : Problem
            A problem object that follows the Problem class interface.
        """
        self.prob = problem

    @property
    def vocs(self) -> VOCS:
        """Return the VOCS object."""
        # Construct the decision variables
        lbs = self.prob.var_lower_bounds
        ubs = self.prob.var_upper_bounds
        vars = {f"x{i}": [lb, ub] for i, (lb, ub) in enumerate(zip(lbs, ubs))}

        # Construct the objectives
        objs = {f"f{i}": "MINIMIZE" for i in range(self.prob.n_objs)}

        # The constraints
        constraints = {
            f"g{i}": ["GREATER_THAN", 0] for i in range(self.prob.n_constraints)
        }

        # Construct VOCS object
        return VOCS(variables=vars, objectives=objs, constraints=constraints)

    def __call__(self, input_dict: dict) -> dict:
        """
        Evaluate the problem using the dict -> dict convention for xopt.

        Parameters
        ----------
        input_dict : dict
            A dictionary containing the decision variables

        Returns
        -------
        dict
            A dictionary with the objectives and constraints
        """
        # Convert the input dictionary to a NumPy array of decision variables
        x = np.array([input_dict[f"x{i}"] for i in range(self.prob.n_vars)]).T

        # Evaluate the problem
        pop = self.prob(x)  # Pass single batch

        # Convert the result to the format expected by Xopt
        ret = {}
        ret.update({f"f{i}": pop.f[:, i] for i in range(self.prob.n_objs)})
        ret.update({f"g{i}": pop.g[:, i] for i in range(self.prob.n_constraints)})
        return ret

    def __repr__(self):
        return f"XoptProblemWrapper({self.prob.to_line_fmt()})"


def import_cnsga_population(
    path: Union[str, os.PathLike[str]], vocs: VOCS, errors_as_constraints: bool = False
):
    df = pd.read_csv(path)

    # Get base constraints if they exist
    g = -vocs.constraint_data(df).to_numpy() if vocs.constraints else None
    names_g = vocs.constraint_names.copy() if vocs.constraints else []

    # Handle error column if requested
    if errors_as_constraints:
        # Convert boolean strings to ±1, reshape to 2D array
        error_constraints = np.where(
            df["xopt_error"].astype(str).str.lower() == "true", -1.0, 1.0
        )[:, np.newaxis]  # Makes it 2D: (n_samples, 1)

        # Combine with existing constraints if present
        if g is not None:
            g = np.hstack([g, error_constraints])
            names_g.append("xopt_error")
        else:
            g = error_constraints
            names_g = ["xopt_error"]

    return Population(
        x=vocs.variable_data(df).to_numpy(),
        f=vocs.objective_data(df).to_numpy(),
        g=g,
        names_x=vocs.variable_names,
        names_f=vocs.objective_names,
        names_g=names_g,
    )


def import_cnsga_history(
    output_path: Union[str, os.PathLike[str]],
    vocs: Optional[VOCS] = None,
    config_file: Union[None, str, os.PathLike[str]] = None,
    problem: str = "",
    errors_as_constraints: bool = False,
):
    if (vocs is None) and (config_file is None):
        raise ValueError("Must specify one of vocs or config_file")

    # Get vocs from config file
    if vocs is None:
        xx = Xopt.from_yaml(config_file)
        vocs = xx.vocs

    # Get list of population files and their datetimes
    population_files = []
    population_datetimes = []

    # Regex pattern to match both datetime formats
    datetime_pattern = r"cnsga_population_(\d{4}-\d{2}-\d{2}T\d{2}[_:]\d{2}[_:]\d{2}\.\d+[-+]\d{2}[_:]\d{2})\.csv"

    # Walk through all files in directory
    for filename in os.listdir(output_path):
        match = re.match(datetime_pattern, filename)
        if match:
            # Get datetime string and convert _ to : if needed
            dt_str = match.group(1).replace("_", ":")

            try:
                # Parse datetime string
                dt = datetime.fromisoformat(dt_str)

                # Store filename and datetime
                population_files.append(filename)
                population_datetimes.append(dt)
            except ValueError as e:
                print(
                    f"Warning: Could not parse datetime from filename {filename}: {e}"
                )

    # Sort both lists based on datetime
    population_files = [
        os.path.join(output_path, x)
        for _, x in sorted(zip(population_datetimes, population_files))
    ]

    # Import files as populations
    pops = list(
        map(
            partial(
                import_cnsga_population,
                vocs=vocs,
                errors_as_constraints=errors_as_constraints,
            ),
            population_files,
        )
    )

    # Update fevals
    fevals = 0
    for pop in pops:
        fevals += len(pop)
        pop.fevals = fevals

    return History(reports=pops, problem=problem)
