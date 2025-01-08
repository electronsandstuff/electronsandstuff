from xopt import VOCS, Xopt
import numpy as np
from paretobench import Problem
import os
from typing import Union, Optional


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


def import_cnsga_history(
    output_path: Union[str, os.PathLike[str]],
    vocs: Optional[VOCS] = None,
    config_file: Union[None, str, os.PathLike[str]] = None,
):
    if (vocs is None) and (config_file is None):
        raise ValueError("Must specify one of vocs or config_file")

    # Get vocs from config file
    if vocs is None:
        xx = Xopt.from_yaml(config_file)
        vocs = xx.vocs
