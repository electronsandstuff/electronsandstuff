from electronsandstuff.paretobench.xopt import XoptProblemWrapper
from paretobench import Problem
from xopt import Xopt, Evaluator
from xopt.generators.ga.cnsga import CNSGAGenerator
import tempfile
import numpy as np

from electronsandstuff.paretobench.xopt import import_cnsga_history


def test_cnsga_importer(n_generations=50, population_size=50):
    # Our test problem
    prob = XoptProblemWrapper(Problem.from_line_fmt("WFG1 (n=16, k=2, m=2)"))

    # A place to store the output file
    with tempfile.TemporaryDirectory() as dir:
        # Setup NSGA-II in xopt to solve it
        ev = Evaluator(function=prob, vectorized=True, max_workers=population_size)
        X = Xopt(
            generator=CNSGAGenerator(
                vocs=prob.vocs, population_size=population_size, output_path=dir
            ),
            evaluator=ev,
            vocs=prob.vocs,
        )

        # Run the optimizer
        for _ in range(n_generations):
            X.step()

        # Load the data
        hist = import_cnsga_history(dir, vocs=prob.vocs)

        # Some basic checks
        assert len(hist.reports) == n_generations
        assert len(hist.reports[0]) == population_size
        for report in hist.reports:
            assert report.n == prob.prob.n_vars
            assert report.m == prob.prob.n_objs
            assert report.n_constraints == prob.prob.n_constraints

        # Load with errors
        hist = import_cnsga_history(dir, vocs=prob.vocs, errors_as_constraints=True)

        # Some basic checks
        assert len(hist.reports) == n_generations
        assert len(hist.reports[0]) == population_size
        for report in hist.reports:
            assert report.n == prob.prob.n_vars
            assert report.m == prob.prob.n_objs
            assert report.n_constraints == prob.prob.n_constraints + 1
            np.testing.assert_equal(np.ones(len(report)), report.g[:, -1])
