from electronsandstuff.paretobench.xopt import XoptProblemWrapper
from paretobench import Problem
from xopt import Xopt, Evaluator
from xopt.generators.ga.cnsga import CNSGAGenerator
import tempfile


def test_cnsga_importer(n_generations=50):
    # Our test problem
    prob = XoptProblemWrapper(Problem.from_line_fmt("WFG1 (n=16, k=2, m=2)"))

    # A place to store the output file
    dir = tempfile.mkdtemp()

    # Setup NSGA-II in xopt to solve it
    population_size = 50
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
    print(dir)
