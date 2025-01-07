from xopt import Generator, VOCS, Xopt
import numpy as np
from typing import List, Union
from pydantic import ConfigDict, Field, SerializeAsAny
import pandas as pd


def extract_decision_vars(dat: Union[pd.DataFrame, List[dict]], vocs: VOCS):
    # If dataframe
    if isinstance(dat, pd.DataFrame):
        return dat[vocs.variable_names].values

    # If dicts
    return np.array([[d[vn] for vn in vocs.variable_names] for d in dat])


class DeduplicatedGenerator(Generator):
    name = "deduplicated-generator"
    supports_multi_objective: bool = True
    generator: SerializeAsAny[Generator] = Field(
        description="generator object for Xopt"
    )
    model_config = ConfigDict(extra="allow")

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Setup the decision variable archive
        self._decision_vars = np.empty((0, self.vocs.n_variables))
        if self.generator.data is not None:
            self._decision_vars = np.unique(
                np.concatenate(
                    (
                        self._decision_vars,
                        extract_decision_vars(self.generator.data, self.vocs),
                    ),
                    axis=0,
                ),
                axis=0,
            )
        if (
            hasattr(self.generator, "_loaded_population")
            and self.generator._loaded_population is not None
        ):
            self._decision_vars = np.unique(
                np.concatenate(
                    (
                        self._decision_vars,
                        extract_decision_vars(
                            self.generator._loaded_population, self.vocs
                        ),
                    ),
                    axis=0,
                ),
                axis=0,
            )

    @property
    def is_done(self):
        return self.generator.is_done

    def generate(self, n_candidates) -> list[dict]:
        # Create never before seen candidates by calling underlying generator and only taking unique
        # value from it until we have `n_candidates` values.
        candidates = []
        while len(candidates) < n_candidates:
            from_generator = self.generator.generate(n_candidates - len(candidates))
            _, idx = np.unique(
                np.concatenate(
                    (
                        self._decision_vars,  # Must go first since first instance of unique elements are included
                        extract_decision_vars(
                            from_generator, self.vocs
                        ),  # Do not accept repeated elements here
                    ),
                    axis=0,
                ),
                return_index=True,
                axis=0,
            )
            idx = idx - len(self._decision_vars)
            idx = idx[idx >= 0]
            for i in idx:
                candidates.append(from_generator[i])

        # Make sure to store the generated solutoins for future testing
        self._decision_vars = np.concatenate(
            (self._decision_vars, extract_decision_vars(candidates, self.vocs)), axis=0
        )

        # Hand candidates back to user
        return candidates[:n_candidates]

    def add_data(self, new_data: pd.DataFrame):
        """
        update dataframe with results from new evaluations.

        This is intended for generators that maintain their own data.
        """
        return self.generator.add_data(new_data)

    @classmethod
    def from_generator(cls, generator: Generator):
        return DeduplicatedGenerator(
            generator=generator,
            supports_multi_objective=generator.supports_multi_objective,
            supports_batch_generation=generator.supports_batch_generation,
            vocs=generator.vocs,
        )

    @staticmethod
    def inject(xopt: Xopt):
        # Avoid chaining generators
        if isinstance(xopt.generator, DeduplicatedGenerator):
            raise ValueError(
                f"xopt generator is already DeduplicatedGenerator: {xopt.generator}"
            )

        # Insert ourself
        xopt.generator = DeduplicatedGenerator.from_generator(xopt.generator)

        # Copy any existing data into our decision variable archive
        if xopt.data is not None:
            xopt.generator._decision_vars = np.unique(
                np.concatenate(
                    (
                        xopt.generator._decision_vars,
                        extract_decision_vars(xopt.data, xopt.generator.vocs),
                    ),
                    axis=0,
                ),
                axis=0,
            )
