"""
Implementation of various constraints, whose violations must be tested by the metrics.
"""

from typing import Iterable
from abc import ABC, abstractmethod
import numpy as np
from numpy.core.multiarray import array as array


class Constraint(ABC):
    @abstractmethod
    def violation(self, samples: np.array, scaling: float) -> np.array:
        """
        Function to calculate a combined metric that considers a region of interest and constraints.
        This method should be invariant to the number of variables in the samples.

        Parameters:
        ----------
        samples: np.array
            The forecast values. (n_samples, variable dimensions)
        scaling: float
            Scaling factor for both the samples, and the constraint values.

        Returns:
        --------
        violation: np.array
            By how much the constraint is violated in each sample.
            If 0, it is satisfied. Always >= 0. (n_samples)
        """
        pass

    def __call__(self, samples: np.array, scaling: float) -> np.array:
        return self.violation(samples, scaling)

    @abstractmethod
    def __repr__(self) -> str:
        pass


class ListConstraint(Constraint):
    """
    A list of constraints, each of which having to be satisfied.
    """

    def __init__(self, constraints: Iterable[Constraint]) -> None:
        super().__init__()
        self.constraints = list(constraints)

    def __getitem__(self, index) -> Constraint:
        return self.constraints[index]

    def __len__(self) -> int:
        return len(self.constraints)

    def violation(self, samples: np.array, scaling: float) -> np.array:
        return sum(
            constraint.violation(samples, scaling) for constraint in self.constraints
        )

    def __repr__(self) -> str:
        sub_constraints = ", ".join(str(constraint) for constraint in self.constraints)
        return f"ListConstraint([{sub_constraints}])"


class MaxConstraint(Constraint):
    """
    Constraint of the form: x_i <= threshold
    """

    def __init__(self, threshold: float) -> None:
        super().__init__()

        self.threshold = threshold

    def violation(self, samples: np.array, scaling: float) -> float:
        scaled_samples = scaling * samples
        scaled_threshold = scaling * self.threshold

        return (scaled_samples - scaled_threshold).clip(min=0).mean(axis=1)

    def __repr__(self) -> str:
        return f"MaxConstraint(max={self.threshold})"


class MinConstraint(Constraint):
    """
    Constraint of the form: x_i >= threshold
    """

    def __init__(self, threshold: float) -> None:
        super().__init__()

        self.threshold = threshold

    def violation(self, samples: np.array, scaling: float) -> float:
        scaled_samples = scaling * samples
        scaled_threshold = scaling * self.threshold

        return (scaled_threshold - scaled_samples).clip(min=0).mean(axis=1)

    def __repr__(self) -> str:
        return f"MinConstraint(min={self.threshold})"


class MeanEqualityConstraint(Constraint):
    """
    Constraint of the form: mean(x) == value
    """

    def __init__(self, value: float) -> None:
        super().__init__()

        self.value = value

    def violation(self, samples: np.array, scaling: float) -> float:
        scaled_samples = scaling * samples
        scaled_value = scaling * self.value

        return abs(scaled_value - scaled_samples.mean(axis=1))

    def __repr__(self) -> str:
        return f"MeanEqualityConstraint(mean={self.value})"
