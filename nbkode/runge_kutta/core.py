from __future__ import annotations

import numpy as np

from ..core import Solver, VariableStep
from ..nbcompat import numba
from ..util import classproperty


class RungeKutta(Solver, abstract=True):
    GROUP = "Runge-Kutta"

    FIXED_STEP = True

    # class attributes
    LEN_HISTORY = 2
    A: np.ndarray  # (n, n)
    B: np.ndarray  # (n,)
    C: np.ndarray  # (n,)
    order: float

    # instance attributes
    K: np.ndarray

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.K = np.empty((self.stages, self.y.size))

    @staticmethod
    def _fixed_step():
        raise NotImplementedError

    @classmethod
    def _step_builder(cls):
        fixed_step = cls._fixed_step

        @numba.njit
        def _step(rhs, cache, h, K):
            t, y = fixed_step(rhs, cache, h, K)
            cache.push(t, y, rhs(t, y))

        return _step

    def _step_args(self):
        return self.rhs, self.cache, self.h, self.K

    @classproperty
    def stages(cls):
        return len(cls.A)

    @classproperty
    def consistent(cls):
        return np.isclose(np.sum(cls.B), 1)


class AdaptiveRungeKutta(VariableStep, RungeKutta, abstract=True):
    FIXED_STEP = False

    # class attributes
    B2: np.ndarray  # (n,)
    E: np.ndarray  # (n,)
    error_estimator_order: int

    @classproperty
    def E(cls):
        """Error coefficients."""
        return cls.B - cls.B2

    @classproperty
    def error_exponent(cls):
        return 1 / (cls.error_estimator_order + 1)

    @staticmethod
    @numba.njit(inline="always")
    def _step_error(h, K, E):
        return h * (E @ K)

    @staticmethod
    @numba.njit(inline="always")
    def _scaled_error_norm(y, y_prev, error, options):
        scale = options.atol + options.rtol * np.maximum(np.abs(y), np.abs(y_prev))
        error_norm = np.sqrt(np.mean((error / scale) ** 2))
        return error_norm

    @staticmethod
    @numba.njit(inline="always")
    def _step_update(error_norm, h, options, error_exponent) -> bool:
        """Checks if step is accepted and updates step size h.

        Updates h in place, so it must be an scalar numpy array.
        Returns True if step is accepted.
        """
        if error_norm == 0:
            h *= options.max_factor
            return True  # step accepted

        factor = options.safety_factor / error_norm ** error_exponent
        h *= min(options.max_factor, max(options.min_factor, factor))
        return error_norm < 1

    @classmethod
    def _step_builder(cls):
        E, error_exponent = cls.E, cls.error_exponent
        fixed_step = cls._fixed_step
        step_error = cls._step_error
        scaled_error_norm = cls._scaled_error_norm
        step_update = cls._step_update

        @numba.njit
        def _step(t_bound, rhs, cache, h, K, options):
            if cache.t + h > t_bound:
                return False

            while True:
                t, y = fixed_step(rhs, cache, h, K)
                error = step_error(h, K, E)
                error_norm = scaled_error_norm(y, cache.y, error, options)
                if step_update(error_norm, h, options, error_exponent):
                    cache.push(t, y, rhs(t, y))
                    break

            return True

        return _step

    @property
    def _step_args(self):
        return self.rhs, self.cache, self.h, self.K, self.options
