import numpy as np

from ..nbcompat import numba
from ..nbcompat.zeros import newton_hd
from ..order import Implicit, Solver
from ..util import classproperty
from .core import AdaptiveRungeKutta, RungeKutta


class FIRK(RungeKutta, Implicit):
    """Fully Implicit Runge-Kutta (FIRK) methods."""

    def __init_subclass__(cls, *args, **kwargs) -> None:
        assert cls.implicit
        super().__init_subclass__(*args, **kwargs)

    @classproperty
    def implicit(cls):
        return np.any(np.triu(cls.A) != 0)

    @property
    def _step_args(self):
        return self.rhs, self.cache, self.h, self.K

    @classmethod
    def _build_fixed_step(cls):
        A, B, C = cls.A, cls.B, cls.C
        root_solver = newton_hd

        @numba.njit(inline="always")
        def implicit_K(K, rhs, t, y, h):
            K_new = np.empty_like(K)
            for i, (a, c, k) in enumerate(zip(A, C, K)):
                K_new[i] = k - rhs(t + h * c, y + h * a @ K)
            return K_new

        @numba.njit(inline="always")
        def _fixed_step(rhs, cache, h, K):
            t = cache.t
            y = cache.y

            K_new = root_solver(implicit_K, K, args=(rhs, t, y, h))
            t = t + h
            y = y + h * B @ K_new
            return t, y

        return _fixed_step


class AdaptiveFIRK(AdaptiveRungeKutta):
    @staticmethod
    @numba.njit
    def _step_error():
        """Shampine error estimate.

        From equation 8.19, Hairer."""
        raise NotImplementedError


class RadauIA3(FIRK, Solver):
    """Radau IA method of order 3

    From Hairer section IV.5, table 5.3."""

    A = np.array([[1, -1], [1, 5 / 3]]) / 4
    B = np.array([1, 3]) / 4
    C = np.array([0, 2 / 3])


class RadauIA5(FIRK, Solver):
    """Radau IA method of order 5

    From Hairer section IV.5, table 5.4."""

    # A =
    B = (np.array([4, 16, 16]) + np.sqrt(6) * np.array([0, 1, -1])) / 36
    C = (1 + np.sqrt(6) * np.array([0, -1, 1])) / 10


class RadauIIA3(FIRK, Solver):
    """Radau IIA method of order 3

    From Hairer section IV.5, table 5.5."""

    A = np.array([[5, -1], [9, 3]]) / 12
    B = np.array([3, 1]) / 4
    C = np.array([1, 3]) / 3


class RadauIIA(FIRK, Solver):
    """Radau IIA method of order 5

    From Hairer section IV.5, table 5.6."""

    A = np.array(
        [
            [
                (88 - 7 * np.sqrt(6)) / 360,
                (296 - 169 * np.sqrt(6)) / 1800,
                (-2 + 3 * np.sqrt(6)) / 225,
            ],
            [
                (296 + 169 * np.sqrt(6)) / 1800,
                (88 + 7 * np.sqrt(6)) / 360,
                (-2 - 3 * np.sqrt(6)) / 225,
            ],
            [(16 - 1 * np.sqrt(6)) / 36, (16 + 1 * np.sqrt(6)) / 36, 1 / 9],
        ]
    )
    C = np.array([(4 - np.sqrt(6)) / 10, (4 + np.sqrt(6)) / 10, 1])
    B = np.array([(16 - np.sqrt(6)) / 36, (16 + np.sqrt(6)) / 36, 1 / 9])


class AdaptiveRadauIIA(RadauIIA, AdaptiveFIRK):
    # This is for a different solver, diagonalizing A.
    # E = np.array([-13 - 7 * np.sqrt(6), -13 + 7 * np.sqrt(6), -1]) / 3
    # error_estimator_order = 3
    pass


from ..nbcompat import numba

func = numba.njit(lambda t, y: -y)

s = RadauIIA(func, 0, np.array([1]), h=0.1, root_solver=newton_hd)
s._step(s.rhs, s.cache, *s._step_args)
