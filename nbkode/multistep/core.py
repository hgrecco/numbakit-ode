from __future__ import annotations

import numpy as np

from ..core import Solver
from ..nbcompat import numba
from ..nbcompat.zeros import j_newton, newton_hd
from ..util import classproperty


class Multistep(Solver, abstract=True):
    GROUP = "Multistep"

    FIXED_STEP = True
    FIRST_STEPPER_CLS = "RungeKutta45"

    # class attributes
    A: np.ndarray  # (n,)
    B: np.ndarray  # (n,)
    Bn: float
    order: float

    @classproperty
    def IMPLICIT(cls):
        return cls.Bn == 0

    @classproperty
    def LEN_HISTORY(cls):
        return len(cls.A)

    def __init__(self, *args, first_stepper_cls=None, **kwargs):
        super().__init__(*args, **kwargs)

        if self.LEN_HISTORY == 1:
            return

        if first_stepper_cls is None:
            first_stepper_cls = self.FIRST_STEPPER_CLS
        if isinstance(first_stepper_cls, str):
            import nbkode

            first_stepper_cls = getattr(nbkode, first_stepper_cls)

        if first_stepper_cls.FIXED_STEP:
            # For fixed step solver we do N steps with the step size.
            solver = first_stepper_cls(self.rhs, self.t, self.y, h=self.h)
            for _ in range(1, self.LEN_HISTORY):
                solver.step()
                self.cache.t = solver.t
                self.cache.y = solver.y
                self.cache.f = solver.f
        else:
            # For variable step solver we run N times until h, 2h, 3h .. (ORDER - 1)h
            solver = first_stepper_cls(self.rhs, self.t, self.y)
            ts, ys = solver.run(np.arange(1, self.LEN_HISTORY) * self.h)
            for t, y in zip(ts, ys):
                self.cache.t = t
                self.cache.y = y
                self.cache.f = self.rhs(t, y)

    @staticmethod
    def _fixed_step():
        raise NotImplementedError

    @classmethod
    def _step_builder(cls):
        fixed_step = cls._fixed_step

        @numba.njit
        def _step(t_bound, rhs, cache, h):
            if cache.t + h > t_bound:
                return False

            t, y = fixed_step(rhs, cache, h)
            cache.t = t
            cache.y = y
            cache.f = rhs(t, y)
            return True

        return _step

    @property
    def _step_args(self):
        return self.rhs, self.cache, self.h


class ExplicitMultistep(Multistep, abstract=True):
    Bn = 0

    @classmethod
    def _fixed_step_builder(cls):
        A, B = cls.A, cls.B

        @numba.njit
        def _fixed_step(rhs, cache, h):
            t_new = cache.t + h
            y_new = h * B @ cache.fs - A @ cache.ys
            return t_new, y_new

        return _fixed_step


class ImplicitMultistep(Multistep, abstract=True):
    @classmethod
    def _fixed_step_builder(cls):
        A, B, Bn = cls.A, cls.B, cls.Bn

        @numba.njit(inline="always")
        def implicit_root(y, rhs, t, h_Bn, K):
            return y - h_Bn * rhs(t, y) - K

        @numba.njit
        def _fixed_step(rhs, cache, h):
            t_new = cache.t + h
            K = h * B @ cache.fs - A @ cache.ys

            if cache.y.size == 1:
                y_new = j_newton(
                    implicit_root,
                    cache.y,
                    args=(rhs, t_new, h * Bn, K),
                )
            else:
                y_new = newton_hd(
                    implicit_root,
                    cache.y,
                    args=(rhs, t_new, h * Bn, K),
                )

            return t_new, y_new

        return _fixed_step
