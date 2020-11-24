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

    ORDER: int

    @classproperty
    def ORDER(cls):
        return cls.A.size

    @classproperty
    def IMPLICIT(cls):
        return cls.Bn == 0

    @classproperty
    def LEN_HISTORY(cls):
        return max(cls.ORDER, 2)

    def __init__(self, *args, first_stepper_cls="auto", **kwargs):
        super().__init__(*args, **kwargs)

        if first_stepper_cls is None or self.ORDER == 1:
            # We push 1 less because one was done at Solver.s
            for _ in range(self.LEN_HISTORY - 1):
                self.cache.push(self.t, self.y, self.f)
            return

        if first_stepper_cls == "auto":
            first_stepper_cls = self.FIRST_STEPPER_CLS

        if isinstance(first_stepper_cls, str):
            import nbkode

            first_stepper_cls = getattr(nbkode, first_stepper_cls)

        if first_stepper_cls.FIXED_STEP:
            # For fixed step solver we do N steps with the step size.
            solver = first_stepper_cls(self.rhs, self.t, self.y, h=self.h)
            for _ in range(self.ORDER - 1):
                solver.step()
                self.cache.push(solver.t, solver.y, solver.f)
        else:
            # For variable step solver we run N times until h, 2h, 3h .. (ORDER - 1)h
            solver = first_stepper_cls(self.rhs, self.t, self.y)
            ts, ys = solver.run(np.arange(1, self.ORDER) * self.h)
            for t, y in zip(ts, ys):
                self.cache.push(t, y, self.rhs(t, y))

    @staticmethod
    def _fixed_step():
        raise NotImplementedError

    @classmethod
    def _step_builder(cls):
        fixed_step = cls._fixed_step

        @numba.njit
        def _step(rhs, cache, h):
            t, y = fixed_step(rhs, cache, h)
            cache.push(t, y, rhs(t, y))

        return _step

    @property
    def _step_args(self):
        return self.rhs, self.cache, self.h


class ExplicitMultistep(Multistep, abstract=True):
    Bn = 0

    @classmethod
    def _fixed_step_builder(cls):
        A, B = cls.A, cls.B

        if A.size < cls.LEN_HISTORY or B.size < cls.LEN_HISTORY:

            deltaA = cls.LEN_HISTORY - A.size
            deltaB = cls.LEN_HISTORY - B.size

            @numba.njit
            def _fixed_step(rhs, cache, h):
                t_new = cache.t + h
                y_new = h * B @ cache.fs[deltaB:] - A @ cache.ys[deltaA:]
                return t_new, y_new

        else:

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

        if A.size < cls.LEN_HISTORY or B.size < cls.LEN_HISTORY:

            deltaA = cls.LEN_HISTORY - A.size
            deltaB = cls.LEN_HISTORY - B.size

            @numba.njit
            def _fixed_step(rhs, cache, h):
                t_new = cache.t + h
                K = h * B @ cache.fs[deltaB:] - A @ cache.ys[deltaA:]

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

        else:

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
