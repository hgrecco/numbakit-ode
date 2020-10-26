"""
    nbkode.core
    ~~~~~~~~~~~

    Definition for Solver base class.


    :copyright: 2020 by nbkode Authors, see AUTHORS for more details.
    :license: BSD, see LICENSE for more details.
"""


from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Collection
from numbers import Real
from typing import Callable, Tuple, Union

import numpy as np

from .nbcompat import numba
from .util import CaseInsensitiveDict


class Solver(ABC):
    """Base class for all solvers

    Parameters
    ----------
    rhs : callable
        Right-hand side of the system. The calling signature is ``fun(t, y)``.
        Here ``t`` is a scalar, andthe ndarray ``y`` hasna shape (n,);
        then ``fun`` must return array_like with shape (n,).
    t0 : float
        Initial time.
    y0 : array_like, shape (n,)
        Initial state.
    args : tuple
        Extra arguments to be passed to the fun as ``fun(t, y, *args)``
    t_bound : float
        Boundary time - the integration won't continue beyond it. It also
        determines the direction of the integration.
    Attributes
    ----------
    t : float
        Current time.
    y : ndarray
        Current state.
    t_prev : float
        Last time
    y_prev : ndarray
        Last step
    t_old : float
        Previous time. None if no steps were made yet.
    step_size : float
        Size of the last successful step. None if no steps were made yet.
    """

    SOLVERS = CaseInsensitiveDict()

    LEN_HISTORY: int = 2

    GROUP: str = None
    IMPLICIT: bool
    FIXED_STEP: bool

    #: Callable provided by the user
    rhs: Callable

    #: Reduced and njitted rhs
    #: (compiled and arguments fixed)
    jrhs: Callable

    #: Last LEN_HISTORY times
    _ts: np.ndarray

    #: Last LEN_HISTORY states
    _ys: np.ndarray

    #: Last LEN_HISTORY evaluations of fun at (_t, _y)
    _ys: np.ndarray

    #: extra arguments for the user callable
    args: np.ndarray

    def __init__(self, rhs: Callable, t0: float, y0: np.ndarray, args: tuple = ()):

        y0 = np.asarray(y0)

        self.rhs = _rhs = rhs

        if not hasattr(_rhs, "inspect_llvm"):
            _rhs = numba.njit()(rhs)
            _ = _rhs(t0, y0, *args)

        @numba.njit()
        def jrhs(t, y):
            return _rhs(t, y, *args)

        f = jrhs(t0, y0)
        self.jrhs = jrhs
        self._fs = np.full((self.LEN_HISTORY, y0.size), f, dtype=float)
        self._ts = np.full(self.LEN_HISTORY, t0, dtype=float)
        self._ys = np.full((self.LEN_HISTORY, y0.size), y0, dtype=float)

    def __init_subclass__(cls, **kwargs):
        if cls.LEN_HISTORY and cls.LEN_HISTORY < 1:
            raise ValueError(
                f"While defining {cls.__name__}, "
                f"LEN_HISTORY cannot be smaller than 1"
            )
        if cls.GROUP and not cls.__name__.startswith("_"):
            cls.SOLVERS[cls.GROUP].append(cls)

    @property
    def t(self):
        return self._ts[-1]

    @property
    def y(self):
        return self._ys[-1]

    @property
    def f(self):
        return self._fs[-1]

    @property
    def t_old(self):
        return self._ts[-2]

    @property
    def y_old(self):
        return self._ys[-2]

    @property
    def f_old(self):
        return self._fs[-2]

    def run(self, t: Union[Real, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """Run solver.
        If t is a scalar, run freely up to t.
        If t is an array-like, return solution evaluated at t.
        """
        if isinstance(t, Real):
            self._check_time(t)
            return self._run_free(t)
        elif isinstance(t, Collection):
            t = np.asarray(t)
            self._check_time(t[0])
            return self._run_eval(t)

    def move_to(self, t: float):
        """Advance simulation upto t.
        Saves the result in self.t, self.y
        """
        self._check_time(t)
        self._move_to(t)

    def _check_time(self, t):
        if t < self.t:
            raise ValueError(f"Time {t} is smaller than current solver time t={self.t}")

    @abstractmethod
    def _step(t_bound, rhs, t, y, f, *extra_args):
        """Numba-compiled."""
        raise NotImplementedError

    def _steps_args(self):
        return self.jrhs, self._ts, self._ys, self._fs

    def _steps_extra_args(self) -> tuple:
        """Arguments to be passed to _step after t_bound.
        """
        return tuple()

    @staticmethod
    @numba.njit()
    def _interpolate(t_eval, rhs, t, y, f, *extra_args):
        """Interpolate solution at t_eval
        """
        if not (t[0] <= t_eval <= t[-1]):
            raise ValueError(f"Time to interpolate outside range")

        y_out = np.empty(y[0].shape)
        for ndx in range(len(y_out)):
            y_out[ndx] = np.interp(t_eval, t, y[:, ndx])

        return y_out

    def step(self, t_bound=np.inf):
        """Advance simulation one time step.

        Modifies in-place self.t, self.y, self.f
        """
        self._step(t_bound, *self._steps_args(), *self._steps_extra_args())

    def interpolate(self, t):
        """Interpolate solution at t."""
        return self._interpolate(t, *self._steps_args(), *self._steps_extra_args())

    def _move_to(self, t: float):
        """Move solver to time t.

        If input t is current solver time, do nothing.
        """
        self._move_to_numba(
            t, self._step, *self._steps_args(), *self._steps_extra_args()
        )

    @staticmethod
    @numba.njit
    def _move_to_numba(t_end: float, step, rhs, t, y, f, *extra_args):
        while t[-1] < t_end:
            step(t_end, rhs, t, y, f, *extra_args)

    def _run_free(self, t: float) -> tuple[np.ndarray, np.ndarray]:
        """Run freely upto t and return (t, y) as arrays."""
        return self._run_free_numba(
            t, self._step, *self._steps_args(), *self._steps_extra_args()
        )

    @staticmethod
    @numba.njit
    def _run_free_numba(
        t_end: float, step, rhs, t, y, f, *extra_args
    ) -> tuple[np.ndarray, np.ndarray]:
        """Run freely upto t and return (t, y) as arrays."""
        t_out = []
        t_out.append(t[-1])
        y_out = []
        y_out.append(y[-1])

        while t[-1] < t_end:

            step(t_end, rhs, t, y, f, *extra_args)
            t_out.append(t[-1])
            y_out.append(y[-1])

        _t_out = np.empty(len(t_out))
        _y_out = np.empty((len(t_out), y[-1].size))
        for ndx in range(len(t_out)):
            _t_out[ndx] = _t_out[ndx]
            _y_out[ndx] = _y_out[ndx]

        return _t_out, _y_out

    def _run_eval(self, t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Run up to t, evaluating y at given t and return (t, y) as arrays.
        """
        return self._run_free_numba(
            t,
            self._step,
            self._interpolate,
            *self._steps_args(),
            *self._steps_extra_args(),
        )

    @staticmethod
    @numba.njit
    def _run_eval_numba(
        t_eval: np.ndarray, step, interpolate, rhs, t, y, f,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Run up to t, evaluating y at given t and return (t, y) as arrays.
        """

        t_bound = t_eval[-1]
        y_eval = np.empty((t_eval.size, y.size))

        for i, ti in enumerate(t):
            while t[-1] < ti:
                step(t_bound, rhs, t, y, f)
            y_eval[i] = interpolate(ti, rhs, t, y, f)

        return t_eval, y_eval


def check(solver, implicit=None, fixed_step=None):
    if implicit is not None:
        if solver.IMPLICIT is not implicit:
            return False
    if fixed_step is not None:
        if solver.FIXED_STEP is not fixed_step:
            return False
    return True


def get_solvers(*groups, implicit=None, fixed_step=None):
    if not groups:
        groups = Solver.SOLVERS.keys()
    out = []
    for group in groups:
        try:
            out.extend(
                filter(
                    lambda solver: check(solver, implicit, fixed_step),
                    Solver.SOLVERS[group],
                )
            )
        except KeyError:
            m = tuple(Solver.SOLVERS.keys())
            raise KeyError(f"Group {group} not found. Valid values: {m}")
    return tuple(out)