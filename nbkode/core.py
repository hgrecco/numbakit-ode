"""
    nbkode.core
    ~~~~~~~~~~~

    Definition for Solver base class.


    :copyright: 2020 by nbkode Authors, see AUTHORS for more details.
    :license: BSD, see LICENSE for more details.
"""


from __future__ import annotations

from abc import ABC, ABCMeta, abstractmethod
from numbers import Real
from typing import Callable, Tuple, Union

import numpy as np

from .nbcompat import is_jitted, numba
from .util import CaseInsensitiveDict


class MetaSolver(ABCMeta):
    def __repr__(cls):
        return f"<{cls.__name__}>"


class Solver(ABC, metaclass=MetaSolver):
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
    params : array_like
        Extra arguments to be passed to the fun as ``fun(t, y, params)``
    t_bound : float, optional (default np.inf)
        The integration won’t continue beyond this value. Use it only to stop
        the integrator when the solution or ecuation has problems after this point.
        To obtain the solution at a given timepoint use `run`.
        In fixed step methods, the integration stops just before t_bound.
        In variable step methods, the integration stops at t_bound.

    Attributes
    ----------
    t : float
        Current time.
    y : ndarray
        Current state.
    f : ndarray
        last evaluation of the rhs.
    step_size : float
        Size of the last successful step. None if no steps were made yet.
    """

    SOLVERS = CaseInsensitiveDict()

    LEN_HISTORY: int = 2

    GROUP: str = None
    IMPLICIT: bool
    FIXED_STEP: bool

    #: Callable provided by the user
    #: The signature should be (t: float, y: ndarray)  -> ndarray
    #: or
    #: The signature should be (t: float, y: ndarray, p: ndarray)  -> ndarray
    rhs: Callable

    #: user rhs (same as rhs if it was originally jitted and with the right signature)
    user_rhs: Callable

    #: Last LEN_HISTORY times
    _ts: np.ndarray

    #: Last LEN_HISTORY states
    _ys: np.ndarray

    #: Last LEN_HISTORY evaluations of fun at (_t, _y)
    _ys: np.ndarray

    #: extra arguments for the user callable
    params: np.ndarray or None

    #: Function that build the _step function for a particular method.
    #: (*args) -> Callable
    _step_builder: Callable

    def __init__(
        self,
        rhs: Callable,
        t0: float,
        y0: np.ndarray,
        params: np.ndarray = None,
        t_bound=np.inf,
    ):
        self.t_bound = t_bound
        y0 = np.ascontiguousarray(y0)
        if params is not None:
            params = np.ascontiguousarray(params)

        self.user_rhs = rhs

        # TODO: check if it is jitted or njitted. Not sure if this is possible
        # if it has not been executed.
        if not is_jitted(rhs):
            rhs = numba.njit()(rhs)

        # TODO: A better way to make it partial?
        if params is None:
            self.rhs = rhs
        else:

            @numba.njit()
            def _rhs(t, y):
                return rhs(t, y, params)

            self.rhs = _rhs

        f = self.rhs(t0, y0)
        self._fs = np.full((self.LEN_HISTORY, y0.size), f, dtype=float)
        self._ts = np.full(self.LEN_HISTORY, t0, dtype=float)
        self._ys = np.full((self.LEN_HISTORY, y0.size), y0, dtype=float)

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if cls.LEN_HISTORY and cls.LEN_HISTORY < 1:
            raise ValueError(
                f"While defining {cls.__name__}, "
                f"LEN_HISTORY cannot be smaller than 1"
            )
        if cls.is_final_class():
            if cls.GROUP not in cls.SOLVERS:
                cls.SOLVERS[cls.GROUP] = []
            cls.SOLVERS[cls.GROUP].append(cls)
            cls._step = staticmethod(cls._step_builder(*cls._step_builder_args()))

    @classmethod
    @abstractmethod
    def _step_builder_args(cls):
        """Arguments provided to the _step_builder function."""

    @classmethod
    def is_final_class(cls):
        """True if the class represents a method and not a family/group of methods."""
        return cls.GROUP and not cls.__name__.startswith("_")

    @property
    def t(self):
        return self._ts[-1]

    @property
    def y(self):
        return self._ys[-1]

    @property
    def f(self):
        return self._fs[-1]

    def _check_time(self, t):
        if t > self.t_bound:
            raise ValueError(
                f"Time {t} is larger than solver bound time t_bound={self.t_bound}"
            )

    def step(self, *, n: int = None, upto_t: float = None) -> Tuple[np.array, np.array]:
        """Advance simulation `n` steps or until the next timepoint will go beyond `upto_t`.

        It records and output all intermediate steps.

        - `step()` is equivalent to `step(n=1)`
        - `step(n=<number>)` is equivalent to `step(n=<number>, upto_t=np.inf)`
        - `step(upto_t=<number>)` is similar to `step(n=`np.inf`, upto_t=<number>)`

        If `upto_t < self.t`, returns empty arrays for time and state.

        Parameters
        ----------
        n : int, optional
            Number of steps.
        upto_t : float, optional

        Returns
        -------
        np.ndarray, np.ndarray
            time vector, state array

        Raises
        ------
        ValueError
            One of the timepoints provided is outside the valid range.
        RuntimeError
            The integrator reached `t_bound`.
        """
        if upto_t is not None and upto_t < self.t:
            return np.asarray([]), np.asarray([])

        if n is None and upto_t is None:
            # No parameters, make one step.
            if self._step(self.t_bound, *self._step_args(), *self._step_extra_args()):
                return np.atleast_1d(self.t), self.y
        elif upto_t is None:
            # Only n is given, make n steps. If t_bound is reached, raise an exception.
            ts, ys, scon = self._nsteps(
                n,
                self.t_bound,
                self._step,
                *self._step_args(),
                *self._step_extra_args(),
            )
            if scon:
                raise RuntimeError("Integrator reached t_bound.")
            return ts, ys
        elif n is None:
            # Only upto_t is given, move until that value.
            # t_bound will not be reached a it due to validation in _check_time
            self._check_time(upto_t)
            ts, ys, scon = self._steps(
                upto_t, self._step, *self._step_args(), *self._step_extra_args()
            )
            return ts, ys
        else:
            # Both parameters are given, move until either condition is reached.
            # t_bound will not be reached a it due to validation in _check_time
            self._check_time(upto_t)
            ts, ys, scon = self._nsteps(
                n, upto_t, self._step, *self._step_args(), *self._step_extra_args()
            )
            return ts, ys

    def skip(self, *, n: int = None, upto_t: float = None) -> None:
        """Advance simulation `n` steps or until the next timepoint will go beyond `upto_t`.

        Unlike `step` or `run`, this method does not output the time and state.

        - `skip()` is equivalent to `skip(n=1)`
        - `skip(n=<number>)` is equivalent to `skip(n=<number>, upto_t=np.inf)`
        - `skip(upto_t=<number>)` is similar to `skip(n=`np.inf`, upto_t=<number>)`

        If `upto_t < self.t`, does nothing.

        Parameters
        ----------
        n : int, optional
            Number of steps.
        upto_t : float, optional
            Time to reach.

        Raises
        ------
        ValueError
            One of the timepoints provided is outside the valid range.
        RuntimeError
            The integrator reached `t_bound`.
        """
        if upto_t is not None and upto_t < self.t:
            return

        if n is None and upto_t is None:
            # No parameters, make one step.
            self._nskip(
                1,
                self.t_bound,
                self._step,
                *self._step_args(),
                *self._step_extra_args(),
            )
        elif upto_t is None:
            # Only n is given, make n steps. If t_bound is reached, raise an exception.
            if self._nskip(
                n,
                self.t_bound,
                self._step,
                *self._step_args(),
                *self._step_extra_args(),
            ):
                raise RuntimeError("Integrator reached t_bound.")
        elif n is None:
            # Only upto_t is given, move until that value.
            # t_bound will not be reached a it due to validation in _check_time
            self._check_time(upto_t)
            self._skip(upto_t, self._step, *self._step_args(), *self._step_extra_args())
        else:
            # Both parameters are given, move until either condition is reached.
            # t_bound will not be reached a it due to validation in _check_time
            self._check_time(upto_t)
            self._nskip(
                n, upto_t, self._step, *self._step_args(), *self._step_extra_args()
            )

    def run(self, t: Union[Real, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """Integrates the ODE interpolating at each of the timepoints `t`.

        Parameters
        ----------
        t : float or array-like

        Returns
        -------
        np.ndarray, np.ndarray
            time vector, state vector

        Raises
        ------
        ValueError
            One of the timepoints provided is outside the valid range.
        """
        t = np.atleast_1d(t)

        is_t_sorted = t.size == 1 or np.all(t[:-1] <= t[1:])

        if not is_t_sorted:
            ndx = np.argsort(t)
            t = t[ndx]

        if t[0] < self._ts[0]:
            raise ValueError(
                f"Cannot interpolate at t={t[0]} as it is smaller "
                f"than the current smallest value in history ({self._ts[0]})"
            )

        self._check_time(np.max(t))

        to_interpolate = t <= self._ts[-1]
        is_to_interpolate = np.any(to_interpolate)
        if is_to_interpolate:
            t_old = t[to_interpolate]
            y_old = np.asarray([self.interpolate(_t) for _t in t_old])
            t_to_run = t[np.logical_not(to_interpolate)]
        else:
            t_to_run = t

        # t_bound will not be reached a it due to validation in _check_time
        ts, ys, scon = self._run_eval(
            self.t_bound,
            t_to_run,
            self._step,
            self._interpolate,
            *self._step_args(),
            *self._step_extra_args(),
        )

        if is_to_interpolate:
            ts = np.concatenate((t_old, ts))
            ys = np.concatenate((y_old, ys))

        if is_t_sorted:
            return ts, ys

        ondx = np.argsort(ndx)
        return ts[ondx], ys[ondx]

    def interpolate(self, t: float) -> float:
        """Interpolate solution at t.

        This only works for values within the recorded history of the solver.
        of the solver instance

        Parameters
        ----------
        t : float

        Raises
        ------
        ValueError
            if the time is outside the recorded history.
        """

        # TODO: make this work for array T

        if not (self._ts[0] <= t <= self._ts[-1]):
            raise ValueError(f"Time {t} to interpolate outside range")

        return self._interpolate(t, *self._step_args(), *self._step_extra_args())

    def _step_args(self):
        return self.rhs, self._ts, self._ys, self._fs

    def _step_extra_args(self) -> tuple:
        """Arguments to be passed to _step after t_bound."""
        return tuple()

    @staticmethod
    @abstractmethod
    def _step(t_bound, rhs, ts, ys, fs, *extra_args) -> int:
        """Perform one integration step."""
        raise NotImplementedError

    @staticmethod
    @numba.njit
    def _steps(t_end, step, rhs, ts, ys, fs, *extra_args):
        """Step forward until:
            - the next step goes beyond `t_end`

        The stop condition is in the output to unify the API with
        `nsteps`

        Returns
        -------
        np.ndarray, np.ndarray, bool
            time vector, state array, stop condition (always True)
        """
        t_out = []
        y_out = []

        while step(t_end, rhs, ts, ys, fs, *extra_args):
            t_out.append(ts[-1])
            y_out.append(np.copy(ys[-1]))

        out = np.empty((len(y_out), ys.shape[1]))
        for ndx, yi in enumerate(y_out):
            out[ndx] = yi

        return np.array(t_out), out, True

    @staticmethod
    @numba.njit
    def _nsteps(n_steps, t_end, step, rhs, ts, ys, fs, *extra_args):
        """Step forward until:
            - the next step goes beyond `t_end`
            - `n_steps` steps are done.

        Returns
        -------
        np.ndarray, np.ndarray, bool
            time vector, state array, stop condition

        Stop condition
            True if the integrator stopped due to the time condition.
            False, otherwise (it was able to run all all steps).
        """

        t_out = np.empty((n_steps,))
        y_out = np.empty((n_steps, ys.shape[-1]))

        for ndx in range(n_steps):

            if not step(t_end, rhs, ts, ys, fs, *extra_args):
                return t_out[:ndx], y_out[:ndx], True

            t_out[ndx] = ts[-1]
            y_out[ndx] = ys[-1]

        return t_out, y_out, False

    @staticmethod
    @numba.njit
    def _skip(t_end, step, rhs, ts, ys, fs, *extra_args) -> bool:
        """Perform all steps required, stopping just before going beyond t_end.

        The stop condition is in the output to unify the API with `nsteps`

        Returns
        -------
        bool
            stop_condition (always True)
        """

        while step(t_end, rhs, ts, ys, fs, *extra_args):
            pass
        return True

    @staticmethod
    @numba.njit
    def _nskip(n_steps, t_end, step, rhs, ts, ys, fs, *extra_args) -> bool:
        """Step forward until:
            - the next step goes beyond `t_end`
            - `n_steps` steps are done.

        Returns
        -------
        np.ndarray, np.ndarray, bool
            time vector, state array, stop condition

        Stop condition
            True if the integrator stopped due to the time condition.
            False, otherwise (it was able to run all all steps).
        """
        for _ in range(n_steps):
            if not step(t_end, rhs, ts, ys, fs, *extra_args):
                return True
        return False

    @staticmethod
    @numba.njit()
    def _interpolate(t_eval, rhs, ts, ys, fs, *extra_args):
        """Interpolate solution at t_eval"""
        y_out = np.empty(ys[0].shape)
        for ndx in range(len(y_out)):
            y_out[ndx] = np.interp(t_eval, ts, ys[:, ndx])

        return y_out

    @staticmethod
    @numba.njit
    def _run_eval(
        t_bound: float,
        t_eval: np.ndarray,
        step,
        interpolate,
        rhs,
        ts,
        ys,
        fs,
        *extra_args,
    ) -> tuple[np.ndarray, np.ndarray, bool]:
        """Run up to t, evaluating y at given t and return (t, y) as arrays."""

        y_out = np.empty((t_eval.size, ys.shape[1]))

        for ndx, ti in enumerate(t_eval):
            while ts[-1] < ti:
                if not step(t_bound, rhs, ts, ys, fs, *extra_args):
                    return t_eval[:ndx], y_out[:ndx], True
            y_out[ndx] = interpolate(ti, rhs, ts, ys, fs, *extra_args)

        return t_eval, y_out, False


def check(solver, implicit=None, fixed_step=None):
    if implicit is not None:
        if solver.IMPLICIT is not implicit:
            return False
    if fixed_step is not None:
        if solver.FIXED_STEP is not fixed_step:
            return False
    return True


def get_solvers(*groups, implicit=None, fixed_step=None):
    """Get available solvers.

    Parameters
    ----------
    groups : str
        name of the group to filter
    implicit : bool
        if True, only implicit solvers will be returned.
    fixed_step : bool
        if True, only fixed step solvers will be returned.

    Returns
    -------
    tuple(Solver)
    """
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


def get_groups():
    """Get group names."""
    return tuple(sorted(Solver.SOLVERS.keys()))
