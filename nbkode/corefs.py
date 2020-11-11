"""
    nbkode.corefs
    ~~~~~~~~~~~~~

    Definitions for linear multistep methods.

    See: https://en.wikipedia.org/wiki/Linear_multistep_method

    :copyright: 2020 by nbkode Authors, see AUTHORS for more details.
    :license: BSD, see LICENSE for more details.
"""


from typing import Callable

import numpy as np

from nbkode.nbcompat import numba

from .core import Solver
from .nbcompat import j_newton, newton_hd


def forward_step_builder(A):
    """Perform a single fixed step.

    This outer function should only contains attributes
    associated with the solver class not with the solver instance.

    Parameters
    ----------
    A : ndarray, shape (1, ORDER + 1)
        Coefficients for combining previous stages to compute the next
        stage.

    Returns
    -------
    callable
    """

    @numba.njit
    def _step(t_bound, rhs, ts, ys, fs, h):
        """Perform a single fixed step.

        This inner function should only contains attributes
        associated with the solver instance not with the solver class.

        Parameters
        ----------
        t_bound : float
            Boundary time - the integration won’t continue beyond it.
            It also determines the direction of the integration.
        rhs : Callable
            Right-hand side of the system.
        ts : ndarray, shape (LEN_HISTORY, )
            Last ORDER + 1 times
        ys : ndarray, shape (LEN_HISTORY, n)
            Last ORDER + 1 values, for each n dimension
        fs : ndarray, shape (LEN_HISTORY, n)
            Last ORDER + 1 value of the derivative, for each n dimension
        h : float
            Step to use.

        ts, ys and fs are modified in place.

        Returns
        -------
        bool
            True if a step was done, False otherwise.
        """

        t_new = ts[-1] + h
        if t_new > t_bound:
            return False
        f_new = rhs(ts[-1], ys[-1])

        fs[:-1] = fs[1:]
        fs[-1] = f_new

        y_new = ys[-1] + h * A @ fs

        ts[:-1] = ts[1:]
        ts[-1] = t_new

        ys[:-1] = ys[1:]
        ys[-1] = y_new

        return True

    return _step


def backward_step_builder(A):
    """Perform a single fixed step.

    This outer function should only contains attributes
    associated with the solver class not with the solver instance.

    Parameters
    ----------
    A : ndarray, shape (n_stages, )
        Coefficients for combining previous stages to compute the next
        stage.

    Returns
    -------
    callable

    """

    @numba.njit
    def _to_solve(y_new, rhs, t_new, y_last, f, h):
        f[-1] = rhs(t_new, y_last)

        return y_last + h * (A @ f).ravel() - y_new

    @numba.njit
    def _step(t_bound, rhs, ts, ys, fs, h, rtol, atol, max_iter):
        """Perform a single fixed step.

        This inner function should only contains attributes
        associated with the solver instance not with the solver class.

        Parameters
        ----------
        t_bound : float
            Boundary time - the integration won’t continue beyond it.
            It also determines the direction of the integration.
        rhs : Callable
            Right-hand side of the system.
        ts : ndarray, shape (LEN_HISTORY, )
            Last ORDER + 1 times
        ys : ndarray, shape (LEN_HISTORY, n)
            Last ORDER + 1 values, for each n dimension
        fs : ndarray, shape (LEN_HISTORY, n)
            Last ORDER + 1 value of the derivative, for each n dimension
        h : float
            Step to use.
        rtol : float, optional
            Tolerance (relative) for termination.
        atol : float, optional
            The allowable error of the zero value. If `func` is complex-valued,
            a larger `tol` is recommended as both the real and imaginary parts
            of `x` contribute to ``|x - x0|``.
        max_iter : int, optional
            Maximum number of iterations.

        ts, ys and fs are modified in place.

        Returns
        -------
        bool
            True if a step was done, False otherwise.
        """

        t_new = ts[-1] + h
        if t_new > t_bound:
            return False

        fs[:-1] = fs[1:]

        # TODO: For size > 1, we need something like this.
        # y_new = root(_to_solve, ys[-1], args=(rhs, t_new, ys[-1], fs)).x
        if ys[-1].size == 1:
            y_new = j_newton(
                _to_solve,
                ys[-1],
                args=(rhs, t_new, ys[-1], fs, h),
                tol=atol,
                rtol=rtol,
                maxiter=max_iter,
            )
        else:
            y_new = newton_hd(
                _to_solve,
                ys[-1],
                args=(rhs, t_new, ys[-1], fs, h),
                atol=atol,
                rtol=rtol,
                maxiter=max_iter,
            )
        ts[:-1] = ts[1:]
        ts[-1] = t_new

        ys[:-1] = ys[1:]
        ys[-1] = y_new

        return True

    return _step


class _FixedStepBaseSolver(Solver):
    """Forward fixed step solver.

    Parameters
    ----------
    h : float
        Step size
    first_stepper_cls : Solver or 'str' or None
        - subclass of Solver.
        - 'auto'
        - str that will be used as getattr(nbkode, first_stepper)
        - None

    See `Solver` for information about the other arguments.
    """

    COEFS: np.ndarray

    ORDER: int

    FIXED_STEP = True

    FIRST_STEPPER_CLS = "RungeKutta45"

    @classmethod
    def _step_builder_args(cls):
        return (cls.COEFS,)

    def __init__(
        self,
        rhs: Callable,
        t0: float,
        y0: np.ndarray,
        params: np.ndarray = None,
        *,
        t_bound=np.inf,
        h: float = 1,
        first_stepper_cls=None,
    ):
        super().__init__(rhs, t0, y0, params, t_bound=t_bound)

        # TODO: CHECK VALID VALUES
        self._h = h

        # Code to handle the first steps in multistep methods
        # Some of these methods require to build up a history of states
        # integrated using different methods.
        if first_stepper_cls == "auto":
            first_stepper_cls = self.FIRST_STEPPER_CLS

        if first_stepper_cls is not None:
            if isinstance(first_stepper_cls, str):
                import nbkode

                first_stepper_cls = getattr(nbkode, first_stepper_cls)

            if first_stepper_cls.FIXED_STEP:
                # For fixed step solver we do N steps with the step size.
                solver = first_stepper_cls(rhs, t0, y0, params, h=h)
                for ndx in range(self.ORDER - 1):
                    solver.step()
                    self.push(solver.t, solver.y, solver.f)
            else:
                # For variable step solver we run N times until h, 2h, 3h .. (ORDER - 1)h
                solver = first_stepper_cls(rhs, t0, y0, params)
                if self.ORDER > 1:
                    ts, ys = solver.run(np.arange(1, self.ORDER) * h)
                    for ndx in range(len(ts)):
                        self.push(ts[ndx], ys[0], self.rhs(ts[ndx], ys[ndx]))

    def __init_subclass__(cls, **kwargs):
        if hasattr(cls, "COEFS"):
            cls.COEFS = np.ascontiguousarray(cls.COEFS, dtype=float).flatten()
            cls.ORDER = cls.COEFS.size
            if cls.ORDER == 1:
                cls.COEFS = np.append(cls.COEFS, 0.0)
            cls.COEFS.shape = (1, cls.COEFS.size)
            cls.LEN_HISTORY = cls.COEFS.size

        super().__init_subclass__(**kwargs)

    @staticmethod
    @numba.njit()
    def _interpolate(t_eval, rhs, ts, ys, fs, *extra_args):
        """Interpolate solution at t_eval"""

        if not (ts[0] <= t_eval <= ts[-1]):
            raise ValueError("Time to interpolate outside range")

        y_out = np.empty(ys[0].shape)
        for ndx in range(len(y_out)):
            y_out[ndx] = np.interp(t_eval, ts, ys[:, ndx])

        return y_out

    @property
    def step_size(self):
        return self._h

    def push(self, t_new, y_new, f_new):
        self._ts[:-1] = self._ts[1:]
        self._ts[-1] = t_new

        self._ys[:-1] = self._ys[1:]
        self._ys[-1] = y_new

        self._fs[:-1] = self._fs[1:]
        self._fs[-1] = f_new


class FFixedStepBaseSolver(_FixedStepBaseSolver):
    """Forward fixed step solver.

    Parameters
    ----------
    h : float
        Step size

    See `Solver` for information about the other arguments.
    """

    IMPLICIT = False

    _step_builder = forward_step_builder

    def __init__(
        self,
        rhs: Callable,
        t0: float,
        y0: np.ndarray,
        params: np.ndarray = None,
        *,
        t_bound=np.inf,
        h: float = 1,
        first_stepper_cls=None,
    ):
        super().__init__(
            rhs,
            t0,
            y0,
            params,
            t_bound=t_bound,
            h=h,
            first_stepper_cls=first_stepper_cls,
        )

    def _step_extra_args(self):
        return (self._h,)


class BFixedStepBaseSolver(_FixedStepBaseSolver):
    """Backward fixed step solver.

    Parameters
    ----------
    h : float
        Step size
    rtol : float, optional
        Tolerance (relative) for termination.
    atol : float, optional
        The allowable error of the zero value. If `func` is complex-valued,
        a larger `tol` is recommended as both the real and imaginary parts
        of `x` contribute to ``|x - x0|``.
    max_iter : int, optional
        Maximum number of iterations.

    `rtol`, `atol` and `max_iter` are given to solve the implicit equation.

    See `Solver` for information about the other arguments.
    """

    IMPLICIT = True

    _step_builder = backward_step_builder

    def __init__(
        self,
        rhs: Callable,
        t0: float,
        y0: np.ndarray,
        params: np.ndarray = None,
        *,
        t_bound=np.inf,
        h: float = 1,
        rtol=0.0,
        atol=1.48e-8,
        max_iter=50,
        first_stepper_cls=None,
    ):
        super().__init__(
            rhs,
            t0,
            y0,
            params,
            t_bound=t_bound,
            h=h,
            first_stepper_cls=first_stepper_cls,
        )

        self.atol = atol
        self.rtol = rtol
        self.max_iter = max_iter

    def _step_extra_args(self):
        return self._h, self.rtol, self.atol, self.max_iter
