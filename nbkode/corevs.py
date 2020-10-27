"""
    nbkode.corevs
    ~~~~~~~~~~~~~

    Definitions for Variable Step Methods.

    See: https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta_methods

    :copyright: 2020 by nbkode Authors, see AUTHORS for more details.
    :license: BSD, see LICENSE for more details.
"""


from functools import partial
from typing import Callable, Tuple

import numpy as np
from scipy.integrate._ivp.common import (
    select_initial_step,
    validate_max_step,
    validate_tol,
)

from nbkode.nbcompat import clip, numba

from .core import Solver

SAFETY = 0.9  # Multiply steps computed from asymptotic behaviour of errors by this.
MIN_FACTOR = 0.2  # Minimum allowed decrease in a step size.
MAX_FACTOR = 10  # Maximum allowed increase in a step size.


@numba.njit(cache=True)
def rk_step(rhs, t, y, f, h, A, B, C, K):
    """Perform a single Runge-Kutta step.
    This function computes a prediction of an explicit Runge-Kutta method and
    also estimates the error of a less accurate method.
    Notation for Butcher tableau is as in [1]_.

    Parameters
    ----------
    fun : callable
        Right-hand side of the system.
    t : float
        Current time.
    y : ndarray, shape (n,)
        Current state.
    f : ndarray, shape (n,)
        Current value of the derivative, i.e., ``fun(x, y)``.
    h : float
        Step to use.
    A : ndarray, shape (n_stages, n_stages)
        Coefficients for combining previous RK stages to compute the next
        stage. For explicit methods the coefficients at and above the main
        diagonal are zeros.
    B : ndarray, shape (n_stages,)
        Coefficients for combining RK stages for computing the final
        prediction.
    C : ndarray, shape (n_stages,)
        Coefficients for incrementing time for consecutive RK stages.
        The value for the first stage is always zero.
    K : ndarray, shape (n_stages + 1, n)
        Storage array for putting RK stages here. Stages are stored in rows.
        The last row is a linear combination of the previous rows with
        coefficients

    Returns
    -------
    y_new : ndarray, shape (n,)
        Solution at t + h computed with a higher accuracy.
    f_new : ndarray, shape (n,)
        Derivative ``fun(t + h, y_new)``.

    References
    ----------
    .. [1] E. Hairer, S. P. Norsett G. Wanner, "Solving Ordinary Differential
           Equations I: Nonstiff Problems", Sec. II.4.
    """
    K[0] = f
    for s in range(1, len(C)):
        a, c = A[s], C[s]
        dy = (K[:s].T @ a[:s]) * h
        K[s] = rhs(t + c * h, y + dy)

    y_new = y + h * (K[:-1].T @ B)
    f_new = rhs(t + h, y_new)
    K[-1] = f_new

    return y_new, f_new


@numba.njit(cache=True)
def _step(t_bound, rhs, t, y, f, h, K, closure_args):
    A, B, C, E, error_exponent, atol, rtol, max_step = closure_args

    t_cur = t[-1]
    y_cur = y[-1]
    f_cur = f[-1]

    min_step = 10 * (np.nextafter(t_cur, np.inf) - t_cur)
    _h = clip(h[0], min_step, max_step)
    step_rejected = False
    while True:
        if _h < min_step:
            raise RuntimeError("Required step is too small.")

        t_new = min(t_cur + _h, t_bound)
        _h = t_new - t_cur

        y_new, f_new = rk_step(rhs, t_cur, y_cur, f_cur, _h, A, B, C, K)

        # Estimate norm of scaled error
        scale = atol + np.maximum(np.abs(y_cur), np.abs(y_new)) * rtol

        # _estimate_error
        scaled_error = (K.T @ E) * _h / scale

        # _estimate_error_norm
        error_norm = (np.sum(scaled_error ** 2) / scaled_error.size) ** 0.5

        if error_norm < 1:
            if error_norm == 0:
                factor = MAX_FACTOR
            else:
                factor = min(MAX_FACTOR, SAFETY * error_norm ** error_exponent)

            if step_rejected:
                factor = min(1, factor)

            _h *= factor

            break
        else:
            _h *= max(MIN_FACTOR, SAFETY * error_norm ** error_exponent)
            step_rejected = True

    f[:-1] = f[1:]
    f[-1] = f_new

    t[:-1] = t[1:]
    t[-1] = t_new

    y[:-1] = y[1:]
    y[-1] = y_new

    h[0] = _h


class VariableStepRungeKutta(Solver):
    """Base class for explicit Runge-Kutta methods.

    Parameters
    ----------
    max_step : float, optional
        Maximum allowed step size. Default is np.inf, i.e., the step size is not
        bounded and determined solely by the solver.
    rtol, atol : float and array_like, optional
        Relative and absolute tolerances. The solver keeps the local error
        estimates less than ``atol + rtol * abs(y)``. Here, `rtol` controls a
        relative accuracy (number of correct digits). But if a component of `y`
        is approximately below `atol`, the error only needs to fall within
        the same `atol` threshold, and the number of correct digits is not
        guaranteed. If components of y have different scales, it might be
        beneficial to set different `atol` values for different components by
        passing array_like with shape (n,) for `atol`. Default values are
        1e-3 for `rtol` and 1e-6 for `atol`.

    See `Solver` for information about the other arguments.
    """

    FIXED_STEP = False

    C: np.ndarray
    A: np.ndarray
    B: np.ndarray
    E: np.ndarray
    P: np.ndarray
    order: int
    error_estimator_order: int
    n_stages: int

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.error_exponent = -1 / (cls.error_estimator_order + 1)

    def __init__(
        self,
        rhs: Callable,
        t0: float,
        y0: np.ndarray,
        args: tuple = (),
        *,
        max_step=np.inf,
        rtol=1e-3,
        atol=1e-6,
    ):
        super().__init__(rhs, t0, y0, args)

        self.max_step = validate_max_step(max_step)
        # self._step = step_builder(
        #     self.A, self.B, self.C, self.E, self.error_exponent, atol, rtol, max_step
        # )
        self._step = _step
        self.rtol, self.atol = validate_tol(rtol, atol, self.y.size)

        h = select_initial_step(
            self.jrhs,
            self.t,
            self.y,
            self.f,
            1,
            self.error_estimator_order,
            self.rtol,
            self.atol,
        )

        # This is required to allow editing in place.
        self.h = np.atleast_1d(h)
        self.K = np.empty((self.n_stages + 1, self.y.size), dtype=float)

    def _steps_extra_args(self):
        return self.h, self.K, (
            self.A, self.B, self.C, self.E, self.error_exponent, self.atol, self.rtol, self.max_step
        )

    @staticmethod
    def _step(t_bound, rhs, t, y, f):
        raise RuntimeError("This should have been replaced during init.")
