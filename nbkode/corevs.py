"""
    nbkode.corevs
    ~~~~~~~~~~~~~

    Definitions for Variable Step Methods.

    See: https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta_methods

    :copyright: 2020 by nbkode Authors, see AUTHORS for more details.
    :license: BSD, see LICENSE for more details.
"""


from typing import Callable

import numpy as np
from scipy.integrate._ivp.common import (
    select_initial_step,
    validate_max_step,
    validate_tol,
)

from nbkode.nbcompat import numba

from .core import Solver

SAFETY = 0.9  # Multiply steps computed from asymptotic behaviour of errors by this.
MIN_FACTOR = 0.2  # Minimum allowed decrease in a step size.
MAX_FACTOR = 10  # Maximum allowed increase in a step size.


@numba.njit()
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
    error_exponent: float

    def __init_subclass__(cls, **kwargs):
        cls.error_exponent = -1 / (cls.error_estimator_order + 1)
        super().__init_subclass__(**kwargs)

    def __init__(
        self,
        rhs: Callable,
        t0: float,
        y0: np.ndarray,
        params: np.ndarray = None,
        *,
        t_bound=np.inf,
        rtol=1e-3,
        atol=1e-6,
        max_step=np.inf,
    ):
        super().__init__(rhs, t0, y0, params, t_bound=t_bound)

        self.max_step = validate_max_step(max_step)
        self.rtol, self.atol = validate_tol(rtol, atol, self.y.size)

        h = select_initial_step(
            self.rhs,
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

    def _step_extra_args(self):
        return self.h, self.K, self.rtol, self.atol, self.max_step
