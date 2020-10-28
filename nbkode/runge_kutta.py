"""
    nbkode.runge_kutta
    ~~~~~~~~~~~~~~~~~~

    Methods of Runge-Kutta family:
    - RungeKutta23
    - RungeKutta45

    :copyright: 2020 by nbkode Authors, see AUTHORS for more details.
    :license: BSD, see LICENSE for more details.
"""

from typing import Callable

import numpy as np

from .nbcompat import numba
from . import dop853_coefficients
from .corevs import VariableStepRungeKutta


class RungeKutta23(VariableStepRungeKutta):
    """Explicit Runge-Kutta method of order 3(2).

    This uses the Bogacki-Shampine pair of formulas [1]_. The error is controlled
    assuming accuracy of the second-order method, but steps are taken using the
    third-order accurate formula (local extrapolation is done). A cubic Hermite
    polynomial is used for the dense output.
    Can be applied in the complex domain.



    References
    ----------
    .. [1] P. Bogacki, L.F. Shampine, "A 3(2) Pair of Runge-Kutta Formulas",
           Appl. Math. Lett. Vol. 2, No. 4. pp. 321-325, 1989.
    """

    GROUP = "Runge-Kutta"
    IMPLICIT = False

    order = 3
    error_estimator_order = 2
    n_stages = 3
    C = np.ascontiguousarray([0, 1 / 2, 3 / 4], dtype=float)
    A = np.ascontiguousarray([[0, 0, 0], [1 / 2, 0, 0], [0, 3 / 4, 0]], dtype=float)
    B = np.ascontiguousarray([2 / 9, 1 / 3, 4 / 9], dtype=float)
    E = np.ascontiguousarray([5 / 72, -1 / 12, -1 / 9, 1 / 8], dtype=float)
    P = np.ascontiguousarray(
        [[1, -4 / 3, 5 / 9], [0, 1, -2 / 3], [0, 4 / 3, -8 / 9], [0, -1, 1]],
        dtype=float,
    )


class RungeKutta45(VariableStepRungeKutta):
    """Explicit Runge-Kutta method of order 5(4).

    This uses the Dormand-Prince pair of formulas [1]_. The error is controlled
    assuming accuracy of the fourth-order method accuracy, but steps are taken
    using the fifth-order accurate formula (local extrapolation is done).
    A quartic interpolation polynomial is used for the dense output [2]_.
    Can be applied in the complex domain.

    References
    ----------
    .. [1] J. R. Dormand, P. J. Prince, "A family of embedded Runge-Kutta
           formulae", Journal of Computational and Applied Mathematics, Vol. 6,
           No. 1, pp. 19-26, 1980.
    .. [2] L. W. Shampine, "Some Practical Runge-Kutta Formulas", Mathematics
           of Computation,, Vol. 46, No. 173, pp. 135-150, 1986.
    """

    GROUP = "Runge-Kutta"
    IMPLICIT = False

    order = 5
    error_estimator_order = 4
    n_stages = 6
    C = np.ascontiguousarray([0, 1 / 5, 3 / 10, 4 / 5, 8 / 9, 1], dtype=float)
    A = np.ascontiguousarray(
        [
            [0, 0, 0, 0, 0],
            [1 / 5, 0, 0, 0, 0],
            [3 / 40, 9 / 40, 0, 0, 0],
            [44 / 45, -56 / 15, 32 / 9, 0, 0],
            [19372 / 6561, -25360 / 2187, 64448 / 6561, -212 / 729, 0],
            [9017 / 3168, -355 / 33, 46732 / 5247, 49 / 176, -5103 / 18656],
        ],
        dtype=float,
    )
    B = np.ascontiguousarray(
        [35 / 384, 0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84], dtype=float
    )
    E = np.ascontiguousarray(
        [-71 / 57600, 0, 71 / 16695, -71 / 1920, 17253 / 339200, -22 / 525, 1 / 40],
        dtype=float,
    )
    # Corresponds to the optimum value of c_6 from [2]_.
    P = np.ascontiguousarray(
        [
            [
                1,
                -8048581381 / 2820520608,
                8663915743 / 2820520608,
                -12715105075 / 11282082432,
            ],
            [0, 0, 0, 0],
            [
                0,
                131558114200 / 32700410799,
                -68118460800 / 10900136933,
                87487479700 / 32700410799,
            ],
            [
                0,
                -1754552775 / 470086768,
                14199869525 / 1410260304,
                -10690763975 / 1880347072,
            ],
            [
                0,
                127303824393 / 49829197408,
                -318862633887 / 49829197408,
                701980252875 / 199316789632,
            ],
            [
                0,
                -282668133 / 205662961,
                2019193451 / 616988883,
                -1453857185 / 822651844,
            ],
            [0, 40617522 / 29380423, -110615467 / 29380423, 69997945 / 29380423],
        ],
        dtype=float,
    )


class DOP853(VariableStepRungeKutta):
    """Explicit Runge-Kutta method of order 8.
    This is a Python implementation of "DOP853" algorithm originally written
    in Fortran [1]_, [2]_. Note that this is not a literate translation, but
    the algorithmic core and coefficients are the same.
    Can be applied in the complex domain.
    Parameters
    ----------
    fun : callable
        Right-hand side of the system. The calling signature is ``fun(t, y)``.
        Here, ``t`` is a scalar, and there are two options for the ndarray ``y``:
        It can either have shape (n,); then ``fun`` must return array_like with
        shape (n,). Alternatively it can have shape (n, k); then ``fun``
        must return an array_like with shape (n, k), i.e. each column
        corresponds to a single column in ``y``. The choice between the two
        options is determined by `vectorized` argument (see below).
    t0 : float
        Initial time.
    y0 : array_like, shape (n,)
        Initial state.
    t_bound : float
        Boundary time - the integration won't continue beyond it. It also
        determines the direction of the integration.
    first_step : float or None, optional
        Initial step size. Default is ``None`` which means that the algorithm
        should choose.
    max_step : float, optional
        Maximum allowed step size. Default is np.inf, i.e. the step size is not
        bounded and determined solely by the solver.
    rtol, atol : float and array_like, optional
        Relative and absolute tolerances. The solver keeps the local error
        estimates less than ``atol + rtol * abs(y)``. Here `rtol` controls a
        relative accuracy (number of correct digits). But if a component of `y`
        is approximately below `atol`, the error only needs to fall within
        the same `atol` threshold, and the number of correct digits is not
        guaranteed. If components of y have different scales, it might be
        beneficial to set different `atol` values for different components by
        passing array_like with shape (n,) for `atol`. Default values are
        1e-3 for `rtol` and 1e-6 for `atol`.
    vectorized : bool, optional
        Whether `fun` is implemented in a vectorized fashion. Default is False.
    Attributes
    ----------
    n : int
        Number of equations.
    status : string
        Current status of the solver: 'running', 'finished' or 'failed'.
    t_bound : float
        Boundary time.
    direction : float
        Integration direction: +1 or -1.
    t : float
        Current time.
    y : ndarray
        Current state.
    t_old : float
        Previous time. None if no steps were made yet.
    step_size : float
        Size of the last successful step. None if no steps were made yet.
    nfev : int
        Number evaluations of the system's right-hand side.
    njev : int
        Number of evaluations of the Jacobian. Is always 0 for this solver
        as it does not use the Jacobian.
    nlu : int
        Number of LU decompositions. Is always 0 for this solver.
    References
    ----------
    .. [1] E. Hairer, S. P. Norsett G. Wanner, "Solving Ordinary Differential
           Equations I: Nonstiff Problems", Sec. II.
    .. [2] `Page with original Fortran code of DOP853
            <http://www.unige.ch/~hairer/software.html>`_.
    """

    GROUP = "Runge-Kutta"
    IMPLICIT = False

    n_stages = dop853_coefficients.N_STAGES
    order = 8
    error_estimator_order = 7
    A = dop853_coefficients.A[:n_stages, :n_stages]
    B = dop853_coefficients.B
    C = dop853_coefficients.C[:n_stages]
    E3 = dop853_coefficients.E3
    E5 = dop853_coefficients.E5
    D = dop853_coefficients.D

    A_EXTRA = dop853_coefficients.A[n_stages + 1:]
    C_EXTRA = dop853_coefficients.C[n_stages + 1:]

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
        super().__init__(rhs, t0, y0, args, max_step=max_step, rtol=rtol, atol=atol)

        self.K_extended = np.empty((dop853_coefficients.N_STAGES_EXTENDED,
                                    y0.size), dtype=y0.dtype)
        self.K = self.K_extended[:self.n_stages + 1]

    @staticmethod
    @numba.njit
    def _estimate_error_norm(K, h, scale, een_extra_args):
        E5, E3 = een_extra_args
        err5 = np.dot(K.T, E5) / scale
        err3 = np.dot(K.T, E3) / scale
        err5_norm_2 = np.linalg.norm(err5)**2
        err3_norm_2 = np.linalg.norm(err3)**2
        if err5_norm_2 == 0 and err3_norm_2 == 0:
            return 0.0
        denom = err5_norm_2 + 0.01 * err3_norm_2
        return np.abs(h) * err5_norm_2 / np.sqrt(denom * len(scale))

    def _estimate_error_norm_args(self):
        return self.E5, self.E3