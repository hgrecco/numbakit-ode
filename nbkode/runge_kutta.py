"""
    nbkode.runge_kutta
    ~~~~~~~~~~~~~~~~~~

    Methods of Runge-Kutta family:
    - RungeKutta23
    - RungeKutta45

    :copyright: 2020 by nbkode Authors, see AUTHORS for more details.
    :license: BSD, see LICENSE for more details.
"""

import numpy as np

from . import corevs
from .corevs import MAX_FACTOR, MIN_FACTOR, SAFETY, rk_step
from .nbcompat import clip, numba  # noqa: F401


def step_builder(A, B, C, E, error_exponent):
    """Perform a single fixed step.

    This outer function should only contains attributes
    associated with the solver class not with the solver instance.

    Parameters
    ----------
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
    E :
    error_exponent :

    Returns
    -------
    callable
    """

    @numba.njit
    def _step(t_bound, rhs, t, y, f, h, K, rtol, atol, max_step):
        """Perform a single fixed step.

        This inner function should only contains attributes
        associated with the solver instance not with the solver class.

        Parameters
        ----------
        t_bound : float
            Boundary time - the integration won’t continue beyond it.
            It also determines the direction of the integration.
        rhs : callable
            Right-hand side of the system.
        ts : ndarray, shape (LEN_HISTORY, )
            Last ORDER + 1 times
        ys : ndarray, shape (LEN_HISTORY, n)
            Last ORDER + 1 values, for each n dimension
        fs : ndarray, shape (LEN_HISTORY, n)
            Last ORDER + 1 value of the derivative, for each n dimension
        h : float
            Step to use.
        K : ndarray, shape (n_stages + 1, n)
            Storage array for putting RK stages here. Stages are stored in rows.
            The last row is a linear combination of the previous rows with
            coefficients
        rtol : float
            Tolerance (relative) for termination.
        atol : float
            The allowable error of the zero value. If `func` is complex-valued,
            a larger `tol` is recommended as both the real and imaginary parts
            of `x` contribute to ``|x - x0|``.
        max_step : float
            Maximum allowed step size.

        ts, ts and fs are modified in place.
        h and K is modified in place.

        Returns
        -------
        bool
            True if a step was done, False otherwise.
        """

        t_cur = t[-1]
        y_cur = y[-1]
        f_cur = f[-1]

        min_step = 10 * (np.nextafter(t_cur, np.inf) - t_cur)
        _h = clip(h[0], min_step, max_step)
        step_rejected = False
        while True:
            if _h < min_step:
                raise RuntimeError("Required step is too small.")

            t_new = t_cur + _h
            if t_new > t_bound:
                return False

            y_new, f_new = rk_step(rhs, t_cur, y_cur, f_cur, _h, A, B, C, K)

            # Estimate norm of scaled error
            scale = atol + np.maximum(np.abs(y_cur), np.abs(y_new)) * rtol
            scaled_error = (K.T @ E) * _h / scale
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

        return True

    return _step


class RungeKutta23(corevs.VariableStepRungeKutta):
    """Explicit Runge-Kutta method of order 3(2).

    This uses the Bogacki-Shampine pair of formulas [#]_. The error is controlled
    assuming accuracy of the second-order method, but steps are taken using the
    third-order accurate formula (local extrapolation is done). A cubic Hermite
    polynomial is used for the dense output.
    Can be applied in the complex domain.



    References
    ----------
    .. [#] P. Bogacki, L.F. Shampine, "A 3(2) Pair of Runge-Kutta Formulas",
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

    _step_builder = step_builder

    @classmethod
    def _step_builder_args(cls):
        return cls.A, cls.B, cls.C, cls.E, cls.error_exponent


class RungeKutta45(corevs.VariableStepRungeKutta):
    """Explicit Runge-Kutta method of order 5(4).

    This uses the Dormand-Prince pair of formulas [#]_. The error is controlled
    assuming accuracy of the fourth-order method accuracy, but steps are taken
    using the fifth-order accurate formula (local extrapolation is done).
    A quartic interpolation polynomial is used for the dense output [#]_.
    Can be applied in the complex domain.

    References
    ----------
    .. [#] J. R. Dormand, P. J. Prince, "A family of embedded Runge-Kutta
           formulae", Journal of Computational and Applied Mathematics, Vol. 6,
           No. 1, pp. 19-26, 1980.
    .. [#] L. W. Shampine, "Some Practical Runge-Kutta Formulas", Mathematics
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

    _step_builder = step_builder

    @classmethod
    def _step_builder_args(cls):
        return cls.A, cls.B, cls.C, cls.E, cls.error_exponent
