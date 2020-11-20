import numpy as np

from ..nbcompat import numba
from ..util import classproperty
from . import dop853_coefficients
from .core import AdaptiveRungeKutta, RungeKutta


class ERK(RungeKutta, abstract=True):
    """Explicit Runge-Kutta (ERK) method."""

    IMPLICIT = False

    def __init_subclass__(cls, *args, abstract=False, **kwargs) -> None:
        if not abstract:
            assert cls.explicit
        super().__init_subclass__(*args, abstract=abstract, **kwargs)

    @classproperty
    def explicit(cls):
        """The method is explicit if the upper triangle of the A matrix is 0."""
        return np.all(np.triu(cls.A) == 0)

    @property
    def _step_args(self):
        return self.rhs, self.cache, self.h, self.K

    @classmethod
    def _fixed_step_builder(cls):
        A, B, C = cls.A, cls.B, cls.C

        @numba.njit(inline="always")
        def _fixed_step(rhs, cache, h, K):
            t = cache.t
            y = cache.y

            K[0] = cache.f
            for s, c in enumerate(C[1:], 1):
                dy = A[s, :s] @ K[:s]
                K[s] = rhs(t + h * c, y + h * dy)

            t = t + h
            y = y + h * B @ K
            return t, y

        return _fixed_step


class FSAL(AdaptiveRungeKutta, ERK, abstract=True):
    @classproperty
    def stages(cls):
        return len(cls.A) + 1

    @classproperty
    def E(cls):
        E = -cls.B2
        E[:-1] += cls.B
        return E

    @classmethod
    def _fixed_step_builder(cls):
        A, B, C = cls.A, cls.B, cls.C

        @numba.njit(inline="always")
        def _fixed_step(rhs, cache, h, K):
            t = cache.t
            y = cache.y

            K[0] = cache.f
            for s, c in enumerate(C[1:], 1):
                dy = A[s, :s] @ K[:s]
                K[s] = rhs(t + h * c, y + h * dy)

            t = t + h
            y = y + h * B @ K[:-1]
            K[-1] = rhs(t, y)
            return t, y

        return _fixed_step

    @classmethod
    def _step_builder(cls):
        E, error_exponent = cls.E, cls.error_exponent
        fixed_step = cls._fixed_step
        step_error = cls._step_error
        scaled_error_norm = cls._scaled_error_norm
        step_update = cls._step_update

        @numba.njit
        def _step(t_bound, rhs, cache, h, K, options):
            if cache.t + h > t_bound:
                return False

            while True:
                t, y = fixed_step(rhs, cache, h, K)
                error = step_error(h, K, E)
                error_norm = scaled_error_norm(y, cache.y, error, options)
                if step_update(error_norm, h, options, error_exponent):
                    cache.t = t
                    cache.y = y
                    cache.f = K[-1]
                    break

            return True

        return _step


class Runge2(ERK):
    A = np.array([[0, 0], [1 / 2, 0]])
    B = np.array([0, 1.0])
    C = np.array([0, 1 / 2])


class Runge3(ERK):
    A = np.array([[0, 0, 0, 0], [1 / 2, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])
    B = np.array([1, 4, 0, 1]) / 6
    C = np.array([0, 1 / 2, 1, 1])


class Heun3(ERK):
    A = np.array([[0, 0, 0], [1, 0, 0], [0, 2, 0]]) / 3
    B = np.array([1, 0, 3]) / 4
    C = np.array([0, 1, 2]) / 3


class RungeKutta4(ERK):
    A = np.array([[0, 0, 0, 0], [1 / 2, 0, 0, 0], [0, 1 / 2, 0, 0], [0, 0, 1, 0]])
    B = np.array([1, 2, 2, 1]) / 6
    C = np.array([0, 1, 1, 2]) / 2


class RungeKutta3_8(ERK):
    A = np.array([[0, 0, 0, 0], [1 / 3, 0, 0, 0], [-1 / 3, 1, 0, 0], [1, -1, 1, 0]])
    B = np.array([1, 3, 3, 1]) / 8
    C = np.array([0, 1, 2, 3]) / 3


class RungeKutta23(FSAL):
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

    C = np.array([0, 1 / 2, 3 / 4], dtype=float)
    A = np.array([[0, 0, 0], [1 / 2, 0, 0], [0, 3 / 4, 0]], dtype=float)
    B = np.array([2 / 9, 1 / 3, 4 / 9], dtype=float)
    B2 = np.array([7 / 24, 1 / 4, 1 / 3, 1 / 8], dtype=float)
    error_estimator_order = 2


class RungeKutta45(FSAL):
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

    A = np.array(
        [
            [0, 0, 0, 0, 0, 0],
            [1 / 5, 0, 0, 0, 0, 0],
            [3 / 40, 9 / 40, 0, 0, 0, 0],
            [44 / 45, -56 / 15, 32 / 9, 0, 0, 0],
            [19372 / 6561, -25360 / 2187, 64448 / 6561, -212 / 729, 0, 0],
            [9017 / 3168, -355 / 33, 46732 / 5247, 49 / 176, -5103 / 18656, 0],
        ]
    )
    B = np.array([35 / 384, 0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84])
    C = np.array([0, 1 / 5, 3 / 10, 4 / 5, 8 / 9, 1])

    # Corresponds to the optimum value of c_6 from [2]_.
    P = np.array(
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
        ]
    )

    error_estimator_order = 4
    E = np.array(
        [-71 / 57600, 0, 71 / 16695, -71 / 1920, 17253 / 339200, -22 / 525, 1 / 40]
    )


class Fehlberg45(AdaptiveRungeKutta, ERK, abstract=True):
    """Explicit Runge-Kutta method of order 5 by Fehlberg (1969).

    Error is controlled by embedded formula of order 4 (see AdaptiveFehlberg45).

    Table 5.1 of Hairer.

    Check
    """

    A = np.array(
        [
            [0, 0, 0, 0, 0, 0],
            [1 / 4, 0, 0, 0, 0, 0],
            [3 / 32, 9 / 32, 0, 0, 0, 0],
            [1932 / 2197, -7200 / 2197, 7296 / 2197, 0, 0, 0],
            [439 / 216, -8, 3680 / 513, -845 / 4104, 0, 0],
            [-8 / 27, 2, -3544 / 2565, 1859 / 4104, -11 / 40, 0],
        ]
    )
    B = np.array([25 / 216, 0, 1408 / 2565, 2197 / 4104, -1 / 5, 0])
    C = np.array([0, 1 / 4, 3 / 8, 12 / 13, 1, 1 / 2])
    B2 = np.array([16 / 135, 0, 6656 / 12825, 28561 / 56430, -9 / 50, 2 / 55])
    error_estimator_order = 4


class DOP853(FSAL):
    """Explicit Runge-Kutta method of order 8.

    This is a Python implementation of "DOP853" algorithm originally written
    in Fortran [#]_, [#]_. Note that this is not a literate translation, but
    the algorithmic core and coefficients are the same.
    Can be applied in the complex domain.


    References
    ----------
    .. [#] E. Hairer, S. P. Norsett G. Wanner, "Solving Ordinary Differential
           Equations I: Nonstiff Problems", Sec. II.
    .. [#] `Page with original Fortran code of DOP853
            <http://www.unige.ch/~hairer/software.html>`_.
    """

    order = 8
    error_estimator_order = 7
    A = dop853_coefficients.A[:12, :12]
    B = dop853_coefficients.B
    C = dop853_coefficients.C[:12]
    E3 = dop853_coefficients.E3
    E5 = dop853_coefficients.E5

    @classmethod
    def _step_builder(cls):
        E3, E5, error_exponent = cls.E3, cls.E5, cls.error_exponent
        fixed_step = cls._fixed_step
        step_error = cls._step_error
        scaled_error_norm = cls._scaled_error_norm
        step_update = cls._step_update

        @numba.njit
        def _step(t_bound, rhs, cache, h, K, options):
            if cache.t + h > t_bound:
                return False

            while True:
                t, y = fixed_step(rhs, cache, h, K)
                err5 = step_error(h, K, E5)
                err3 = step_error(h, K, E3)
                err5 = scaled_error_norm(y, cache.y, err5, options)
                err3 = scaled_error_norm(y, cache.y, err3, options)
                error_norm = err5 / np.sqrt(1 + ((err3 / (10 * err5)) ** 2))
                if step_update(error_norm, h, options, error_exponent):
                    cache.t = t
                    cache.y = y
                    cache.f = K[-1]
                    break

            return True

        return _step
