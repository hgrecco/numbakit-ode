import numpy as np

from ..nbcompat import numba
from ..order import Solver
from ..util import classproperty
from .core import AdaptiveRungeKutta, RungeKutta


@numba.njit
def jacobian(rhs, t, y, eps=1e-10):
    out_t = (rhs(t + eps, y) - rhs(t - eps, y)) / (2 * eps)
    out_y = np.empty((y.size, y.size))
    for i in range(y.size):
        x1 = y.copy()
        x2 = y.copy()
        x1[i] -= eps
        x2[i] += eps
        out_y[i] = (rhs(t, x2) - rhs(t, x1)) / (2 * eps)
    return out_t, out_y


class LIRK(RungeKutta, abstract=True):
    """Linearly Implicit Runge-Kutta (LIRK) methods.

    K = h f(t + h C, y + A @ K) + dC h^2 df/dt + h df/dy G @ K
    """

    Alpha: np.ndarray
    Gamma: np.ndarray
    B: np.ndarray
    B2: np.ndarray

    @classproperty
    def Alphai(cls):
        """Time coefficients."""
        return np.tril(cls.Alpha, k=-1).sum(1)

    @classproperty
    def Gammai(cls):
        """Time Jacobian coefficients."""
        return np.tril(cls.Gamma).sum(1)

    @classproperty
    def A(cls):
        return cls.Alpha @ np.linalg.inv(cls.Gamma)

    @classproperty
    def M(cls):
        return cls.B @ np.linalg.inv(cls.Gamma)

    @classproperty
    def M2(cls):
        return cls.B2 @ np.linalg.inv(cls.Gamma)

    @classproperty
    def E(cls):
        return (cls.B - cls.B2) @ np.linalg.inv(cls.Gamma)

    @classproperty
    def C(cls):
        return np.diag(1 / np.diag(cls.Gamma)) - np.linalg.inv(cls.Gamma)

    @property
    def _step_args(self):
        return self.rhs, self.cache, self.h, self.K

    @classmethod
    def _build_fixed_step(cls):
        A, C, M = cls.A, cls.C, cls.M
        Gamma = cls.Gamma
        Alphai, Gammai = cls.Alphai, cls.Gammai

        @numba.njit(inline="always")
        def _fixed_step(rhs, cache, h, U):
            t = cache.t
            y = cache.y

            eye = np.eye(y.size)
            ft, J = jacobian(rhs, t, y)

            for i, (alphai, gammai) in enumerate(zip(Alphai, Gammai)):
                du = A[i, :i] @ U[:i]
                duj = C[i, :i] @ U[:i]
                b = rhs(t + h * alphai, y + du) + duj / h + gammai * h * ft
                U[i] = np.linalg.solve(eye / h / Gamma[i, i] - J, b)

            t_new = t + h
            y_new = y + M @ U
            return t_new, y_new

        return _fixed_step


class SLIRK(LIRK, abstract=True):
    gamma: float

    @classmethod
    def _build_fixed_step(cls):
        A, C, M = cls.A, cls.C, cls.M
        gamma = cls.gamma
        Alphai, Gammai = cls.Alphai, cls.Gammai

        @numba.njit(inline="always")
        def _fixed_step(rhs, cache, h, U):
            t = cache.t
            y = cache.y

            ft, J = jacobian(rhs, t, y)
            LU = np.eye(y.size) / h / gamma - J

            for i, (alphai, gammai) in enumerate(zip(Alphai, Gammai)):
                du = A[i, :i] @ U[:i]
                duj = C[i, :i] @ U[:i]
                b = rhs(t + h * alphai, y + du) + duj / h + gammai * h * ft
                U[i] = np.linalg.solve(LU, b)

            t_new = t + h
            y_new = y + M @ U
            return t_new, y_new

        return _fixed_step


class ROS3P(SLIRK, Solver):
    """ROS3P

    Lang, J., Verwer, J. ROS3P—An Accurate Third-Order Rosenbrock Solver Designed for Parabolic Problems.
    BIT Numerical Mathematics 41, 731–738 (2001). https://doi.org/10.1023/A:1021900219772
    """

    gamma = (3 + np.sqrt(3)) / 6
    Alpha = np.array([[0, 0, 0], [1, 0, 0], [1, 0, 0]])
    Gamma = np.array(
        [[gamma, 0, 0], [-1, gamma, 0], [-gamma, 1 / 2 - 2 * gamma, gamma]]
    )
    B = np.array([2, 0, 1]) / 3


class AdaptiveROS3P(AdaptiveRungeKutta, ROS3P):
    B2 = np.array([1, 1, 1]) / 3
    error_estimator_order = 3


if __name__ == "__main__":
    rhs = numba.njit(lambda t, y: -y)
    t, y = 0, np.array([1.0])
    # s = ROS3P(rhs, t, y, h=0.1)
    s = AdaptiveROS3P(rhs, t, y, h=0.1)
    s.step()
