import numpy as np

from ..util import classproperty
from .core import ExplicitMultistep


class AdamsBashforth(ExplicitMultistep, abstract=True):
    GROUP = "Adams-Bashforth"

    @classproperty
    def A(cls):
        A = np.zeros_like(cls.B)
        A[-1] = -1
        return A


class AdamsBashforth1(AdamsBashforth):
    """The Adams–Bashforth method with ONE is equivalent to Euler"""

    B = np.array([1.0])


class AdamsBashforth2(AdamsBashforth):
    """The Adams–Bashforth method with TWO.

    ::

        u[n+1] = u[n] + h * (3/2 *f(u[n], t[n]) - 1/2 * f(u[n-1], t[n-1]))
    """

    B = np.array([3, -1]) / 2


class AdamsBashforth3(AdamsBashforth):
    """The Adams–Bashforth method with THREE.

    ::

        u[n+1] = u[n] + h /12 .*(23*f(u[n], t[n]) - 16*f(u[n-1], t[n-1])
                                + 5*f(u[n-2], t[n-2]))

    """

    B = np.array([23, -16, 5]) / 12


class AdamsBashforth4(AdamsBashforth):
    """The Adams–Bashforth method with FOUR.

    ::

        u[n+1] = u[n] + h / 24 .* (55.*f(u[n], t[n]) - 59*f(u[n-1], t[n-1]) +
                                   37*f(u[n-2], t[n-2]) - 9*f(u[n-3], t[n-3]))

    """

    B = np.array([55, -59, 37, -9]) / 24


class AdamsBashforth5(AdamsBashforth):
    B = np.array([1901, -2774, 2616, -1274, 251]) / 720
