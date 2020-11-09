"""
    nbkode.adams_bashforth
    ~~~~~~~~~~~~~~~~~~~~~~

    Methods of the Adams–Bashforth family (explicit methods).
    - AdamsBashforth1
    - AdamsBashforth2
    - AdamsBashforth3
    - AdamsBashforth4
    - AdamsBashforth5

    See: https://en.wikipedia.org/wiki/Linear_multistep_method

    :copyright: 2020 by nbkode Authors, see AUTHORS for more details.
    :license: BSD, see LICENSE for more details.
"""

import numpy as np

from .corefs import FFixedStepBaseSolver


class _AdamsBashforth(FFixedStepBaseSolver):

    GROUP = "Adams-Bashforth"


class AdamsBashforth1(_AdamsBashforth):
    """The Adams–Bashforth method with ONE is equivalent to Euler"""

    COEFS = 1.0


class AdamsBashforth2(_AdamsBashforth):
    """The Adams–Bashforth method with TWO.

    ::

        u[n+1] = u[n] + h * (3/2 *f(u[n], t[n]) - 1/2 * f(u[n-1], t[n-1]))
    """

    COEFS = np.asarray([3, -1]) / 2


class AdamsBashforth3(_AdamsBashforth):
    """The Adams–Bashforth method with THREE.

    ::

        u[n+1] = u[n] + h /12 .*(23*f(u[n], t[n]) - 16*f(u[n-1], t[n-1])
                                + 5*f(u[n-2], t[n-2]))

    """

    COEFS = np.asarray([23, -16, 5]) / 12


class AdamsBashforth4(_AdamsBashforth):
    """The Adams–Bashforth method with FOUR.

    ::

        u[n+1] = u[n] + h / 24 .* (55.*f(u[n], t[n]) - 59*f(u[n-1], t[n-1]) +
                                   37*f(u[n-2], t[n-2]) - 9*f(u[n-3], t[n-3]))

    """

    COEFS = np.asarray([55, -59, 37, -9]) / 24


class AdamsBashforth5(_AdamsBashforth):
    """The Adams–Bashforth method with FIVE.

    ::

        u[n+1] = u[n] + h / 24 .* (1901.*f(u[n], t[n]) - 2774*f(u[n-1], t[n-1]) +
                                   2616*f(u[n-2], t[n-2]) - 1274*f(u[n-3], t[n-3]) +
                                   251 *f(u[n-4], t[n-4]) )

    """

    COEFS = np.asarray([1901, -2774, 2616, -1274, 251]) / 720
