"""
    nbkode.adams_moulton
    ~~~~~~~~~~~~~~~~~~~~

    Methods of the Adams–Moulton family (implicit methods).
    - AdamsMoulton1
    - AdamsMoulton2
    - AdamsMoulton3
    - AdamsMoulton4
    - AdamsMoulton5

    See: https://en.wikipedia.org/wiki/Linear_multistep_method

    :copyright: 2020 by nbkode Authors, see AUTHORS for more details.
    :license: BSD, see LICENSE for more details.
"""


import numpy as np

from .corefs import BFixedStepBaseSolver


class _AdamsMoulton(BFixedStepBaseSolver):

    GROUP = "Adams-Moulton"


class AdamsMoulton1(_AdamsMoulton):
    """The Adams–Bashforth method with ONE is equivalent to Euler"""

    COEFS = 1.0


class AdamsMoulton2(_AdamsMoulton):
    """The Adams–Bashforth method with TWO.

    ::

        u[n+1] = u[n] + h * (3/2 *f(u[n], t[n]) - 1/2 * f(u[n-1], t[n-1]))
    """

    COEFS = np.asarray([1, 1]) / 2


class AdamsMoulton3(_AdamsMoulton):
    """The Adams–Bashforth method with THREE.

    ::

        u[n+1] = u[n] + h /12 .*(23*f(u[n], t[n]) - 16*f(u[n-1], t[n-1])
                                + 5*f(u[n-2], t[n-2]))

    """

    COEFS = np.asarray([5, 8, -1]) / 12


class AdamsMoulton4(_AdamsMoulton):
    """The Adams–Bashforth method with FOUR.

    ::

        u[n+1] = u[n] + h / 24 .* (55.*f(u[n], t[n]) - 59*f(u[n-1], t[n-1]) +
                                   37*f(u[n-2], t[n-2]) - 9*f(u[n-3], t[n-3]))

    """

    COEFS = np.asarray([9, 19, -5, 1]) / 24


class AdamsMoulton5(_AdamsMoulton):
    """The Adams–Bashforth method with FIVE.

    ::

        u[n+1] = u[n] + h / 24 .* (1901.*f(u[n], t[n]) - 2774*f(u[n-1], t[n-1]) +
                                   2616*f(u[n-2], t[n-2]) - 1274*f(u[n-3], t[n-3]) +
                                   251 *f(u[n-4], t[n-4]) )

    """

    COEFS = np.asarray([251, 646, -264, 106, -19]) / 720
