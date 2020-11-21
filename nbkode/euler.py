"""
    nbkode.euler
    ~~~~~~~~~~~~

    Methods of the Adams–Moulton family (implicit methods).
    - ForwardEuler (Euler)
    - BackwardEuler

    See: https://en.wikipedia.org/wiki/Linear_multistep_method

    :copyright: 2020 by nbkode Authors, see AUTHORS for more details.
    :license: BSD, see LICENSE for more details.
"""

from .multistep.adams_bashforth import AdamsBashforth1
from .multistep.adams_moulton import AdamsMoulton1


class ForwardEuler(AdamsBashforth1):
    """The simple explicit (forward) Euler scheme

    y[n+1] = y[n] + h * f(t[n], y[n])
    """

    ALIASES = ("Euler",)

    GROUP = "Euler"


class BackwardEuler(AdamsMoulton1):
    """The simple implicit (backward) Euler scheme

    y[n+1] = y[n] + h * f(t[n+1], y[n+1])
    """

    GROUP = "Euler"
