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

from .corefs import BFixedStepBaseSolver, FFixedStepBaseSolver


class ForwardEuler(FFixedStepBaseSolver):
    """The simple explicit (forward) Euler scheme

    y[n+1] = y[n] + f(t[n], y[n]) * h
    """

    GROUP = "Euler"

    COEFS = 1.0


Euler = ForwardEuler


class BackwardEuler(BFixedStepBaseSolver):
    """The simple explicit (forward) Euler scheme

    y[n+1] = y[n] + f(t[n+1], y[n+1]) * h
    """

    GROUP = "Euler"
    COEFS = 1.0
