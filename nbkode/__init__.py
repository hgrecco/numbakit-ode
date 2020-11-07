"""
    nbkode
    ~~~~~~

    numbakit-ode (nbkode) is a Python package to solve
    **ordinary differential equations (ODE)** that uses
    numba to compile code and therefore speed up calculations.

    :copyright: 2020 by nbkode Authors, see AUTHORS for more details.
    :license: BSD, see LICENSE for more details.
"""

from .adams_bashforth import (
    AdamsBashforth1,
    AdamsBashforth2,
    AdamsBashforth3,
    AdamsBashforth4,
    AdamsBashforth5,
)
from .adams_moulton import (
    AdamsMoulton1,
    AdamsMoulton2,
    AdamsMoulton3,
    AdamsMoulton4,
    AdamsMoulton5,
)
from .core import get_groups, get_solvers
from .dop853 import DOP853
from .euler import BackwardEuler, Euler, ForwardEuler
from .runge_kutta import RungeKutta23, RungeKutta45

try:
    from importlib.metadata import version
except ImportError:
    # Backport for Python < 3.8
    from importlib_metadata import version

try:  # pragma: no cover
    __version__ = version("numbakit-ode")
except Exception:  # pragma: no cover
    # we seem to have a local copy not installed without setuptools
    # so the reported version will be unknown
    __version__ = "unknown"


def test():  # pragma: no cover
    """Run all tests.

    Returns
    -------
    unittest.TestResult
    """
    from .testsuite import run

    return run()
