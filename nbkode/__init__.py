"""
    nbkode
    ~~~~~~

    numbakit-ode (nbkode) is a Python package to solve
    **ordinary differential equations (ODE)** that uses
    numba to compile code and therefore speed up calculations.

    :copyright: 2020 by nbkode Authors, see AUTHORS for more details.
    :license: BSD, see LICENSE for more details.
"""

# These imports are necessary as only when corresponding module is imported
# each solvers get added to the list.
from . import euler
from .core import get_groups, get_solver, get_solvers, list_solvers
from .multistep import adams_bashforth, adams_moulton, bdf
from .runge_kutta.explicit import DOP853, RungeKutta23, RungeKutta45

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

del version


def test():  # pragma: no cover
    """Run all tests.

    Returns
    -------
    unittest.TestResult
    """
    from .testsuite import run

    return run()


_SOLVERS_NAMES = None


def __dir__():
    global _SOLVERS_NAMES

    if not _SOLVERS_NAMES:
        _SOLVERS_NAMES = ["get_groups", "get_solvers", "get_solver", "test"]
        _SOLVERS_NAMES += list_solvers(include_alias=False)
        _SOLVERS_NAMES = sorted(_SOLVERS_NAMES)

    return _SOLVERS_NAMES


def __getattr__(name):
    try:
        solver = get_solver(name)
    except ValueError:
        raise AttributeError(f"module {__name__} has no attribute {name}")
    if solver.__name__ != name:
        from warnings import warn

        warn(
            f"The name '{name}' is an alias and its use is discouraged. "
            f"Use '{solver.__name__}' instead.",
            stacklevel=2,
        )
    return solver
