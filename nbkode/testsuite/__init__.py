"""
    nbkode.testsuite
    ~~~~~~~~~~~~~~~~

    numbakit-ode (nbkode) is a Python package to solve
    **ordinary differential equations (ODE)** that uses
    numba to compile code and therefore speed up calculations.

    :copyright: 2020 by nbkode Authors, see AUTHORS for more details.
    :license: BSD, see LICENSE for more details.
"""


def run():
    """Run all tests."""

    try:
        import pytest
    except ImportError:
        print("pytest not installed. Install it\n    pip install pytest")
        raise

    return pytest.main()
