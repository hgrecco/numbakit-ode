"""
    nbkode.testsuite.test_introspections
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    Capabilities of the package provide information about
    the different solvers.

    :copyright: 2020 by nbkode Authors, see AUTHORS for more details.
    :license: BSD, see LICENSE for more details.
"""

import pytest

import nbkode


def test_get_solvers():
    assert len(nbkode.get_solvers()) == 26


def test_get_solvers_euler():
    assert len(nbkode.get_solvers("Euler")) == 2
    assert len(nbkode.get_solvers("euler")) == 2


def test_get_solvers_implicit():
    for solver in nbkode.get_solvers(implicit=True):
        assert solver.IMPLICIT is True
    for solver in nbkode.get_solvers(implicit=False):
        assert solver.IMPLICIT is False
    assert set(
        nbkode.get_solvers(implicit=False) + nbkode.get_solvers(implicit=True)
    ) == set(nbkode.get_solvers())


def test_get_solvers_fixed_step():
    for solver in nbkode.get_solvers(fixed_step=True):
        assert solver.FIXED_STEP is True
    for solver in nbkode.get_solvers(fixed_step=False):
        assert solver.FIXED_STEP is False
    assert set(
        nbkode.get_solvers(fixed_step=False) + nbkode.get_solvers(fixed_step=True)
    ) == set(nbkode.get_solvers())


def test_get_solver():
    assert nbkode.get_solver("forwardeuler") is nbkode.get_solver("ForwardEuler")

    assert nbkode.get_solver("euler") is nbkode.get_solver("ForwardEuler")

    with pytest.raises(ValueError):
        nbkode.get_solver("not_a_solver")


def test_nbkode_module():
    assert len(dir(nbkode)) == 26 + 4

    with pytest.warns(UserWarning):
        sol = nbkode.__getattr__("euler")
        # For some reason
        # nbkode.euler or getattr(nbkode, "euler")
        # does not warn

    assert sol is nbkode.get_solver("ForwardEuler")

    with pytest.raises(AttributeError):
        sol = nbkode.not_a_solver

    assert getattr(nbkode, "ForwardEuler") is nbkode.get_solver("ForwardEuler")
