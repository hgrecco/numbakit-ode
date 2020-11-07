"""
    nbkode.testsuite.solver_fs
    ~~~~~~~~~~~~~~~~~~~~~~~~~~

    Test fixed step solvers for:
    - argument handling
    - jittability
    - running different methods.

    :copyright: 2020 by nbkode Authors, see AUTHORS for more details.
    :license: BSD, see LICENSE for more details.
"""

import numpy as np
import pytest

import nbkode

solvers_i = nbkode.get_solvers(implicit=True, fixed_step=True)
solvers_e = nbkode.get_solvers(implicit=False, fixed_step=True)
solvers = nbkode.get_solvers(fixed_step=True)


y0_1 = np.atleast_1d(1.0)
y0_2 = np.atleast_1d([1.0, 2.0])


def f1(t, x, k):
    return -k * x


def f2(t, x, k):
    return -k * x


@pytest.mark.parametrize("solver", solvers)
def test_f1_step(solver):
    solver = solver(f1, 0.0, y0_1, params=(0.01,))
    solver.step()
    solver.run(1)


@pytest.mark.parametrize("solver", solvers)
def test_f1(solver):

    solver = solver(f1, 0.0, y0_1, params=(0.01,))
    solver.run(10)

    # TODO: This is a rather large tolerance.
    np.testing.assert_allclose(solver.y, np.exp(-0.01 * 10), atol=0.1)


@pytest.mark.parametrize("solver", solvers)
def test_f2_step(solver):
    solver = solver(f2, 0.0, y0_2, params=(0.01, 0.05))
    solver.step()
    solver.run(1)


@pytest.mark.parametrize("solver", solvers)
def test_f2(solver):

    solver = solver(f2, 0.0, y0_2, params=(0.01, 0.05))
    solver.run(10)

    # TODO: This is a rather large tolerance.
    np.testing.assert_allclose(solver.y[0], np.exp(-0.01 * 10), rtol=0.15)
    np.testing.assert_allclose(solver.y[1], 2.0 * np.exp(-0.05 * 10), rtol=0.15)
