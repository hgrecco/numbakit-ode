"""
    nbkode.testsuite.solver
    ~~~~~~~~~~~~~~~~~~~~~~~~~~

    Test solvers methods

    :copyright: 2020 by nbkode Authors, see AUTHORS for more details.
    :license: BSD, see LICENSE for more details.
"""

import numpy as np
import pytest

import nbkode

solvers = nbkode.get_solvers()


y0_1 = np.atleast_1d(1.0)
y0_2 = np.atleast_1d([1.0, 2.0])


def f1(t, x, k):
    return -k * x


@pytest.mark.parametrize("solver", solvers)
def test_f1_public_api(solver):
    # This is just running the public api, not checking correctness
    solver: nbkode.core.Solver = solver(f1, 0.0, y0_1, params=(0.01,))
    solver.step()
    solver.nsteps(5)
    assert isinstance(solver.t, float)
    assert solver.y.shape == y0_1.shape
    assert solver.f.shape == y0_1.shape

    t, y = solver.move_to(solver.t + 1)
    assert isinstance(t, float)
    assert y.shape == y0_1.shape

    t, y = solver.run(solver.t + 1)
    assert y.shape == (len(t),) + y0_1.shape

    ts = np.linspace(solver.t, solver.t + 10, 20)
    t, y = solver.run(ts)
    assert np.allclose(ts, t)
    assert y.shape == (len(t),) + y0_1.shape
