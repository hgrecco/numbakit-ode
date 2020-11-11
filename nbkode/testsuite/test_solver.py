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
    sol: nbkode.core.Solver = solver(f1, 0.0, y0_1, params=(0.01,))
    ts, ys = sol.step(n=20)

    ########
    # STEP
    ########
    sol: nbkode.core.Solver = solver(f1, 0.0, y0_1, params=(0.01,))
    t, y = sol.step()
    assert np.allclose(t, np.atleast_1d(sol.t))
    assert np.allclose(t, ts[0])
    assert np.allclose(y, np.atleast_1d(sol.y))
    assert isinstance(sol.t, float)
    assert sol.y.shape == y0_1.shape
    assert sol.f.shape == y0_1.shape

    t, y = sol.step(n=2)
    assert np.allclose(t, ts[1:3])
    assert np.allclose(y, ys[1:3])

    t, y = sol.step(upto_t=ts[5])
    assert len(t) == 3
    assert np.allclose(t[-2:], ts[4:6])
    assert np.allclose(y[-2:], ys[4:6])

    t, y = sol.step(n=5, upto_t=ts[7])
    assert len(t) == 2
    assert np.allclose(t[-2:], ts[6:8])
    assert np.allclose(y[-2:], ys[6:8])

    t, y = sol.step(n=1, upto_t=ts[-1])
    assert len(t) == 1
    assert sol.t == ts[8]

    ########
    # SKIP
    ########
    sol.skip()
    assert sol.t == ts[9]

    sol.skip(n=2)
    assert sol.t == ts[11]

    sol.skip(upto_t=ts[13])
    assert sol.t == ts[13]

    sol.skip(n=5, upto_t=ts[15])
    assert sol.t == ts[15]

    sol.skip(n=1, upto_t=ts[-1])
    assert sol.t == ts[16]

    #######
    # t_bound
    #######
    sol: nbkode.core.Solver = solver(f1, 0.0, y0_1, params=(0.01,), t_bound=ts[0])

    with pytest.raises(ValueError):
        sol.step(upto_t=ts[2])

    with pytest.raises(ValueError):
        sol.step(n=3, upto_t=ts[2])

    with pytest.raises(ValueError):
        sol.skip(upto_t=ts[2])

    with pytest.raises(ValueError):
        sol.skip(n=3, upto_t=ts[2])

    with pytest.raises(RuntimeError):
        sol.step(n=2)

    with pytest.raises(RuntimeError):
        sol.skip(n=2)

    sol: nbkode.core.Solver = solver(f1, 0.0, y0_1, params=(0.01,))
    tvec = np.linspace(0, 10, 20)
    t, y = sol.run(tvec)
    assert np.allclose(tvec, t)
    assert y.shape == (len(t),) + y0_1.shape
