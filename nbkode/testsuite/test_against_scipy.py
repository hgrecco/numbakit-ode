"""
    nbkode.testsuite.test_against_scipy
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    Comparisons using SciPy as a gold standard.

    :copyright: 2020 by nbkode Authors, see AUTHORS for more details.
    :license: BSD, see LICENSE for more details.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy import integrate

import nbkode.dop853
from nbkode import runge_kutta

equivalents = [
    (runge_kutta.RungeKutta23, integrate.RK23),
    (runge_kutta.RungeKutta45, integrate.RK45),
    (nbkode.dop853.DOP853, integrate.DOP853),
]


def exponential1(t, x):
    return -0.01 * x


def exponential2(t, x):
    return np.asarray([-0.01, -0.05]) * x


y0_1 = np.atleast_1d(1.0)
y0_2 = np.atleast_1d([1.0, 2.0])


@pytest.mark.parametrize("nbkode_cls, scipy_cls", equivalents)
def test_exponential1(nbkode_cls, scipy_cls):
    nbkode_sol = nbkode_cls(exponential1, 0, y0_1)
    scipy_sol = scipy_cls(exponential1, 0, y0_1, t_bound=30)
    assert_allclose(nbkode_sol.f, scipy_sol.f)
    assert_allclose(nbkode_sol.h, scipy_sol.h_abs)
    ndx = 0
    while True:
        ndx += 1
        nbkode_sol.step()
        scipy_sol.step()
        if scipy_sol.status != "running":
            break
        # We do not compare the last state as Scipy solvers are bound within step
        # and nbkode are not.
        msg = f"Step: {ndx}, Time: {scipy_sol.t}"
        assert_allclose(nbkode_sol.t, scipy_sol.t, err_msg=msg)
        assert_allclose(nbkode_sol.y, scipy_sol.y, err_msg=msg)
        assert_allclose(nbkode_sol.f, scipy_sol.f, err_msg=msg)
        assert_allclose(nbkode_sol.h, scipy_sol.h_abs, err_msg=msg)
        assert_allclose(nbkode_sol.K, scipy_sol.K, err_msg=msg)


@pytest.mark.parametrize("nbkode_cls, scipy_cls", equivalents)
def test_exponential2(nbkode_cls, scipy_cls):
    nbkode_sol = nbkode_cls(exponential2, 0, y0_2)
    scipy_sol = scipy_cls(exponential2, 0, y0_2, t_bound=30)
    assert_allclose(nbkode_sol.f, scipy_sol.f)
    assert_allclose(nbkode_sol.h, scipy_sol.h_abs)
    ndx = 0
    while True:
        ndx += 1
        nbkode_sol.step()
        scipy_sol.step()
        if scipy_sol.status != "running":
            break
        # We do not compare the last state as Scipy solvers are bound within step
        # and nbkode are not.
        msg = f"Step: {ndx}, Time: {scipy_sol.t}"
        assert nbkode_sol.t == scipy_sol.t, msg
        assert_allclose(nbkode_sol.y, scipy_sol.y)
        assert_allclose(nbkode_sol.f, scipy_sol.f)
        assert_allclose(nbkode_sol.h, scipy_sol.h_abs, err_msg=msg)
        assert_allclose(nbkode_sol.K, scipy_sol.K, err_msg=msg)
