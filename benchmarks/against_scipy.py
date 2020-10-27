"""
    benchmarks.against_scipy
    ~~~~~~~~~~~~~~~~~~~~~~~~

    Comparisons using SciPy as a gold standard.

    :copyright: 2020 by nbkode Authors, see AUTHORS for more details.
    :license: BSD, see LICENSE for more details.
"""

import numpy as np
from scipy import integrate

from nbkode import runge_kutta
from nbkode.nbcompat import numba

from .common import NumbaStepModes

by_name = {
    "scipy": {"RungeKutta23": integrate.RK23, "RungeKutta45": integrate.RK45,},
    "nbkode": {
        "RungeKutta23": runge_kutta.RungeKutta23,
        "RungeKutta45": runge_kutta.RungeKutta45,
    },
}

y0 = np.atleast_1d(1.0)


def create_f(package):
    if package == "nbkode":

        def f(t, y, k):
            return k * y

        args = 0.0, y0, -0.01
    else:
        k = -0.01

        def f(t, y):
            return k * y

        args = 0.0, y0

    return f, args


PACKAGES = tuple(by_name.keys())
INTEGRATORS = tuple(by_name["scipy"].keys())
BOOLEANS = (True, False)
NUMBA_MODES = tuple(NumbaStepModes.__members__.keys())


func = None


def define_func(package, jit_rhs_before):
    global func

    if jit_rhs_before:
        func, args = create_f(package)
        func = numba.njit()(func)
    else:
        func, args = create_f(package)

    # Test (and compile) func
    func(*args)


sol = None


def define_sol(package, integrator):
    global sol, func
    solver_cls = by_name[package][integrator]

    if package == "nbkode":
        sol = solver_cls(func, 0.0, y0, args=(-0.01,))
    else:
        sol = solver_cls(func, 0.0, y0, t_bound=10_000_000_000)


###############
# Instantiate
###############


def setup_time_f1_instantiate(package, integrator, jit_rhs_before):
    define_func(package, jit_rhs_before)


def time_f1_instantiate(package, integrator, jit_rhs_before):
    """Measures the time required to instantiate the solver
    """
    define_sol(package, integrator)


time_f1_instantiate.setup = setup_time_f1_instantiate
time_f1_instantiate.params = (PACKAGES, INTEGRATORS, BOOLEANS)
time_f1_instantiate.param_names = ["package", "integrator", "jit_rhs_before"]


###############
# First Step
###############


def setup_time_f1_first_step(package, integrator):
    define_func(package, True)
    define_sol(package, integrator)


def time_f1_first_step(package, integrator):
    sol.step()


time_f1_first_step.setup = setup_time_f1_first_step
time_f1_first_step.params = (PACKAGES, INTEGRATORS)
time_f1_first_step.param_names = ["package", "integrator"]


###############
# Run 10k
###############


def setup_time_f1_run10k(package, integrator, other):
    if other == NumbaStepModes.INTERNAL_LOOP.name and package == "scipy":
        raise NotImplementedError

    define_func(package, True)
    define_sol(package, integrator)
    sol.step()
    if hasattr(sol, "nsteps"):
        sol.nsteps(1)


def time_f1_run10k(package, integrator, other):
    if other == NumbaStepModes.INTERNAL_LOOP.name:
        sol.nsteps(10_000)
    else:
        for n in range(10_000):
            sol.step()


time_f1_run10k.setup = setup_time_f1_run10k
time_f1_run10k.params = (PACKAGES, INTEGRATORS, (NumbaStepModes.INTERNAL_LOOP.name,
                                                 NumbaStepModes.EXTERNAL_LOOP.name))
time_f1_run10k.param_names = ["package", "integrator", "other"]