"""
    nbkode.nbcompat.newton
    ~~~~~~~~~~~~~~~~~~~~~~

    Newton methods.

    :copyright: 2020 by nbkode Authors, see AUTHORS for more details.
    :license: BSD, see LICENSE for more details.
"""

import enum
import operator
import warnings

import numpy as np

try:
    from scipy.optimize import _zeros_py as zeros
except ImportError:
    # SciPy < 1.8.0
    from scipy.optimize import zeros

from .common import isclose
from .nb_to_import import numba


class NewtonEnum(enum.IntEnum):
    OK = 0
    DERIVATIVE_WAS_ZERO = 1
    TOLERANCE_REACHED = 2
    NOT_CONVERGED = 3


@numba.njit
def _results_select(*args):
    """Select from a tuple of (root, funccalls, iterations, flag)"""
    return args


_ECONVERR = zeros._ECONVERR
_ECONVERGED = zeros._ECONVERGED


def newton(
    func,
    x0,
    fprime=None,
    args=(),
    tol=1.48e-8,
    maxiter=50,
    fprime2=None,
    x1=None,
    rtol=0.0,
    full_output=False,
    disp=True,
):
    """
    Find a zero of a real or complex function using the Newton-Raphson
    (or secant or Halley's) method.

    Find a zero of the function `func` given a nearby starting point `x0`.
    The Newton-Raphson method is used if the derivative `fprime` of `func`
    is provided, otherwise the secant method is used. If the second order
    derivative `fprime2` of `func` is also provided, then Halley's method is
    used.

    If `x0` is a sequence with more than one item, then `newton` returns an
    array, and `func` must be vectorized and return a sequence or array of the
    same shape as its first argument. If `fprime` or `fprime2` is given, then
    its return must also have the same shape.

    Parameters
    ----------
    func : callable
        The function whose zero is wanted. It must be a function of a
        single variable of the form ``f(x,a,b,c...)``, where ``a,b,c...``
        are extra arguments that can be passed in the `args` parameter.
    x0 : float, sequence, or ndarray
        An initial estimate of the zero that should be somewhere near the
        actual zero. If not scalar, then `func` must be vectorized and return
        a sequence or array of the same shape as its first argument.
    fprime : callable, optional
        The derivative of the function when available and convenient. If it
        is None (default), then the secant method is used.
    args : tuple, optional
        Extra arguments to be used in the function call.
    tol : float, optional
        The allowable error of the zero value. If `func` is complex-valued,
        a larger `tol` is recommended as both the real and imaginary parts
        of `x` contribute to ``|x - x0|``.
    maxiter : int, optional
        Maximum number of iterations.
    fprime2 : callable, optional
        The second order derivative of the function when available and
        convenient. If it is None (default), then the normal Newton-Raphson
        or the secant method is used. If it is not None, then Halley's method
        is used.
    x1 : float, optional
        Another estimate of the zero that should be somewhere near the
        actual zero. Used if `fprime` is not provided.
    rtol : float, optional
        Tolerance (relative) for termination.
    full_output : bool, optional
        If `full_output` is False (default), the root is returned.
        If True and `x0` is scalar, the return value is ``(x, r)``, where ``x``
        is the root and ``r`` is a `RootResults` object.
        If True and `x0` is non-scalar, the return value is ``(x, converged,
        zero_der)`` (see Returns section for details).
    disp : bool, optional
        If True, raise a RuntimeError if the algorithm didn't converge, with
        the error message containing the number of iterations and current
        function value. Otherwise, the convergence status is recorded in a
        `RootResults` return object.
        Ignored if `x0` is not scalar.
        *Note: this has little to do with displaying, however,
        the `disp` keyword cannot be renamed for backwards compatibility.*

    Returns
    -------
    root : float, sequence, or ndarray
        Estimated location where function is zero.
    r : `RootResults`, optional
        Present if ``full_output=True`` and `x0` is scalar.
        Object containing information about the convergence. In particular,
        ``r.converged`` is True if the routine converged.
    converged : ndarray of bool, optional
        Present if ``full_output=True`` and `x0` is non-scalar.
        For vector functions, indicates which elements converged successfully.
    zero_der : ndarray of bool, optional
        Present if ``full_output=True`` and `x0` is non-scalar.
        For vector functions, indicates which elements had a zero derivative.

    See Also
    --------
    brentq, brenth, ridder, bisect
    fsolve : find zeros in N dimensions.

    Notes
    -----
    The convergence rate of the Newton-Raphson method is quadratic,
    the Halley method is cubic, and the secant method is
    sub-quadratic. This means that if the function is well-behaved
    the actual error in the estimated zero after the nth iteration
    is approximately the square (cube for Halley) of the error
    after the (n-1)th step. However, the stopping criterion used
    here is the step size and there is no guarantee that a zero
    has been found. Consequently, the result should be verified.
    Safer algorithms are brentq, brenth, ridder, and bisect,
    but they all require that the root first be bracketed in an
    interval where the function changes sign. The brentq algorithm
    is recommended for general use in one dimensional problems
    when such an interval has been found.

    When `newton` is used with arrays, it is best suited for the following
    types of problems:

    * The initial guesses, `x0`, are all relatively the same distance from
      the roots.
    * Some or all of the extra arguments, `args`, are also arrays so that a
      class of similar problems can be solved together.
    * The size of the initial guesses, `x0`, is larger than O(100) elements.
      Otherwise, a naive loop may perform as well or better than a vector.

    Examples
    --------

    >>> from scipy import optimize
    >>> import matplotlib.pyplot as plt
    >>> def f(x):
    ...     return (x**3 - 1)  # only one real root at x = 1

    ``fprime`` is not provided, use the secant method:
    >>> root = optimize.newton(f, 1.5)
    >>> root
    1.0000000000000016
    >>> root = optimize.newton(f, 1.5, fprime2=lambda x: 6 * x)
    >>> root
    1.0000000000000016

    Only ``fprime`` is provided, use the Newton-Raphson method:
    >>> root = optimize.newton(f, 1.5, fprime=lambda x: 3 * x**2)
    >>> root
    1.0

    Both ``fprime2`` and ``fprime`` are provided, use Halley's method:
    >>> root = optimize.newton(f, 1.5, fprime=lambda x: 3 * x**2,
    ...                        fprime2=lambda x: 6 * x)
    >>> root
    1.0

    When we want to find zeros for a set of related starting values and/or
    function parameters, we can provide both of those as an array of inputs:
    >>> f = lambda x, a: x**3 - a
    >>> fder = lambda x, a: 3 * x**2
    >>> np.random.seed(4321)
    >>> x = np.random.randn(100)
    >>> a = np.arange(-50, 50)
    >>> vec_res = optimize.newton(f, x, fprime=fder, args=(a, ))

    The above is the equivalent of solving for each value in ``(x, a)``
    separately in a for-loop, just faster:
    >>> loop_res = [optimize.newton(f, x0, fprime=fder, args=(a0,))
    ...             for x0, a0 in zip(x, a)]
    >>> np.allclose(vec_res, loop_res)
    True

    Plot the results found for all values of ``a``:
    >>> analytical_result = np.sign(a) * np.abs(a)**(1/3)
    >>> fig = plt.figure()
    >>> ax = fig.add_subplot(111)
    >>> ax.plot(a, analytical_result, 'o')
    >>> ax.plot(a, vec_res, '.')
    >>> ax.set_xlabel('$a$')
    >>> ax.set_ylabel('$x$ where $f(x, a)=0$')
    >>> plt.show()

    """

    if tol <= 0:
        raise ValueError("tol too small (%g <= 0)" % tol)
    maxiter = operator.index(maxiter)
    if maxiter < 1:
        raise ValueError("maxiter must be greater than 0")
    if np.size(x0) > 1:
        return zeros._array_newton(
            func, x0, fprime, args, tol, maxiter, fprime2, full_output
        )

    (full_output, r), my_flag = _newton(
        func,
        x0,
        fprime,
        args,
        tol,
        maxiter,
        fprime2,
        x1,
        rtol,
        full_output,
    )

    if my_flag is not NewtonEnum.OK:

        if my_flag is NewtonEnum.DERIVATIVE_WAS_ZERO:
            msg = "Derivative was zero."
        elif my_flag is NewtonEnum.TOLERANCE_REACHED:
            msg = "Tolerance of ? reached."
        else:
            msg = ""

        if disp:
            p, funcalls, itr, flag = r
            if my_flag is NewtonEnum.NOT_CONVERGED:
                msg += "Failed to converge after %d iterations, value is %s." % (itr, p)
            else:
                msg += " Failed to converge after %d iterations, value is %s." % (
                    itr + 1,
                    p,
                )
            raise RuntimeError(msg)

        warnings.warn(msg, RuntimeWarning)

    return zeros._results_select(full_output, r)


def _newton(
    func,
    x0,
    fprime=None,
    args=(),
    tol=1.48e-8,
    maxiter=50,
    fprime2=None,
    x1=None,
    rtol=0.0,
    full_output=False,
):
    """Same signature as newton but:

    1. Has no `disp` argument
    2. Different output:
        (full_output
        (p0, number of function calls, number of iterations, flag)
        )
        NewtonEnum
    """

    my_flag = NewtonEnum.OK

    # Convert to float (don't use float(x0); this works also for complex x0)
    p0 = 1.0 * x0
    funcalls = 0
    if fprime is not None:
        # Newton-Raphson method
        for itr in range(maxiter):
            # first evaluate fval
            fval = func(p0, *args)
            funcalls += 1
            # If fval is 0, a root has been found, then terminate
            if fval == 0:
                return (
                    _results_select(full_output, (p0, funcalls, itr, _ECONVERGED)),
                    my_flag,
                )
            fder = fprime(p0, *args)
            funcalls += 1
            if fder == 0:
                my_flag = NewtonEnum.DERIVATIVE_WAS_ZERO
                # msg = "Derivative was zero."
                # if disp:
                #     msg += (
                #         " Failed to converge after %d iterations, value is %s."
                #         % (itr + 1, p0))
                #     raise RuntimeError(msg)
                # warnings.warn(msg, RuntimeWarning)
                return (
                    _results_select(full_output, (p0, funcalls, itr + 1, _ECONVERR)),
                    my_flag,
                )
            newton_step = fval / fder
            if fprime2 is not None:
                fder2 = fprime2(p0, *args)
                funcalls += 1
                # Halley's method:
                #   newton_step /= (1.0 - 0.5 * newton_step * fder2 / fder)
                # Only do it if denominator stays close enough to 1
                # Rationale: If 1-adj < 0, then Halley sends x in the
                # opposite direction to Newton. Doesn't happen if x is close
                # enough to root.
                adj = newton_step * fder2 / fder / 2
                if np.abs(adj) < 1:
                    newton_step /= 1.0 - adj
            p = p0 - newton_step
            if isclose(p, p0, rtol=rtol, atol=tol):
                return (
                    _results_select(full_output, (p, funcalls, itr + 1, _ECONVERGED)),
                    my_flag,
                )
            p0 = p
    else:
        # Secant method
        if x1 is not None:
            if x1 == x0:
                raise ValueError("x1 and x0 must be different")
            p1 = x1
        else:
            eps = 1e-4
            p1 = x0 * (1 + eps)
            p1 += eps if p1 >= 0 else -eps
        q0 = func(p0, *args)
        funcalls += 1
        q1 = func(p1, *args)
        funcalls += 1
        if np.abs(q1) < np.abs(q0):
            p0, p1, q0, q1 = p1, p0, q1, q0
        for itr in range(maxiter):
            if q1 == q0:
                if p1 != p0:
                    my_flag = NewtonEnum.TOLERANCE_REACHED
                    # msg = "Tolerance of %s reached." % (p1 - p0)
                    # if disp:
                    #     msg += (
                    #         " Failed to converge after %d iterations, value is %s."
                    #         % (itr + 1, p1))
                    #     raise RuntimeError(msg)
                    # warnings.warn(msg, RuntimeWarning)
                p = (p1 + p0) / 2.0
                return (
                    _results_select(full_output, (p, funcalls, itr + 1, _ECONVERGED)),
                    my_flag,
                )
            else:
                if np.abs(q1) > np.abs(q0):
                    p = (-q0 / q1 * p1 + p0) / (1 - q0 / q1)
                else:
                    p = (-q1 / q0 * p0 + p1) / (1 - q1 / q0)
            if isclose(p, p1, rtol=rtol, atol=tol):
                return (
                    _results_select(full_output, (p, funcalls, itr + 1, _ECONVERGED)),
                    my_flag,
                )
            p0, q0 = p1, q1
            p1 = p
            q1 = func(p1, *args)
            funcalls += 1

    # if disp:
    #     msg = ("Failed to converge after %d iterations, value is %s."
    #            % (itr + 1, p))
    #     raise RuntimeError(msg)

    return (
        _results_select(full_output, (p, funcalls, itr + 1, _ECONVERR)),
        NewtonEnum.NOT_CONVERGED,
    )


_j_newton = numba.njit()(_newton)


@numba.njit
def j_newton(
    func,
    x0,
    fprime=None,
    args=(),
    tol=1.48e-8,
    maxiter=50,
    fprime2=None,
    x1=None,
    rtol=0.0,
    full_output=False,
):
    """Same signature as newton but:

    1. Has no `disp` argument
    2. Different output:
        p0
    3. always raises RunTimeError if it does not converge.
    """

    (full_output, r), my_flag = _j_newton(
        func,
        x0,
        fprime,
        args,
        tol,
        maxiter,
        fprime2,
        x1,
        rtol,
        full_output,
    )
    p, funcalls, itr, flag = r
    if my_flag is not NewtonEnum.OK:
        raise RuntimeError
    return p


@numba.njit()
def jacobian(func, x, args=()):

    eps = 1e-10
    J = np.zeros((len(x), len(x)), dtype=np.float64)

    for i in range(len(x)):
        x1 = x.copy()
        x2 = x.copy()

        x1[i] += eps
        x2[i] -= eps

        f1 = func(x1, *args)
        f2 = func(x2, *args)

        J[:, i] = (f1 - f2) / (2 * eps)

    return J


@numba.njit()
def newton_hd_impl(
    func,
    y,
    args=(),
    atol=1.48e-8,
    rtol=0.0,
    maxiter=50,
):
    """
    Solve nonlinear system F=0 by Newton's method.

    J is the Jacobian of F. Both F and J must be functions of x.
    At input, x holds the start value. The iteration continues
    until ||F|| < eps.
    """
    value = func(y, *args)
    distance = np.linalg.norm(value, ord=2)  # l2 norm of vector
    iteration_counter = 0

    while not isclose(distance, 0, rtol=rtol, atol=atol):
        delta = np.linalg.solve(jacobian(func, y, args), -value)
        y = y + delta
        value = func(y, *args)
        distance = np.linalg.norm(value, ord=2)
        iteration_counter += 1

        if iteration_counter > maxiter:
            return y, NewtonEnum.NOT_CONVERGED

    return y, NewtonEnum.OK


@numba.njit()
def newton_hd(
    func,
    y0,
    args=(),
    atol=1.48e-8,
    rtol=0.0,
    maxiter=50,
):
    y, flag = newton_hd_impl(func, y0, args, atol, rtol, maxiter)
    if flag is not NewtonEnum.OK:
        raise RuntimeError
    return y


@numba.njit()
def bisect(fun, left, right, tol=1.48e-8, maxiter=50, rtol=0.0, args=()):
    if not left < right:
        raise ValueError("In bisect, 'left' must be smaller than 'rigth'.")
    yl = fun(left, *args)
    if yl == 0:
        return left
    yr = fun(right, *args)
    if yr == 0:
        return right
    if not yl * yr < 0:
        raise ValueError(
            "In bisect, 'fun(left, *args)' must have and opposite sign to 'fun(right, *args)'."
        )

    for _ in range(maxiter):
        mid = (left + right) / 2
        ym = fun(mid, *args)
        if isclose(ym, 0, rtol=rtol, atol=tol):
            break
        if np.sign(ym) == np.sign(yl):
            left, yl = mid, ym
        else:
            right, yr = mid, ym

    return mid
