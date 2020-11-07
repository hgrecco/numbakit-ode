
Tutorial
========

Follow the steps below and learn how to use numbakit-ode to solve a system of
ordinary differential equations. The API is very similar to SciPy integrate
module so you will be up to speed in a blink.


Solving a simple equation
-------------------------

Let's solve an equation. Consider as simple differential equation

.. math::

    \frac{dy}{dt} = -0.1 \, y

Let's write the right hand side of the equation in Python

.. doctest::

   >>> def rhs(t, y):
   ...     return -0.1 * y

A few important things:

1. The first argument has to be the time (even if you do not use it).
2. The second argument has to be the variable and accept a numpy array.

Let's integrate this equation using a **RungeKutta45** to a set of specific
time points, with `1` as the initial value.

.. doctest::

   >>> y0 = 1.
   >>> solver = nbkode.RungeKutta45(rhs, y0)
   >>> ts = np.linspace(0, 10, 100)
   >>> ts, ys = solver.run(ts)

and that's it. You can plot this result:

   >>> import matplotlib.pyplot as plt
   >>> plt.plot(ts, ys)

A solver instance remember and therefore the following command will not
recalculate the what has happened between 0 and 10.

   >>> ts2 = np.linspace(20, 40, 100)
   >>> ts, ys2 = solver.run(ts)
   >>> plt.plot(ts2, ys2)

You can do this as many times as you want, as long as you move forward
in time. Notice that in the first run, we integrated from 0 to 10. While in
the second from 20 to 40. The solver automatically integrated from 10 to 20
when `run` was called the second time.s

It is important to consider that the chosen time points might not match the
ones defined by the integration algorithm, nbkode interpolates to the values
you have provided. This is usually what you need. However, if you want to
integrate and get the steps without interpolation just call `run` using
the last time time point as an argument (a number, not an iterable).

.. doctest::

   >>> y0 = 1.
   >>> solver = nbkode.RungeKutta45(rhs, y0)
   >>> ts, ys = solver.run(10)

and and again. You can plot this result.

   >>> import matplotlib.pyplot as plt
   >>> plt.plot(t, y)

Notice that the points are more sparse. They are also unevenly distributed
because RungeKutta45 is a variable step integrator.

Finally, you can use the `move_to` function if you want to calculate until
a specific time without storing the intermediate results

   >>> y0 = 1.
   >>> solver = nbkode.RungeKutta45(rhs, y0)
   >>> t, y = solver.move_to(10)


At any time, you can find out the current time, variable and rhs.

    >>> print(solver.t, solver.y, solver.f)


Parameters
----------

If your right hand side contains an explicit parameter:

.. doctest::

   >>> def rhs(t, y, p):
   ...     return p * y

and you do not want to elide it, the value can be given provided to
the integrator.

.. doctest::

   >>> y0 = 1.
   >>> p = -0.1
   >>> solver = nbkode.RungeKutta45(rhs, y0, params=p)
   >>> ts = np.linspace(0, 10, 100)
   >>> ts, ys = solver.run(ts)


More than one equation
----------------------

If there more than one equation,

.. math::

    \frac{dy_1}{dt} &= -0.1 \, y_1

    \frac{dy_2}{dt} &= -0.5 \, y_2

you just need to make sure that the output of the rhs is a numpy array.

So this is ok:

   >>> def rhs(t, y):
   ...     return np.asarray([-0.1 * y[0], -0.5 * y[1]])

but this is not ok (as the output is a tuple):

   >>> def rhs(t, y):
   ...     return -0.1 * y[0], -0.5 * y[1]

This is also ok (and also more elegant):

   >>> def rhs(t, y, p):
   ...     return p * y

and it can be combined with the `params` argument,

.. doctest::

   >>> y0 = [1., 2.]
   >>> p = [-0.1, -0.5]
   >>> solver = nbkode.RungeKutta45(rhs, y0, params=p)
   >>> ts = np.linspace(0, 10, 100)
   >>> ts, ys = solver.run(ts)


Step by step
------------

It is rare, but you might want to step yourself. There is an API for that:

   >>> y0 = 1.
   >>> solver = nbkode.RungeKutta45(rhs, y0)
   >>> solver.step()
   >>> print(solver.t, solver.y, solver.f)

If you want to step 100 times, it is faster to call `nsteps` as it has a
tighter, compiled loop.

   >>> solver.nsteps(100)
   >>> print(solver.t, solver.y, solver.f)



What's available
----------------

Before using numbakit-ode, you can check what solvers are implemented:

.. doctest::

   >>> import nbkode
   >>> nbkode.get_solvers()
   (<AdamsBashforth1>, <AdamsBashforth2>, <AdamsBashforth3>, <AdamsBashforth4>, <AdamsBashforth5>, <AdamsMoulton1>, <AdamsMoulton2>, <AdamsMoulton3>, <AdamsMoulton4>, <AdamsMoulton5>, <ForwardEuler>, <BackwardEuler>, <RungeKutta23>, <RungeKutta45>, <DOP853>)

Each element of this tuple is a class,

You can filter the output to list only those with fixed steps

.. doctest::

   >>> nbkode.get_solvers(fixed_step=True)
   (<AdamsBashforth1>, <AdamsBashforth2>, <AdamsBashforth3>, <AdamsBashforth4>, <AdamsBashforth5>, <AdamsMoulton1>, <AdamsMoulton2>, <AdamsMoulton3>, <AdamsMoulton4>, <AdamsMoulton5>, <ForwardEuler>, <BackwardEuler>)

or those which are explicit:

.. doctest::

   >>> nbkode.get_solvers(implicit=False)
   (<AdamsBashforth1>, <AdamsBashforth2>, <AdamsBashforth3>, <AdamsBashforth4>, <AdamsBashforth5>, <ForwardEuler>, <RungeKutta23>, <RungeKutta45>, <DOP853>)

or those of a given group:

.. doctest::

   >>> nbkode.get_solvers("euler")
   (<ForwardEuler>, <BackwardEuler>)

or groups:

.. doctest::

   >>> nbkode.get_solvers("Adams-Bashforth", "Euler")
   (<AdamsBashforth1>, <AdamsBashforth2>, <AdamsBashforth3>, <AdamsBashforth4>, <AdamsBashforth5>, <ForwardEuler>, <BackwardEuler>)

To get a list of the groups:

.. doctest::

    >>> nbkode.get_groups()
    ('Adams-Bashforth', 'Adams-Moulton', 'Euler', 'Runge-Kutta')

