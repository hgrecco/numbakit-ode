.. _faq:

Frequently asked questions
==========================

Why the name *numbakit-ode*?
----------------------------

We took inspiration from the scikit project, which has been building
a great ecosystem of science related python packages for a while.
We think that it would be great to have a group of independent packages
that levarage Numba for different tasks. **numbakit-ode** aims to speed
up the integration of ordinary differential equations.

We do not claim ownership of the *numbakit* prefix. On the contrary, we
would be very happy if other projects use it as well.

While we hope that at some point Numba can support a larger subset
of Python constructs, we think that there will always be a place
for numbakit packages as the dynamic nature of Python makes it very
hard to compile everything.


What is the state of the project?
---------------------------------

We have more than a dozen integrators implemented, including 3 of the 6
available in `Scipy Integrate`_. We test our implementations against other
more stablished ones when possible to ensure correctness.

In relation to speed, integration is fast. However, instantiation of a solver
is slow because compilation is required. Therefore, Scipy outperform numbakit-ode
for short simulations

Numba cannot (yet) cache the compiled code. But as soon as it can, this overhead
will be gone.


Is integration faster thant SciPy?
----------------------------------

Yes.


Really?
-------

Yes.


How much?
---------

That depends on the integrator and the ODE system your are trying to use,
It is not uncommon to get a 10x speed up. Take a look at the benchmarks_


How is it possible if the codebase is 100% in Python?
-----------------------------------------------------

Actually Numba_ does all the heavy work, so the applause should go to
the numba devs. We just make use of it.


Why using Numba? Why not c, fortran, <your favorite language>?
--------------------------------------------------------------

We love Python, and Numba allows you to compile Python code into a machine
code.



.. _`NumPy`: http://www.numpy.org/
.. _`SciPy Integrate`: https://docs.scipy.org/doc/scipy/reference/integrate.html
.. _`Numba`: https://numba.pydata.org/
.. _`benchmarks`: https://hgrecco.github.io/numbakit-ode/


