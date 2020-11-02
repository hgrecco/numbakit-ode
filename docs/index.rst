:orphan:

numbakit-ode: leveraging numba to speed up ODE integration
==========================================================

.. image:: _static/logo-full.png
   :alt: numbakit-ode: **physical quantities**
   :class: floatingflask


numbakit-ode (nbkode) is a Python package to solve
**ordinary differential equations (ODE)** that uses
Numba_ to compile code and therefore speed up calculations.

The API is very similar to scipy's integrate module therefore
allowing for easy migration.

It runs in Python 3.7+ depending on NumPy_, SciPy_ and Numba_.
It is licensed under BSD.

It is extremely easy and natural to use:

.. code-block:: python

    >>> import nbkode
    >>> def func(t, y, k):
    ...     return -k * y
    >>> t0 = 0.
    >>> y0 = 1.
    >>> solver = nbkode.Euler(func, t0, y0, args=(1, ))
    >>> solver.run(10)

You can get a list of all solvers:

.. code-block:: python

    >>> import nbkode
    >>> nbkode.get_solvers()

or filter by characteristics or group name (or names).

.. code-block:: python

    >>> nbkode.get_solvers(implicit=False, fixed_step=True)
    >>> nbkode.get_solvers('euler', 'adam-bashforth')


Quick Installation
------------------

To install numbakit-ode, simply (*soon*):

.. code-block:: bash

    $ pip install numbakit-ode

or utilizing conda, with the conda-forge channel (*soon*):

.. code-block:: bash

    $ conda install -c conda-forge numbakit-ode

and then simply enjoy it!


User Guide
----------

.. toctree::
    :maxdepth: 1

    getting
    tutorial
    faq
    developers_reference



Design principles
-----------------

**Fast**: We love Numba_. It allows you to write clean Python code
that translates to optimized machine code at runtime. We aim to
be able to leverage this power to solve a system of ordinary
differential equations.

**Simple but useful API**: Solvers are classes easy to instantiate,
with sensible defaults and convenient methods.

**Correctness**: We check against established libraries like SciPy_
that our implementation match those of established libraries using
automated testing.

**Data driven development**: We take decisions based on data, and for this
purpose we measure the performance of each part of the package, and the effect
of each change we make.


----

numbakit-ode is maintained by a community. See AUTHORS_ for a complete list.

To review an ordered list of notable changes for each version of a project,
see CHANGES_


.. _`NumPy`: http://www.numpy.org/
.. _`SciPy`: http://www.scipy.org/
.. _`Numba`: https://numba.pydata.org/
.. _`pytest`: https://docs.pytest.org/
.. _`airspeed velocity`: https://asv.readthedocs.io
.. _`AUTHORS`: https://github.com/hgrecco/numbakit-ode/blob/master/AUTHORS
.. _`CHANGES`: https://github.com/hgrecco/numbakit-ode/blob/master/CHANGES
