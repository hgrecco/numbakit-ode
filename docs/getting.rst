.. _getting:

Installation
============

numbakit-ode depends on Numpy, Scipy and Numba. In runs on Python_ 3.7+.

While you can install these dependencies with pip, the suggested way is
to use `Anaconda CE`_, a free Python distribution by Continuum Analytics
that includes many scientific packages. The `pip` equivalent in anaconda
is `conda`:

    $ conda install numpy scipy numba

Then you can install it (or upgrade to the latest version) using pip_::

    $ pip install -U nbkode

or with conda, with the conda-forge channel (*soon*):

.. code-block:: bash

    $ conda install -c conda-forge numbakit-ode

and then simply enjoy it!

That's all! You can check that numbakit-ode is correctly installed by
starting up python, and importing nbkode:

.. code-block:: python

    >>> import nbkode
    >>> nbkode.__version__  # doctest: +SKIP

Or running the test suite:

.. code-block:: python

    >>> nbkode.test()


Getting the code
----------------

You can also get the code from PyPI_ or GitHub_. You can either clone the public repository::

    $ git clone git://github.com/hgrecco/nbkode.git

Download the tarball::

    $ curl -OL https://github.com/hgrecco/nbkode/tarball/master

Or, download the zipball::

    $ curl -OL https://github.com/hgrecco/nbkode/zipball/master

Once you have a copy of the source, you can embed it in your Python package, or install it into your site-packages easily::

    $ python setup.py install


.. _easy_install: http://pypi.python.org/pypi/setuptools
.. _Python: http://www.python.org/
.. _pip: http://www.pip-installer.org/
.. _`Anaconda CE`: https://store.continuum.io/cshop/anaconda
.. _PyPI: https://pypi.python.org/pypi/numbakit-ode/
.. _GitHub: https://github.com/hgrecco/numbakit-ode
