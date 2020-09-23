==================
Curing an Interval
==================

.. currentmodule:: pyinduct.shapefunctions

All classes contained in this module can easily be used to cure a given interval.
For example let's approximate the interval from :math:`z=0` to :math:`z=1` with 3
piecewise linear functions::

    >>> from pyinduct import Domain, LagrangeFirstOder
    >>> nodes = Domain(bounds0(0, 1), num=3)
    >>> list(nodes)
    [0.0, 0.5, 1.0]
    >>> funcs = LagrangeFirstOrder.cure_interval(nodes)
