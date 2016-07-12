==============
Shapefunctions
==============

.. automodule:: pyinduct.shapefunctions

Shapefunction Types
-------------------
.. autoclass:: pyinduct.shapefunctions.LagrangeFirstOrder
    :members:
    :undoc-members:
    :show-inheritance:

    Example plot of the functions :code:`funcs` generated with

    >>> nodes, funcs = cure_interval(LagrangeFirstOrder, (0, 1), node_count=7)

    .. image:: ../ressources/lag1st_order.png
        :scale: 70 %
        :align: center

.. autoclass:: pyinduct.shapefunctions.LagrangeSecondOrder
    :members:
    :undoc-members:
    :show-inheritance:

    Example plot of the functions :code:`funcs` generated with

    >>> nodes, funcs = cure_interval(LagrangeSecondOrder, (0, 1), node_count=7)

    .. image:: ../ressources/lag2nd_order.png
        :scale: 70 %
        :align: center

Curing an Interval
------------------
All classes contained in this module can easily be used to cure a given interval.
For example let's approximate the interval from :math:`z=0` to :math:`z=1` with 3
piecewise linear functions::

    >>> nodes, funcs = cure_interval(LagrangeFirstOrder, (0, 1), node_count=3)

The approximation nodes of the functions are chosen automatically::

    >>> list(nodes)
    [0.0, 0.5, 1.0]

.. autofunction:: pyinduct.shapefunctions.cure_interval


