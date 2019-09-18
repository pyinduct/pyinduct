==============
Shapefunctions
==============

.. automodule:: pyinduct.shapefunctions

.. autoclass:: pyinduct.shapefunctions.ShapeFunction
    :members:

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

.. autoclass:: pyinduct.shapefunctions.LagrangeNthOrder
    :members:
    :undoc-members:
    :show-inheritance:

    Example plot of the functions :code:`funcs` generated with

    >>> nodes, funcs = pi.cure_interval(sh.LagrangeNthOrder, (0, 1), node_count=9, order=4)

    .. image:: ../ressources/lag4th_order.png
        :scale: 70 %
        :align: center

Curing an Interval
------------------
All classes contained in this module can easily be used to cure a given interval.
For example let's approximate the interval from :math:`z=0` to :math:`z=1` with 3
piecewise linear functions::

    >>> from pyinduct import Domain, LagrangeFirstOder
    >>> nodes = Domain(bounds0(0, 1), num=3)
    >>> list(nodes)
    [0.0, 0.5, 1.0]
    >>> funcs = LagrangeFirstOrder.cure_interval(nodes)

.. autofunction:: pyinduct.shapefunctions.cure_interval


