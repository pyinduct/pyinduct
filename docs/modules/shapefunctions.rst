==============
Shapefunctions
==============

.. autoapimodule:: pyinduct.shapefunctions
    :no-imported-members:

.. autoapiclass:: pyinduct.shapefunctions.ShapeFunction
    :members:

Shapefunction Types
-------------------
.. autoapiclass:: pyinduct.shapefunctions.LagrangeFirstOrder
    :members:
    :undoc-members:
    :show-inheritance:

    Example plot of the functions :code:`funcs` generated with

    >>> nodes, funcs = cure_interval(LagrangeFirstOrder, (0, 1), node_count=7)

    .. image:: ../ressources/lag1st_order.png
        :scale: 70 %
        :align: center

.. autoapiclass:: pyinduct.shapefunctions.LagrangeSecondOrder
    :members:
    :undoc-members:
    :show-inheritance:

    Example plot of the functions :code:`funcs` generated with

    >>> nodes, funcs = cure_interval(LagrangeSecondOrder, (0, 1), node_count=7)

    .. image:: ../ressources/lag2nd_order.png
        :scale: 70 %
        :align: center

.. autoapiclass:: pyinduct.shapefunctions.LagrangeNthOrder
    :members:
    :undoc-members:
    :show-inheritance:

    Example plot of the functions :code:`funcs` generated with

    >>> nodes, funcs = pi.cure_interval(sh.LagrangeNthOrder, (0, 1), node_count=9, order=4)

    .. image:: ../ressources/lag4th_order.png
        :scale: 70 %
        :align: center




