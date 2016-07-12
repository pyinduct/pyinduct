Transport System
----------------

.. math::
    :nowrap:

    \begin{align*}
    \dot{x}(z,t) + v x'&(z,t) = 0 && z\in (0, l], t>0\\
    x(z,0) &= x_0(z) && z\in [0,l]\\
    x(0,t) &= u(t) && t>0\\
    \end{align*}

.. only:: html

* :math:`x_0(z)=0`
* :math:`u(t)` (:py:class:`pyinduct.trajectory.SignalGenerator`):
    .. image:: /ressources/transport_system.png
        :scale: 60 %
        :align: center

* :math:`x(z,t)`:
    .. image:: /ressources/transport_system.gif
        :scale: 60 %
        :align: center


* source code:

.. literalinclude:: /../examples/transport_system.py
