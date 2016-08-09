R.a.d. eq. with dirichlet b.c. (fem approximation)
--------------------------------------------------

Reaction-advection-diffusion equation with dirichlet boundary condition by :math:`z=0`
and dirichlet actuation by :math:`z=l`.

.. math::
    :nowrap:

    \begin{align*}
        \dot{x}(z,t) = a_2x''&(z,t) + a_1x'(z,t) + a_0x(z,t)  && z\in (0, l), t>0\\
        x(z,0) &= x_0(z) && z\in [0,l]\\
        x(0,t) &= 0 && t>0\\
        x(l,t) &= u(t) && t>0
    \end{align*}

* example: heat equation

    - :math:`a_2=1,\quad a_1= 0, \quad a_0=0, \quad x_0(z)=0`
    - :math:`u(t)` --> :py:class:`pyinduct.trajectory.RadTrajectory`

        .. image:: /ressources/rad_diri_traj.png
            :scale: 40 %
            :align: center

    - :math:`x(z,t)`

        .. image:: /ressources/rad_diri_x_ani.gif
            :scale: 40 %
            :align: center

    - :math:`x'(z,t)`

        .. image:: /ressources/rad_diri_dx_ani.gif
            :scale: 40 %
            :align: center

    - corresponding 3d plots

+-----------------------------------------+------------------------------------------+
|     :math:`x(z,t)`                      |     :math:`x'(z,t)`                      |
+-----------------------------------------+------------------------------------------+
|.. image:: /ressources/rad_diri_3d_x.png |.. image:: /ressources/rad_diri_3d_dx.png |
+-----------------------------------------+------------------------------------------+

* with:

    - inital functions :math:`\varphi_1(z),...,\varphi_{n+1}(z)`
    - test functions :math:`\varphi_1(z),...,\varphi_n(z)`
    - where the functions :math:`\varphi_1(z),..,\varphi_n(z)` met the homogeneous b.c.

        .. math:: \varphi_1(l),..,\varphi_n(l)=\varphi_1(0),..,\varphi_n(0)=0

    - only :math:`\varphi_{n+1}` can draw the actuation
    - functions :math:`\varphi_1(z),...,\varphi_{n+1}(z)` e.g. from type :py:class:`pyinduct.shapefunctions.LagrangeFirstOrder` or
      :py:class:`pyinduct.shapefunctions.LagrangeSecondOrder`, see :py:mod:`pyinduct.shapefunctions`

* approach:

.. math:: x(z,t) = \sum_{i=1}^{n+1} x_i^*(t) \varphi_i(z)\Big|_{x^*_{n+1}=u} =  \underbrace{\sum_{i=1}^{n} x_i^*(t) \varphi_i(z)}_{\hat x(z,t)} + \varphi_{n+1}(z) u(t)

* weak formulation...

.. math::
    :nowrap:

    \begin{align*}
        \langle\dot{x}(z,t),\varphi_j(z)\rangle &=
        a_2 \langle x''(z,t),\varphi_j(z)\rangle \\
        &\hphantom =+
        a_1 \langle x'(z,t), \varphi_j(z)\rangle  +
        a_0 \langle x(z,t), \varphi_j(z)\rangle && j=1,...,n
    \end{align*}

* ... and derivation shift to work with lagrange 1st order initial functions

.. math::
    :nowrap:

        \begin{align*}
        \langle\dot{x}(z,t),\varphi_j(z)\rangle &=
        \overbrace{[a_2 [x'(z,t)\varphi_j(z)]_0^l}^{=0} - a_2 \langle x'(z,t),\varphi'_j(z)\rangle \\
        &\hphantom =+
        a_1 \langle x'(z,t), \varphi_j(z)\rangle  +
        a_0 \langle x(z,t), \varphi_j(z)\rangle && j=1,...,n \\
        \langle\dot{\hat{x}}(z,t),\varphi_j(z)\rangle + \langle\varphi_{N+1}(z),\varphi_j(z)\rangle \dot u(t) &= - a_2 \langle \hat x'(z,t),\varphi'_j(z)\rangle - a_2 \langle \varphi'_{N+1}(z),\varphi'_j(z)\rangle u(t) \\
        &\hphantom =+
        a_1 \langle \hat x'(z,t), \varphi_j(z)\rangle + a_1 \langle \varphi'_{N+1}(z), \varphi_j(z)\rangle u(t)  + \\
        &\hphantom =+
        a_0 \langle \hat x(z,t), \varphi_j(z)\rangle + a_0 \langle \varphi_{N+1}(z), \varphi_j(z)\rangle u(t) && j=1,...,n
    \end{align*}

* leads to state space model for the weights :math:`\boldsymbol{x}^*=(x_1^*,...,x_n^*)^T`:

.. math::
    :nowrap:

    \begin{align*}
        \dot{\boldsymbol{x}}^*(t) = A \boldsymbol x^*(t) + \boldsymbol b_0 u(t) + \boldsymbol b_1 \dot{u}(t)
    \end{align*}

* input derivative elimination through the transformation:

    - :math:`\bar{\boldsymbol{x}}^* = \tilde A \boldsymbol x^* - \boldsymbol{b}_1 u`
    - :math:`\text{e.g.: } \tilde A = I`
    - leads to

.. math::
    :nowrap:

    \begin{align*}
        \dot{\bar{\boldsymbol{x}}}^*(t) &= \tilde A A\tilde A^{-1} \bar{\boldsymbol{x}}^*(t) + \tilde A(A\boldsymbol b_1 + \boldsymbol b_0) u(t) \\
        &= \bar A \bar{\boldsymbol{x}}^*(t) +\bar{\boldsymbol{b}} u(t)
    \end{align*}

.. only:: html

* source code:

.. literalinclude:: /../examples/rad_eq_diri_fem.py

