Simulation of the Euler-Bernoulli Beam
--------------------------------------

In this example, the hyperbolic equation of an euler bernoulli beam, clamped
at one side is considered.
The domain of the vertical beam excitation :math:`x(z,t)` is regarded to be
:math:`[0, 1] \times \mathbb{R}^+` .

The governing equation reads:

.. math::
    :nowrap:

    \begin{align*}
        \partial^2_t x(z,t) &= - \frac{EI}{\mu} \partial^4_z x(z,t)\\
        x(0,t) &= 0 \\
        \partial_z x(0,t) &= 0 \\
        \partial^2_z x(0,t) &= 0 \\
        \partial^3_z x(0,t) &= u(t)
    \end{align*}

With the E-module :math:`E`, the second moment of area :math:`I` and the
specific density :math:`\mu` .
In this example, the input :math:`u(t)` mimics the force impulse occurring if
the beam is hit by a hammer.

Spatial disretization
+++++++++++++++++++++

For further analysis let :math:`D_z(x) = - \frac{EI}{\mu} \partial^4_z x` denote
the spatial operator and

.. math:: R(x) = \begin{pmatrix} x(0,t) \\
                                \partial_z x(0,t) \\
                                \partial^2_z x(1,t) \\
                                \partial^3_z x(1,t)
                \end{pmatrix} = \boldsymbol{0}

denote the boundary operator.

Repeated partial integration of the expression

.. math::
    :nowrap:

    \begin{align*}
        \left< D_z x | \varphi \right>
            &= \frac{EI}{\mu} \left< \partial^4_z x | y \right>\\
        &= \frac{EI}{\mu} \left(
            \left[\partial^3_z x \varphi \right]_0^1
            -\left[\partial^2_z x \partial_z\varphi \right]_0^1
            \left[\partial^1_z x \partial^2_z\varphi \right]_0^1
            -\left[ x \partial^3_z\varphi \right]_0^1
            \right)\\
        &\quad +  \frac{EI}{\mu} \left< x | \partial^4_z y \right>
    \end{align*}

and application of the boundary conditions shows that
:math:`\left< D_z x | y \right> = \left< x | D_z y \right>` if
:math:`Rx = R\varphi` . Therefore, the spatial operator is self-adjoint.


Modal Analysis
++++++++++++++

Since the operator is self-adjoined, the eigenvectors of the operator generate a
orthonormal basis, which can be used for the approximation.

Hence, the problem to solve reads:

.. math::
    \frac{EI}{\nu} \partial^4_z \varphi(z, t) = \lambda \varphi(z, t)

Which is achieved by choosing

.. math::
    :nowrap:

    \begin{align*}
        \varphi(z) &= \cos{\left (\gamma z \right )} - \cosh{\left (\gamma z \right )} \\
        &\quad- \frac{\left(e^{2 \gamma} + 2 e^{\gamma} \cos{\left (\gamma \right )} + 1\right) \sin{\left (\gamma z \right )}}{e^{2 \gamma} + 2 e^{\gamma} \sin{\left (\gamma \right )} - 1} \\
        &\quad+ \frac{\left(e^{2 \gamma} + 2 e^{\gamma} \cos{\left (\gamma \right )} + 1\right) \sinh{\left (\gamma z \right )}}{e^{2 \gamma} + 2 e^{\gamma} \sin{\left (\gamma \right )} - 1}
    \end{align*}

where :math:`\gamma = \left(-\lambda \frac{\nu}{EI}\right)^{\frac{1}{4}}` .
This is done in :func:`calc_eigen` .

Using this basis, the approximation

.. math::
    x(z, t) \approx \sum\limits_{i=1}^N c_i(t) \varphi_i(z)

is introduced.

Projecting the equation on the basis of eigenvectors
:math:`\boldsymbol{\varphi}(z)` yields

.. math::
    \left<\partial^2_t x | \varphi_k \right>  = \left< D_z x | \varphi_k \right>

for every :math:`k=1, \dotsc, N` . Substituting the approximation leads to

.. math::
    \left<\partial^2_t x | \varphi_k \right>  =
        \sum\limits_{i=1}^N c_i(t) \left< D_z \varphi_i | \varphi_k \right>

where the application of :math:`D_z` and the inner product can be swapped since
:math:`D_z` is a bounded operator.
Finally, uing the solution of the eigen problem yields

.. math::
    \left<\partial^2_t x | \varphi_k \right>  =
        \sum\limits_{i=1}^N c_i(t) \lambda_i \left< \varphi_i | \varphi_k \right>

which simplifies to

.. math::
    \left<\partial^2_t x | \varphi_k \right>  = c_k(t)\lambda_k

since, due to orthonormality, :math:`\left< \varphi_i | \varphi_k \right>` is
zero for all :math:`i \neq k` and :math:`1` for :math:`i = k` .

Performing the same steps for the left-hand side yields:

.. math::
    \ddot c_k(t) = \lambda_k c_k(t) .

Thus, the ordinary differential equation system

.. math::
    \dot{\boldsymbol{b}}(t) = \begin{pmatrix}
        \boldsymbol{A} \\
        \boldsymbol{\Lambda}
    \end{pmatrix} \boldsymbol{b}(t)

with the new state vector

.. math::
    \boldsymbol{b}(t) = \begin{pmatrix}
         c_1(t), \dotsc, c_N(t), \dot c_1(t), \dotsc, \dot c_N(t)
    \end{pmatrix}^T

the integrator chain :math:`\boldsymbol{A}` and eigenvalue matrix
:math:`\boldsymbol{\Lambda} = \textup{diag}(\lambda_1, \dotsc, \lambda_N)` is
derived.
Since the resulting system is autonomous, apart from interesting simulations,
not much can be done fro a control perepective.

Alternative Variant
+++++++++++++++++++

Using the weak formulation, which is gained by projecting the original equation
on a set of test functions and fully shifting the spatial operator onto the
test functions and substituting the boundary conditions

.. math::
    :nowrap:

    \begin{align*}
        \left< D_z x | \varphi \right>
            &= \frac{EI}{\mu} \left< \partial^4_z x | y \right>\\
        &= \frac{EI}{\mu} \big(
            u(t) \varphi(1) - \partial^3_z x(0) \varphi(0)
            + \partial^2_z x(0) \partial_z\varphi(0) \\
        &\qquad
            + \partial_z x(1) \partial^2_z\varphi(1)
            - x(1) \partial^3_z\varphi(1) \\
        &\qquad
            + \left< x | \partial^4_z y \right>
            \big)
    \end{align*}

and inserting the modal approximation from above, the system can be simulated
for every arbitrary input :math:`u(t)` . Note that this approximation converges
over the whole spatial domain, but not punctually, since using the eigenvectors
:math:`\partial^3_z \varphi(1) = 0` but :math:`\partial^3_z x(1) = u(t)` .

.. only:: html

* source code:

.. literalinclude:: /../pyinduct/examples/euler_bernoulli_beam.py

