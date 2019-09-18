.. :changelog:

History
-------

0.1.0 (2015-01-15)
---------------------

* First Code

0.2.0 (2015-07-10)
---------------------

* Version 0.2

0.3.0 (2016-01-01)
---------------------

* Version 0.3

0.4.0 (2016-03-21)
---------------------

* Version 0.4
* Change from Python2 to Python3

0.5.0 (2019-09-14)
---------------------
Features:

* Unification of `cure_interval` which can now be called directly as static
* Added functionality to parse pure `TestFunction` products
* Added visualization of functions with noncontinuous domain
* Support for Observer Approximation via `ObserverFeedback`
* Added complete support for `ComposedFunctionVector`
* Concept of `matching` and `intermediate` bases for easier approximation handling
* Added call to clear the base registry
* Added `StackedBase` for easier handling of compound approximation bases
* Added `ConstantFunction` class
* New Example: Simulation of Euler-Bernoulli-Beam
* New Example: Coupled PDEs within a pipe-flow simulation
* New Example: Output feedback for the String-with-Mass System
* Extended Example: Output Feedback for the Reaction-Advection-Diffusion System

Changes:

* Removed former derivative order limit of two
* Deprecated use of `exponent` in `FieldVariable`
* Made `derive` of FieldVariable keyword-only to avoid error
* Extended the test suite by a great amount of cases
* Speed improvements for dot products (a846d2d)
* Refactored the control module into the feedback module to use common calls
  for controller and observer design
* Improved handling and computation of transformation hints
* Made scalar product vectorization explicit and accessible
* Changed license from gpl v3 to bsd 3-clause


Bugfixes:

* Bugfix for `fill_axis` parameter of `EvalData`
* Bugfix in `find_roots` if no roots where found or asked for
* Bugfix for several errors in `visualize_roots`
* Bugfix in `_simplify_product` of `Product` where the location of
  scalar_function was ignored
* Bugfix for `IntegralTerm` where limits were not checked
* Bugfix for boundary values of derivatives in `Lag2ndOrder`
* Fixed Issue concerning complex state space matrices
  method on the class to be used for curing.
* A few fixes on the way to better plotting (739a70b)
* Fixed various deprecation warnings for scipy, numpy and sphinx
* Fixed bug in `Domain` constructor for degenerated case (1 point domain)
* Bugfix for derivatives of `Input`
* Bugfixes  for `SimulationInput`
* Fixed typos in various docstrings
