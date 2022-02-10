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


0.5.1 (2020-09-23)
---------------------

Bugfixes:

* Problem with nan values in EvalData
* Activation of numpy strict mode in normal operation
* Comparison warnings in various places
* Issues with evaluation of ComposedFunctionVector
* Errors in evaluate_approximation with CompoundFunctionVectors
* Deprecation warnings in visualization code
* Broken default color scheme now uses matplotlib defaults
* Corner cases for evaluate approximation
* Made EvalData robust against NaN values in int output data array
* Index error in animation handler of SurfacePlot
* Added support for nan values in SurfacePlot
* Removed strict type check to supply different systems for simulation
* Added correct handling an NaN to spline interpolator of EvalData
* Several issues in PgSurfacePlot
* Introduced fill value for EvalData objects
* Deactivated SplineInterpolator due to bad performance
* Cleanup in SWM example tests
* Suppressed plots in examples for global test run
* Complete weak formulation test case for swm example
* Updated test command since call via setup.py got deprecated

CI related changes:

* Solved issues with screen buffer
* restructured test suite
* test now run on the installed package instead of the source tree
* updated rtd config to enable building the documentation again


0.5.2 (2022-02-10)
---------------------
Changes:

* Switch complex conjugated element in inner product
* Allow complex scales in normalize_base
* Improve robustness of normalize_base
* Add `apply_operator()` method to `BaseFraction` (#102)
* Add missing tests for BaseFraction
* Added tab10 coloring to `visualize_functions`
* Add support for Python 3.10

Bugfixes:

* Fix Some points about scalar product spaces (#101)
* Fix `project_on_base` for Base with only on Fraction (#104)
* Add sesquilinear property to dot_product
* Remove faulty dot product shortcut
* Fix broken imports from collections module

CI related changes:

* Migrated to CI pipeline to Github Actions