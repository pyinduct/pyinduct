==========
Simulation
==========

.. automodule:: pyinduct.simulation
.. currentmodule:: pyinduct.simulation


Simulation
==========

.. autoclass:: Domain
    :members:
    :show-inheritance:
.. autoclass:: SimulationInput
    :members:
    :show-inheritance:
.. autoclass:: SimulationInputSum
    :members:
    :show-inheritance:
.. autoclass:: SimulationInputVector
    :members:
    :show-inheritance:
.. autoclass:: WeakFormulation
    :members:
    :show-inheritance:
.. autoclass:: CanonicalForm
    :members:
    :show-inheritance:
.. autoclass:: CanonicalForms
    :members:
    :show-inheritance:
.. autoclass:: StateSpace
    :members:
    :show-inheritance:

.. autofunction:: simulate_system
.. autofunction:: simulate_systems
.. autofunction:: process_sim_data
.. autofunction:: parse_weak_formulation
.. autofunction:: simulate_state_space
.. autofunction:: evaluate_approximation


Control
=======

.. autoclass:: FeedbackLaw
    :members:
    :show-inheritance:
.. autoclass:: Feedback
    :members:
    :show-inheritance:
.. autoclass:: LawEvaluator
    :members:
    :show-inheritance:

.. autofunction:: approximate_feedback_law
.. autofunction:: _parse_feedback_law


Observer
========

.. autoclass:: ObserverError
    :members:
    :show-inheritance:
.. autoclass:: Observer
    :members:
    :show-inheritance:

.. autofunction:: build_observer_from_state_space
