==========
Simulation
==========

This module consist of three parts.

    Simulation:
        Simulation infrastructure with helpers and data structures for preprocessing of the given equations
        and functions for postprocessing of simulation data.

    Controller:
        All classes and functions related to the creation of controllers as well as the implementation
        for simulation purposes.

    Observer:
        Some objects for observer implementation which are mostly a combination from the objects for
        simulation and control tasks.


Simulation
==========

.. automodule:: pyinduct.simulation
    :members: Domain, SimulationInput, SimulationInputSum, SimulationInputVector, WeakFormulation, CanonicalForm, CanonicalForms, StateSpace, simulate_system, simulate_systems, process_sim_data, parse_weak_formulation, simulate_state_space, evaluate_approximation
    :show-inheritance:


Control
=======

.. automodule:: pyinduct.simulation
    :members: FeedbackLaw, Feedback, LawEvaluator, approximate_feedback_law
    :show-inheritance:

.. autofunction:: pyinduct.simulation._parse_feedback_law


Observer
========

.. automodule:: pyinduct.simulation
    :members: ObserverError, Observer, build_observer_from_state_space
    :show-inheritance:
