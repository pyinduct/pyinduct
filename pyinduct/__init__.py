# -*- coding: utf-8 -*-
# noinspection PyUnresolvedReferences
from .trajectory import SmoothTransition, gevrey_tanh, SignalGenerator
# noinspection PyUnresolvedReferences
from .registry import register_base, deregister_base, get_base, is_registered
# noinspection PyUnresolvedReferences
from .core import Base, Function, normalize_base, find_roots
# noinspection PyUnresolvedReferences
from .control import LawEvaluator, Controller
# noinspection PyUnresolvedReferences
from .placeholder import (Scalars, ScalarTerm, IntegralTerm, FieldVariable, SpatialDerivedFieldVariable,
                          TemporalDerivedFieldVariable, ScalarFunction, Product, TestFunction, Input)
# noinspection PyUnresolvedReferences
from .simulation import *
# from .simulation import (Domain, SimulationInput, SimulationInputSum, WeakFormulation, parse_weak_formulation,
#                          create_state_space, simulate_system, simulate_systems, process_sim_data,
#                          evaluate_approximation)
# noinspection PyUnresolvedReferences
from .shapefunctions import cure_interval, LagrangeFirstOrder, LagrangeSecondOrder
# noinspection PyUnresolvedReferences
from .visualization import EvalData, PgAnimatedPlot, PgSurfacePlot
# noinspection PyUnresolvedReferences
# from .utils import get_parabolic_robin_weak_form

__author__ = "Stefan Ecklebe, Marcus Riesmeier"
__email__ = "stefan.ecklebe@tu-dresden.de, marcus.riesmeier@umit.at"
__version__ = '0.4.0'
