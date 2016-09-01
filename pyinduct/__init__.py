# -*- coding: utf-8 -*-
# noinspection PyUnresolvedReferences
from .trajectory import SmoothTransition, gevrey_tanh
# noinspection PyUnresolvedReferences
from .registry import register_base, deregister_base, get_base, is_registered
# noinspection PyUnresolvedReferences
from .core import Function, normalize_base
# noinspection PyUnresolvedReferences
from pyinduct.simulation import FeedbackLaw, Feedback
# noinspection PyUnresolvedReferences
from .placeholder import (Scalars, ScalarTerm, IntegralTerm, FieldVariable, SpatialDerivedFieldVariable,
                          TemporalDerivedFieldVariable, Product, TestFunction, Input)
# noinspection PyUnresolvedReferences
from .simulation import (Domain, EvalData, SimulationInput, SimulationInputSum, WeakFormulation, simulate_system,
                         process_sim_data, evaluate_approximation)
# noinspection PyUnresolvedReferences
from .shapefunctions import cure_interval, LagrangeFirstOrder, LagrangeSecondOrder
# noinspection PyUnresolvedReferences
from .visualization import PgAnimatedPlot, PgSurfacePlot
# noinspection PyUnresolvedReferences
from .utils import find_roots

__author__ = "Stefan Ecklebe, Marcus Riesmeier"
__email__ = "stefan.ecklebe@tu-dresden.de, marcus.riesmeier@umit.at"
__version__ = '0.4.0'
