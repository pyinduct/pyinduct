# -*- coding: utf-8 -*-
# noinspection PyUnresolvedReferences
from .registry import register_base, deregister_base, get_base, is_registered
# noinspection PyUnresolvedReferences
from .core import Function
# noinspection PyUnresolvedReferences
from .placeholder import (ScalarTerm, IntegralTerm, FieldVariable, SpatialDerivedFieldVariable,
                          TemporalDerivedFieldVariable, Product, TestFunction, Input)
# noinspection PyUnresolvedReferences
from .simulation import Domain, EvalData, SimulationInput, WeakFormulation, simulate_system
# noinspection PyUnresolvedReferences
from .shapefunctions import cure_interval, LagrangeFirstOrder, LagrangeSecondOrder
# noinspection PyUnresolvedReferences
from .visualization import PgAnimatedPlot
# noinspection PyUnresolvedReferences
from .trajectory import SmoothTransition

__author__ = "Stefan Ecklebe, Marcus Riesmeier"
__email__ = "stefan.ecklebe@tu-dresden.de, marcus.riesmeier@tu-dresden.de"
__version__ = '0.3.0'
