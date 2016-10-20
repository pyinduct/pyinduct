# -*- coding: utf-8 -*-
# noinspection PyUnresolvedReferences
from .core import Domain, Base, Function, normalize_base, find_roots, project_on_base
# noinspection PyUnresolvedReferences
from .control import LawEvaluator, Controller
# noinspection PyUnresolvedReferences
from .eigenfunctions import *
# noinspection PyUnresolvedReferences
from .trajectory import SmoothTransition, gevrey_tanh, SignalGenerator, coefficient_recursion, power_series
# noinspection PyUnresolvedReferences
from .registry import register_base, deregister_base, get_base, is_registered
# noinspection PyUnresolvedReferences
from .placeholder import (Scalars, ScalarTerm, IntegralTerm, FieldVariable, SpatialDerivedFieldVariable,
                          TemporalDerivedFieldVariable, ScalarFunction, Product, TestFunction, Input)
# noinspection PyUnresolvedReferences
from .simulation import *
# noinspection PyUnresolvedReferences
from .shapefunctions import cure_interval, LagrangeFirstOrder, LagrangeSecondOrder
# noinspection PyUnresolvedReferences
from .visualization import EvalData, PgAnimatedPlot, PgSurfacePlot, MplSlicePlot
# noinspection PyUnresolvedReferences

__author__ = "Stefan Ecklebe, Marcus Riesmeier"
__email__ = "stefan.ecklebe@tu-dresden.de, marcus.riesmeier@umit.at"
__version__ = '0.4.0'
