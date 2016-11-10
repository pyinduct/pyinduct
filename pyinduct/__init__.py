# -*- coding: utf-8 -*-
import os
import matplotlib as mpl

# make everybody use qt5
mpl.use('Qt5Agg')
os.environ["PYQTGRAPH_QT_LIB"] = "PyQt5"

# noinspection PyUnresolvedReferences
from .core import *
# noinspection PyUnresolvedReferences
from .control import LawEvaluator, Controller
# noinspection PyUnresolvedReferences
from .eigenfunctions import *
# noinspection PyUnresolvedReferences
from .trajectory import (SmoothTransition, gevrey_tanh, SignalGenerator, coefficient_recursion, power_series,
                         ConstantTrajectory, InterpolationTrajectory, temporal_derived_power_series, FlatString)
# noinspection PyUnresolvedReferences
from .registry import register_base, deregister_base, get_base, is_registered
# noinspection PyUnresolvedReferences
from .placeholder import (Scalars, ScalarTerm, IntegralTerm, FieldVariable, SpatialDerivedFieldVariable,
                          TemporalDerivedFieldVariable, ScalarFunction, Product, TestFunction, Input)
# noinspection PyUnresolvedReferences
from .simulation import *
# noinspection PyUnresolvedReferences
from .shapefunctions import *
# noinspection PyUnresolvedReferences
from .visualization import *

# noinspection PyUnresolvedReferences

__author__ = "Stefan Ecklebe, Marcus Riesmeier"
__email__ = "stefan.ecklebe@tu-dresden.de, marcus.riesmeier@umit.at"
__version__ = '0.4.0'
