# -*- coding: utf-8 -*-
import os
import matplotlib as mpl

# make everybody use the same qt version, try Qt5 first
try:
    __import__("PyQt5")
    os.environ["PYQTGRAPH_QT_LIB"] = "PyQt5"
    mpl.use("Qt5Agg")
except ImportError:
    os.environ["PYQTGRAPH_QT_LIB"] = "PyQt4"
    mpl.use("Qt4Agg")

from .core import *
from .feedback import *
from .eigenfunctions import *
from .trajectory import *
from .registry import *
from .placeholder import *
from .simulation import *
from .shapefunctions import *
from .visualization import *


__author__ = "Stefan Ecklebe, Marcus Riesmeier"
__email__ = "stefan.ecklebe@tu-dresden.de, marcus.riesmeier@umit.at"
__version__ = '0.5.0'
