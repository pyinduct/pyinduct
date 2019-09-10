# -*- coding: utf-8 -*-
import os
import numpy as np
import matplotlib as mpl

# make everybody use the same qt version, try Qt5 first
try:
    __import__("PyQt5")
    os.environ["PYQTGRAPH_QT_LIB"] = "PyQt5"
    mpl.use("Qt5Agg")
except ImportError:
    os.environ["PYQTGRAPH_QT_LIB"] = "PyQt4"
    mpl.use("Qt4Agg")

# since this is a serious toolbox
np.seterr(all="raise")

# noinspection PyUnresolvedReferences
from .core import *
# noinspection PyUnresolvedReferences
from .feedback import *
# noinspection PyUnresolvedReferences
from .eigenfunctions import *
# noinspection PyUnresolvedReferences
from .trajectory import *
# noinspection PyUnresolvedReferences
from .registry import *
# noinspection PyUnresolvedReferences
from .placeholder import *
# noinspection PyUnresolvedReferences
from .simulation import *
# noinspection PyUnresolvedReferences
from .shapefunctions import *
# noinspection PyUnresolvedReferences
from .visualization import *
# noinspection PyUnresolvedReferences
from .examples import *

# noinspection PyUnresolvedReferences

__author__ = "Stefan Ecklebe, Marcus Riesmeier"
__email__ = "stefan.ecklebe@tu-dresden.de, marcus.riesmeier@umit.at"
__version__ = '0.5.0'
