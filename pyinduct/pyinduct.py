from __future__ import division
import numpy as np


__author__ = 'stefan'


_registry = {}


def register_initial_functions(label, functions, overwrite=False):
    """
    register a set of initial functions to make them accessible all over the pyinduct framework

    :param functions: array , list or single instance of ref:py:class:Function
    :param label: string that will be used as label
    """
    funcs = np.atleast_1d(functions)
    if label in _registry.keys():
        if overwrite:
            deregister_initial_functions(label)
        else:
            raise ValueError("Function set {0} already in registry!".format(label))

    derivatives = []
    n = 0
    while True:
        try:
            derivatives.append([func.derive(n) for func in functions])
            n += 1
        except ValueError:
            break

    entry = np.array(derivatives)
    _registry[label] = entry


def deregister_initial_functions(label):
    """
    removes a set of initial functions from the packages registry
    :param label: string, label of functions that are to be removed
    :raises ValueError if label is not found in registry
    """
    del _registry[label]


def get_initial_functions(label, order):
    """
    retrieve registered set of initial functions by their label
    :param label: string, label of functions to retrieve
    :return: initial_functions
    """
    if label in _registry.keys():
        return _registry[label][order]
    else:
        raise ValueError("No functions registered for label {0}!".format(label))
