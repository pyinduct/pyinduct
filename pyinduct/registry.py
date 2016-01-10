from __future__ import division
import numpy as np

_registry = {}


def is_registered(label):
    """
    checks whether a specific label has already been registered
    :param label: string, label to check for
    :return: True if registered, False if not
    """
    if not isinstance(label, str):
        raise TypeError("only strings allowed as labels!")

    return label in _registry.keys()


def register_base(label, functions, overwrite=False):
    """
    register a set of initial functions to make them accessible all over the pyinduct framework

    :param functions: array , list or single instance of ref:py:class:Function
    :param label: string that will be used as label
    :param overwrite: force overwrite if label is already present
    """
    if not isinstance(label, str):
        raise TypeError("only strings allowed as labels!")

    funcs = np.atleast_1d(functions)
    if is_registered(label):
        if overwrite:
            deregister_base(label)
        else:
            raise ValueError("Function set '{0}' already in registry!".format(label))

    derivatives = []
    n = 0
    while True:
        try:
            derivatives.append([func.derive(n) for func in funcs])
            n += 1
        except ValueError:
            break

    entry = np.array(derivatives)
    _registry[label] = entry


def deregister_base(label):
    """
    removes a set of initial functions from the packages registry
    :param label: string, label of functions that are to be removed
    :raises ValueError if label is not found in registry
    """
    if not isinstance(label, str):
        raise TypeError("Only strings allowed as label!")
    if not is_registered(label):
        raise ValueError("label {0} not found in registry!".format(label))

    del _registry[label]


def get_base(label, order):
    """
    retrieve registered set of initial functions by their label
    :param label: string, label of functions to retrieve
    :param order: deired derivative order of base
    :return: initial_functions
    """
    if is_registered(label):
        return _registry[label][order]
    else:
        raise ValueError("No functions registered for label: '{0}'!".format(label))
