"""
:mod:`pyinduct.registry` covers the interface for registration of bases (a base is a set of initial functions).
"""

import numpy as np

_registry = {}


def is_registered(label):
    """
    checks whether a specific label has already been registered
    :param label: string, label to check for
    :return: True if registered, False if not
    """
    if not isinstance(label, (str, bytes)):
        raise TypeError("only strings allowed as labels!")

    return label in list(_registry.keys())


def register_base(label, functions, overwrite=False):
    """
    register a set of initial functions to make them accessible all over the pyinduct framework

    :param functions: array , list or single instance of ref:py:class:Function
    :param label: string that will be used as label
    :param overwrite: force overwrite if label is already present
    """
    if not isinstance(label, (str, bytes)):
        raise TypeError("only strings allowed as labels!")

    funcs = np.atleast_1d(functions)
    derivatives = _registry.get(label, {})

    if derivatives:
        if overwrite:
            deregister_base(label)
        else:
            raise ValueError("Function set '{0}' already in registry!".format(label))

    n = 0
    while True:
        try:
            derivatives[n] = np.array([func.derive(n) for func in funcs])
            n += 1
        except ValueError:
            break

    _registry[label] = derivatives


def deregister_base(label):
    """
    removes a set of initial functions from the packages registry
    :param label: string, label of functions that are to be removed
    :raises ValueError if label is not found in registry
    """
    if not isinstance(label, (str, bytes)):
        raise TypeError("Only strings allowed as label!")
    if not is_registered(label):
        raise ValueError("label {0} not found in registry!".format(label))

    del _registry[label]


def get_base(label, order):
    """
    retrieve registered set of initial functions by their label
    :param label: string, label of functions to retrieve
    :param order: desired derivative order of base
    :return: initial_functions
    """
    if is_registered(label):
        base = _registry[label].get(order, None)
        if base is None:
            raise ValueError("base {} not available in order {}!".format(label, order))
        return base
    else:
        raise ValueError("no base registered under label '{0}'!".format(label))
