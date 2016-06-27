"""
:py:mod:`pyinduct.registry` covers the interface for registration of bases (a base is a set of initial functions).
"""

import numpy as np

_registry = {}


def is_registered(label):
    """
    Checks whether a specific label has already been registered.

    Args:
    label (str): Label to check for.

    Return:
        bool: True if registered, False if not.
    """
    if not isinstance(label, (str, bytes)):
        raise TypeError("only strings allowed as labels!")

    return label in list(_registry.keys())


def register_base(label, functions, overwrite=False):
    """
    Register a set of initial functions to make them accessible all over the :py:mod:`pyinduct` framework.

    Args:
        functions: Array , list or single instance of :py:class:`pyinduct.core.Function`.
        label (str): String that will be used as label.
        overwrite: Force overwrite if label is already present.
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
    Removes a set of initial functions from the packages registry.

    Args:
        label (str): String, label of functions that are to be removed.

    Raises:
        ValueError: If label is not found in registry.
    """
    if not isinstance(label, (str, bytes)):
        raise TypeError("Only strings allowed as label!")
    if not is_registered(label):
        raise ValueError("label {0} not found in registry!".format(label))

    del _registry[label]


def get_base(label, order):
    """
    Retrieve registered set of initial functions by their label.

    Args:
        label (str): String, label of functions to retrieve.
        order (int): Desired derivative order of base.

    Return:
        initial_functions
    """
    if is_registered(label):
        base = _registry[label].get(order, None)
        if base is None:
            raise ValueError("base {} not available in order {}!".format(label, order))
        return base
    else:
        raise ValueError("no base registered under label '{0}'!".format(label))
