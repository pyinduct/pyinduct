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
        raise TypeError("Only strings allowed as labels!")

    return label in list(_registry.keys())


def register_base(label, base, overwrite=False):
    """
    Register a basis to make it accessible all over the :py:mod:`pyinduct` framework.

    Args:
        base (:py:class:`pyinduct.core.Base`): base to register
        label (str): String that will be used as label.
        overwrite: Force overwrite if a basis is already registered under this label.
    """
    if not isinstance(label, (str, bytes)):
        raise TypeError("Only strings allowed as labels!")

    new_base = _registry.get(label, None)

    if new_base is not None:
        if overwrite:
            deregister_base(label)
        else:
            raise ValueError("Function set '{0}' already in registry!".format(label))

    _registry[label] = base


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
        raise ValueError("Label '{0}' not found in registry!".format(label))

    del _registry[label]


def get_base(label):
    """
    Retrieve registered set of initial functions by their label.

    Args:
        label (str): String, label of functions to retrieve.

    Return:
        initial_functions
    """
    base = _registry.get(label, None)
    if base is None:
        raise ValueError("No base registered under label '{0}'!".format(label))
    else:
        return base
