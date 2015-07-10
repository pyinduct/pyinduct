from __future__ import division
import numpy as np
from scipy.optimize import fsolve
from core import Function, LagrangeFirstOrder

__author__ = 'stefan'

def cure_interval(test_function_class, interval, node_count=None, element_length=None):
    """
    Uses given test functions to cure a given interval with either node_count nodes or with
    elements of element_length
    :param interval:
    :param test_function_class:
    :return:
    """
    if not issubclass(test_function_class, Function):
        raise TypeError("test_function_class must be a SubClass of Function.")
    # TODO implement more
    if test_function_class is not LagrangeFirstOrder:
        raise TypeError("only LagrangeFirstOrder supported as test_function_class for now.")

    if not isinstance(interval, tuple):
        raise TypeError("interval must be given as tuple.")
    if len(interval) is not 2:
        raise TypeError("interval type not supported, should be (start, end)")

    if node_count and element_length:
        raise ValueError("node_count and element_length provided. Only one can be choosen.")
    if not node_count and not element_length:
        raise ValueError("neither (sensible) node_count nor element_length provided.")

    start = min(interval)
    end = max(interval)

    if node_count:
        nodes, element_length = np.linspace(start=start, stop=end, num=node_count, retstep=True)
    else:
        nodes = np.arange(start, end + element_length, element_length)
        node_count = nodes.shape[0]

    test_functions = [LagrangeFirstOrder(nodes[0], nodes[0], nodes[0] + element_length),
                      LagrangeFirstOrder(nodes[-1] - element_length, nodes[-1], nodes[-1])]
    for i in range(1, node_count-1):
        test_functions.insert(-1, LagrangeFirstOrder(nodes[i] - element_length,
                                                     nodes[i],
                                                     nodes[i] + element_length))

    return nodes, np.asarray(test_functions)

def find_roots(function, count, area=None, atol=1e-7, rtol=1e-1):
    """
    searches roots of the given function and checks them for uniqueness
    :param function:
    :param count:
    :return:
    """
    scale = 2
    count = int(count)
    # increase number to make sure that no root is forgotten
    own_count = scale*count

    if not callable(function):
        raise TypeError("callable handle is needed")

    if area is None:
        area = (0, 1e2)

    roots = []
    rroots = []
    errs = []
    values = np.arange(area[0], scale*area[1], rtol)
    val = iter(values)
    while len(roots) < own_count:
        try:
            root, info, ier, msg = fsolve(function, val.next(), full_output=True)
        except StopIteration:
            break

        if info['fvec'] > atol:
            continue
        if not (area[0] <= root <= area[1]):
            continue

        rroot = np.round(root, -int(np.log10(rtol)))
        if rroot in rroots:
            idx = rroots.index(rroot)
            if errs[idx] > info['fvec']:
                roots[idx] = root
                errs[idx] = info['fvec']
            continue

        roots.append(root)
        rroots.append(rroot)
        errs.append(info['fvec'])

    if len(roots) < count:
        raise ValueError("not enough roots could be detected. Increase Area.")

    return np.atleast_1d(sorted(roots)[:count]).flatten()
