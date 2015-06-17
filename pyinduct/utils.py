__author__ = 'stefan'

import numpy as np
from core import Function, LagrangeFirstOrder

def cure_interval(test_function_class, interval, node_count=None, element_length=None):
    """
    Uses given testfunctions to cure a given interval with either node_count nodes or with
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

    test_functions = []
    np.array((node_count,), dtype=object)
    test_functions.append(LagrangeFirstOrder(nodes[0], nodes[0], nodes[0] + element_length))
    test_functions.append(LagrangeFirstOrder(nodes[-1] - element_length, nodes[-1], nodes[-1]))
    for i in range(1, node_count-1):
        test_functions.insert(-1, LagrangeFirstOrder(nodes[i] - element_length,
                                                     nodes[i],
                                                     nodes[i] + element_length))

    return nodes, np.asarray(test_functions)
