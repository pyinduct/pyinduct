__author__ = 'stefan'

import numpy as np
from scipy.optimize import fsolve
from core import Function, LagrangeFirstOrder


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

def find_roots(function, count, area=None, tol=1e-1):
    """
    searches roots of the given function and checks them for uniqueness
    :param function:
    :param count:
    :return:
    """
    if not callable(function):
        raise TypeError("callable handle is needed")

    scale = 1e1
    max_area = 1e3

    if area is None:
        area = (0, 1e2)

    # compute a lot, just to be sure
    vec_func = np.vectorize(function)
    roots = fsolve(vec_func, np.arange(area[0], area[1]))

    # sort out entries that are close to each other
    unique_roots = np.array([root for root in np.unique(roots)])  # if root >= 0])
    temp_roots = []
    for root in unique_roots:
        if all(abs(root - rt) > tol for rt in temp_roots):
            temp_roots.append(root)
    clean_roots = np.array(temp_roots)

    print("found {0} unique root(s)".format(len(clean_roots)))
    if len(clean_roots) < count:
        # not enough, expand area to the right
        if area[1]*scale > max_area:
            raise ValueError("unable to find enough roots. found {0} requested: {1}".format(len(clean_roots), count))
        return find_roots(function, count, area=(area[0], scale*area[1]))

    return clean_roots[:count]
