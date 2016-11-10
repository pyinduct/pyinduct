"""
The shapefunctions module contains generic shapefunctions that can be used to approximate distributed systems without
giving  any information about the systems themselves. This is achieved by projecting them on generic, piecewise smooth
functions.
"""

import numpy as np
import numpy.polynomial.polynomial as npoly

from .core import Base, Function, Domain

__all__ = ["LagrangeFirstOrder", "LagrangeSecondOrder", "LagrangeNthOrder", "cure_interval"]


class LagrangeNthOrder(Function):
    """
    Lagrangian shape functions of order :math:`n`.

    Note:
        The polynomials between the boundary-polynomials and the peak-polynomials, respectively
        between peak-polynomials and peak-polynomials, are called mid-polynomials.

    Args:
        order (int): Order of the lagrangian polynomials.
        nodes (numpy.array): Nodes on which the piecewise defined functions have to be one/zero.
            Length of nodes must be either :math:`order * 2 + 1` (for peak-polynomials, see notes)
            or 'order +1' (for boundary- and mid-polynomials).
        left (bool): State the first node (*nodes[0]*) to be the left boundary of the considered domain.
        right (bool): State the last node (*nodes[-1]*) to be the right boundary of the considered domain.
        mid_num (int):  Local number of mid-polynomials (see notes) to use (only  used for *order* >= 2) .
            :math:`\\text{mid\\_num} \\in \\{ 1, ..., \\text{order} - 1 \\}`
        boundary (str): provide "left" or "right" to instantiate the according boundary-polynomial.
    """

    def __init__(self, order, nodes, left=False, right=False, mid_num=None, boundary=None):

        if order <= 0:
            raise ValueError("Order must be greater 0.")

        if not all(nodes == sorted(nodes)):
            raise ValueError("Nodes must be sorted.")

        if not all([isinstance(item, bool) for item in (left, right)]):
            raise TypeError("Arguments left and right must be from type bool.")

        if mid_num is not None and (mid_num <= 0 or mid_num >= order):
            raise ValueError("There are no elements of this kind at this position (mid_num).")

        if boundary is not None and boundary not in ("left", "right"):
            raise ValueError("Kwarg 'boundary' can only set to 'left' or 'right'")

        if order * 2 + 1 == len(nodes):
            is_peak_element = True
        elif order + 1 == len(nodes):
            is_peak_element = False
        else:
            raise ValueError("Length of nodes must be either 'order * 2 + 1' or 'order +1'.")

        if (left and not is_peak_element and mid_num is None and boundary is None) or boundary == "left":
            poly = self._poly_factory(nodes[1:], nodes[0])
        elif (right and not is_peak_element and mid_num is None) or boundary == "right":
            poly = self._poly_factory(nodes[:-1], nodes[-1])
        elif mid_num:
            poly = self._poly_factory(np.hstack((nodes[:mid_num], nodes[mid_num + 1:])), nodes[mid_num])
        elif is_peak_element:
            left_poly = self._poly_factory(nodes[:order], nodes[order])
            right_poly = self._poly_factory(nodes[order + 1:], nodes[order])
        else:
            raise NotImplementedError

        if is_peak_element:
            funcs = [self._func_factory(d_ord, order, nodes, left, right, l_poly=left_poly, r_poly=right_poly)
                     for d_ord in range(order + 1)]
        else:
            funcs = [self._func_factory(d_ord, order, nodes, left, right, poly=poly) for d_ord in range(order + 1)]

        Function.__init__(self, funcs[0], nonzero=(nodes[0], nodes[-1]), derivative_handles=funcs[1:])

    def _poly_factory(self, zero_nodes, one_node):
        poly = npoly.Polynomial(npoly.polyfromroots(zero_nodes))
        return npoly.Polynomial(poly.coef / poly(one_node))

    @staticmethod
    def _func_factory(der_order, order, nodes, left, right, poly=None, r_poly=None, l_poly=None):

        if poly:
            if der_order == 0 or (left and right):
                cond_list = lambda z: [np.bitwise_and(nodes[0] <= z, z <= nodes[-1]).flatten()]
                func_list = [poly.deriv(der_order)]
            else:
                if left:
                    cond_list = lambda z: [np.bitwise_and(nodes[0] <= z, z < nodes[-1]).flatten(),
                                           np.array(z == nodes[-1]).flatten()]
                elif right:
                    cond_list = lambda z: [np.bitwise_and(nodes[0] < z, z <= nodes[-1]).flatten(),
                                           np.array(z == nodes[0]).flatten()]
                else:
                    cond_list = lambda z: [np.bitwise_and(nodes[0] < z, z < nodes[-1]).flatten(),
                                           np.bitwise_or(nodes[0] == z, z == nodes[-1]).flatten()]
                func_list = [poly.deriv(der_order), .5 * poly.deriv(der_order)]

        else:
            if der_order == 0:
                cond_list = lambda z: [np.bitwise_and(nodes[0] <= z, z <= nodes[order]).flatten(),
                                       np.bitwise_and(nodes[order] < z, z <= nodes[-1]).flatten()]
                func_list = [l_poly, r_poly]

            else:
                def weighted_comb(z):
                    return .5 * (l_poly.deriv(der_order)(z) + r_poly.deriv(der_order)(z))

                if left and right:
                    cond_list = lambda z: [np.bitwise_and(nodes[0] <= z, z < nodes[order]).flatten(),
                                           np.array(nodes[order] == z).flatten(),
                                           np.bitwise_and(nodes[order] < z, z <= nodes[-1]).flatten()]
                    func_list = [l_poly.deriv(der_order), weighted_comb, r_poly.deriv(der_order)]

                elif left:
                    cond_list = lambda z: [np.bitwise_and(nodes[0] <= z, z < nodes[order]).flatten(),
                                           np.array(nodes[order] == z).flatten(),
                                           np.bitwise_and(nodes[order] < z, z < nodes[-1]).flatten(),
                                           np.array(z == nodes[-1]).flatten()]
                    func_list = [l_poly.deriv(der_order), weighted_comb, r_poly.deriv(der_order),
                                 .5 * r_poly.deriv(der_order)]

                elif right:
                    cond_list = lambda z: [np.bitwise_and(nodes[0] < z, z < nodes[order]).flatten(),
                                           np.array(nodes[order] == z).flatten(),
                                           np.bitwise_and(nodes[order] < z, z <= nodes[-1]).flatten(),
                                           np.array(z == nodes[0]).flatten()]
                    func_list = [l_poly.deriv(der_order), weighted_comb, r_poly.deriv(der_order),
                                 .5 * l_poly.deriv(der_order)]

                else:
                    cond_list = lambda z: [np.bitwise_and(nodes[0] < z, z < nodes[order]).flatten(),
                                           np.array(nodes[order] == z).flatten(),
                                           np.bitwise_and(nodes[order] < z, z < nodes[-1]).flatten(),
                                           np.array(z == nodes[0]).flatten(),
                                           np.array(z == nodes[-1]).flatten()]
                    func_list = [l_poly.deriv(der_order), weighted_comb, r_poly.deriv(der_order),
                                 .5 * l_poly.deriv(der_order), .5 * r_poly.deriv(der_order)]

        def function(zz):
            z = np.array(zz, dtype=np.float_)
            res = np.piecewise(z, cond_list(z), func_list)
            if np.ndim(zz) == 0:
                return np.float_(res)
            else:
                return res.flatten()

        return function

    @staticmethod
    def cure_hint(domain, **kwargs):
        """
        Hint function that will cure the given interval with :py:class:`LagrangeNthOrder`.
        Length of the domain argument :math:`L` must satisfy the condition

        .. math:: L = 1 + (1 + n) order \\quad \\forall n \\in \\mathbb N.

        E.g. \n
        - order = 1 -> :math:`L \\in \\{2, 3, 4, 5, ...\\}`
        - order = 2 -> :math:`L \\in \\{3, 5, 7, 9, ...\\}`
        - order = 3 -> :math:`L \\in \\{4, 7, 10, 13, ...\\}`
        - and so on.

        Args:
            domain (:py:class:`core.Domain`): Domain to be cured.
            order (int): Order of the lagrange polynomials.

        Return:
            tupel: (domain, funcs), where funcs is a set of :py:class:`LagrangeNthOrder` shapefunctions.
        """
        order = kwargs["order"]
        nodes = np.array(domain)
        possible_node_lengths = np.array([(order + 1) + n * order for n in range(len(nodes))], dtype=int)
        if not len(nodes) in possible_node_lengths:
            suggested_indices = np.where(np.isclose(possible_node_lengths, len(nodes), atol=order - 1))[0]
            alternative_node_lengths = possible_node_lengths[suggested_indices]
            raise ValueError("See LagrangeNthOrder.cure_hint docstring.\n"
                             "\tThere are some restrictions to the length of nodes/domain.\n"
                             "\tYour desired (invalid) node count is {}.\n"
                             "\tSuggested valid node count(s): {}.".format(len(nodes), alternative_node_lengths))

        funcs = np.empty((len(nodes),), dtype=LagrangeNthOrder)
        no_peaks = True

        for index in range(1, len(domain) - 1):
            if index % order == 0:
                no_peaks = False
                left = True if index == order else False
                right = True if len(nodes) - 1 - index == order else False
                funcs[index] = LagrangeNthOrder(order, nodes[index - order:index + order + 1], left=left, right=right)
            else:
                mid_num = index % order
                left = True if index == mid_num else False
                right = True if nodes[index] in nodes[-1 - order:-1] else False
                funcs[index] = LagrangeNthOrder(order, nodes[index - mid_num: index - mid_num + order + 1],
                                                mid_num=mid_num, left=left, right=right)

        funcs[0] = LagrangeNthOrder(order, nodes[: order + 1], left=True, right=no_peaks, boundary="left")
        funcs[-1] = LagrangeNthOrder(order, nodes[-(order + 1):], left=no_peaks, right=True, boundary="right")

        return domain, funcs


class LagrangeFirstOrder(Function):
    """
    Lagrangian shape functions of order 1.

    Args:
        start: start node
        top: top node, where :math:`f(x) = 1`
        end: end node

    Keyword Args:
        half:
        right_border:
        left_border:
    """

    def __init__(self, start, top, end, **kwargs):
        if not start <= top <= end or start == end:
            raise ValueError("Input data is nonsense, see Definition.")

        if kwargs.get("half", None) is None:
            args1 = kwargs.copy()
            args1.update({"right_border": False})
            rise_fncs = self._function_factory(start, start, top, **args1)
            args2 = kwargs.copy()
            args2.update({"left_border": False})
            fall_fncs = self._function_factory(top, end, end, **args2)

            def _lag1st_factory(der):
                def _lag1st_complete(z):
                    if z == top:
                        return .5 * (rise_fncs[der](z) + fall_fncs[der](z))
                    else:
                        return rise_fncs[der](z) + fall_fncs[der](z)

                return _lag1st_complete

            funcs = [_lag1st_factory(derivative) for derivative in [0, 1]]
        else:
            funcs = self._function_factory(start, top, end, **kwargs)

        Function.__init__(self, funcs[0], nonzero=(start, end), derivative_handles=funcs[1:])

    @staticmethod
    def _function_factory(start, mid, end, **kwargs):

        if start == mid:
            m = -1 / (start - end)
            n = -m * start
        elif mid == end:
            m = 1 / (start - end)
            n = 1 - m * start
        else:
            raise ValueError

        def _lag1st_half(z):
            if start <= z <= end:
                return m * z + n
            else:
                return 0

        def _lag1st_half_dz(z):
            if z == start and not kwargs.get("left_border", False) or \
                        z == end and not kwargs.get("right_border", False):
                return .5 * m
            if start <= z <= end:
                return m
            else:
                return 0

        return [_lag1st_half, _lag1st_half_dz]

    @staticmethod
    def cure_hint(domain, **kwargs):
        """
        Hint function that will cure the given interval with LagrangeFirstOrder.

        Args:
            domain (:py:class:`core.Domain`): domain to be cured

        Return:
            tuple: (domain, funcs), where funcs is set of :py:class:`LagrangeFirstOrder` shapefunctions.
        """
        funcs = np.empty((len(domain),), dtype=LagrangeFirstOrder)
        funcs[0] = LagrangeFirstOrder(domain[0], domain[1], domain[1], half="left", left_border=True,
                                      right_border=True if len(domain) == 2 else False)
        funcs[-1] = LagrangeFirstOrder(domain[-2], domain[-2], domain[-1], half="right", right_border=True,
                                       left_border=True if len(domain) == 2 else False)

        for idx in range(1, len(domain) - 1):
            funcs[idx] = LagrangeFirstOrder(domain[idx - 1], domain[idx], domain[idx + 1],
                                            left_border=True if idx == 1 else False,
                                            right_border=True if idx == len(domain) - 2 else False)
        return domain, funcs


class LagrangeSecondOrder(Function):
    """
    Lagrangian shape functions of order 2.

    Args:
        start: start node
        mid: middle node, where :math:`f(x) = 1`
        end: end node

    Keyword Args:
        curvature (str): "concave" or "convex"
        half (str): Generate only "left" or "right" half.
    """

    def __init__(self, start, mid, end, **kwargs):
        assert (start <= mid <= end)
        if kwargs["curvature"] == "concave" and "half" not in kwargs:
            # interior special case
            args1 = kwargs.copy()
            args1.update({"right_border": False, "half": "right"})
            func1 = self._function_factory(start, start + (mid - start) / 2, mid, **args1)
            args2 = kwargs.copy()
            args2.update({"left_border": False, "half": "left"})
            func2 = self._function_factory(mid, mid + (end - mid) / 2, end, **args2)

            def composed_func(z):
                if start <= z <= mid:
                    return func1[0](z)
                elif mid < z <= end:
                    return func2[0](z)
                else:
                    return 0

            def composed_func_dz(z):
                if z == mid:
                    return 0
                elif start <= z < mid:
                    return func1[1](z)
                elif mid < z <= end:
                    return func2[1](z)
                else:
                    return 0

            def composed_func_ddz(z):
                if start <= z < mid:
                    return func1[2](z)
                elif z == mid:
                    return func1[2](z) + func2[2](z)
                elif mid <= z <= end:
                    return func2[2](z)
                else:
                    return 0

            funcs = [composed_func, composed_func_dz, composed_func_ddz]
        else:
            funcs = self._function_factory(start, mid, end, **kwargs)

        Function.__init__(self, funcs[0], nonzero=(start, end), derivative_handles=funcs[1:])

    @staticmethod
    def _function_factory(start, mid, end, **kwargs):
        if kwargs["curvature"] == "convex":
            p = -(start + end)
            q = start * end
            s = 1 / (mid ** 2 + p * mid + q)

        elif kwargs["curvature"] == "concave":
            if kwargs["half"] == "left":
                p = -(mid + end)
                q = mid * end
                s = 1 / (start ** 2 + p * start + q)
            elif kwargs["half"] == "right":
                p = -(start + mid)
                q = start * mid
                s = 1 / (end ** 2 + p * end + q)
        else:
            raise ValueError

        def lag2nd(z):
            if start <= z <= end:
                return s * (z ** 2 + p * z + q)
            else:
                return 0

        def lag2nd_dz(z):
            if z == start and not kwargs.get("left_border", False) or \
                        z == end and not kwargs.get("right_border", False):
                return .5 * s * (2 * z + p)
            if start <= z <= end:
                return s * (2 * z + p)
            else:
                return 0

        def lag2nd_ddz(z):
            # if z == start or z == end:
            if z == start and not kwargs.get("left_border", False) or \
                        z == end and not kwargs.get("right_border", False):
                return s
            if start <= z <= end:
                return s * 2
            else:
                return 0

        return [lag2nd, lag2nd_dz, lag2nd_ddz]

    @staticmethod
    def cure_hint(domain, **kwargs):
        """
        Hint function that will cure the given interval with LagrangeSecondOrder.

        Args:
            domain (:py:class:`core.Domain`): domain to be cured

        Return:
            tuple: (domain, funcs), where funcs is set of :py:class:`LagrangeSecondOrder` shapefunctions.
        """
        if len(domain) < 3 or len(domain) % 2 != 1:
            raise ValueError("node count has to be at least 3 and can only be odd for Lag2nd!")

        funcs = np.empty((len(domain),), dtype=LagrangeSecondOrder)

        # boundary special cases
        funcs[0] = LagrangeSecondOrder(domain[0], domain[1], domain[2], curvature="concave", half="left",
                                       left_border=True)
        funcs[-1] = LagrangeSecondOrder(domain[-3], domain[-2], domain[-1], curvature="concave", half="right",
                                        right_border=True)

        # interior
        for idx in range(1, len(domain) - 1):
            if idx % 2 != 0:
                funcs[idx] = LagrangeSecondOrder(domain[idx - 1], domain[idx], domain[idx + 1], curvature="convex",
                                                 left_border=True if idx == 1 else False,
                                                 right_border=True if idx == len(domain) - 2 else False, )
            else:
                funcs[idx] = LagrangeSecondOrder(domain[idx - 2], domain[idx], domain[idx + 2], curvature="concave",
                                                 left_border=True if idx == 2 else False,
                                                 right_border=True if idx == len(domain) - 3 else False, )

        return domain, funcs


def cure_interval(shapefunction_class, interval, node_count=None, node_distance=None, **kwargs):
    """
    Use shape functions to cure an interval with either *node_count* nodes or nodes with *node_distance*.

    Args:
        shapefunction_class: Class to cure the interval (e.g. :py:class:`LagrangeFirstOrder`). The given class
            has to provide a :py:func:`cure_hint`.
        interval (tuple): Limits that constrain the interval.
        node_count (int): Amount of nodes to use.
        node_distance (numbers.Number): Distance of nodes.

    Note:
        Either *node_count* or *node_node_distance* can be specified. If both are given and are not consistent,
        an exception will be raised by :py:class:`pyinduct.simulation.Domain` .

    Raises:
        TypeError: If given class does not provide a static :py:func:`cure_hint` method.

    Return:
        tuple:
            :code:`(domain, funcs)`: Where :code:`domain` is a :py:class:`pyinduct.simulation.Domain` instance
            and :code:`funcs` is a list of (e.g. :py:class:`LagrangeFirstOrder`) shapefunctions.
    """
    domain = Domain(bounds=interval, step=node_distance, num=node_count)

    try:
        nodes, fractions = shapefunction_class.cure_hint(domain, **kwargs)
    except AttributeError:
        raise TypeError("given function class {} offers no cure_hint!".format(shapefunction_class))

    return nodes, Base(fractions)
