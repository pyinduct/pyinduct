import numpy as np

from .core import Function
from .simulation import Domain

"""
This module contains all shape functions that come with PyInduct. Furthermore helper methods
for curing can be found here.
"""


class LagrangeFirstOrder(Function):
    """
    Lagrangian shape functions of order 1

    :param start: start node
    :param top: top node, where :math:`f(x) = 1`
    :param start: end node
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
                        return .5*(rise_fncs[der](z) + fall_fncs[der](z))
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
            n = -m*start
        elif mid == end:
            m = 1 / (start - end)
            n = 1 - m*start
        else:
            raise ValueError

        def _lag1st_half(z):
            if start <= z <= end:
                return m*z + n
            else:
                return 0

        def _lag1st_half_dz(z):
            if z == start and not kwargs.get("left_border", False) or \
                        z == end and not kwargs.get("right_border", False):
                return .5*m
            if start <= z <= end:
                return m
            else:
                return 0

        return [_lag1st_half, _lag1st_half_dz]

    @staticmethod
    def cure_hint(domain):
        """
        hint function that will cure the given interval with this function type
        :param domain: domain to be cured
        :type domain: py:class:pyinduct.Domain
        :return: set of shapefunctions
        """
        funcs = np.empty((len(domain),), dtype=LagrangeFirstOrder)
        funcs[0] = LagrangeFirstOrder(domain[0], domain[1], domain[1], half="left", left_border=True,
                                      right_border=True if len(domain) == 2 else False)
        funcs[-1] = LagrangeFirstOrder(domain[-2], domain[-2], domain[-1], half="right", right_border=True,
                                       left_border=True if len(domain) == 2 else False)

        for idx in range(1, len(domain)-1):
            funcs[idx] = LagrangeFirstOrder(domain[idx-1],
                                            domain[idx],
                                            domain[idx+1],
                                            left_border=True if idx == 1 else False,
                                            right_border=True if idx == len(domain)-2 else False)
        return domain, funcs


class LagrangeSecondOrder(Function):
    # TODO generate svg of 2nd of Lag2nd and remove ascii art from docstring
    """
    Implementation of an lagrangian initial function of order 2::

      ^                                    _
    1-|           ^                      / | \
      |          /|\                   /   |   \
      |         / | \                 /    |    \
      |        /  |  \               /     |     \
    0-|--\----/   |   \----/--------/------|----- \---> z
          \_/     |    \_/
       start    top       end     start   top    end
         |<----- d ------>|        |<---- d/2 --->|


    :param start: start node
    :param mid: middle node, where :math:`f(x) = 1`
    :param end: end node
    :param curvature: concave or convex
    :param half: generate only left or right haf
    """
    def __init__(self, start, mid, end, **kwargs):
        assert(start <= mid <= end)
        if kwargs["curvature"] == "concave" and "half" not in kwargs:
            # interior special case
            args1 = kwargs.copy()
            args1.update({"right_border": False, "half": "right"})
            func1 = self._function_factory(start, start + (mid-start)/2, mid, **args1)
            args2 = kwargs.copy()
            args2.update({"left_border": False, "half": "left"})
            func2 = self._function_factory(mid, mid + (end-mid)/2, end, **args2)

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

            funcs = (composed_func, composed_func_dz, composed_func_ddz)
        else:
            funcs = self._function_factory(start, mid, end, **kwargs)

        Function.__init__(self, funcs[0],
                          nonzero=(start, end),
                          derivative_handles=funcs[1:])

    @staticmethod
    def _function_factory(start, mid, end, **kwargs):
        if kwargs["curvature"] == "convex":
            p = -(start+end)
            q = start*end
            s = 1/(mid**2 + p*mid + q)

        elif kwargs["curvature"] == "concave":
            if kwargs["half"] == "left":
                p = -(mid+end)
                q = mid*end
                s = 1/(start**2 + p*start + q)
            elif kwargs["half"] == "right":
                p = -(start+mid)
                q = start*mid
                s = 1/(end**2 + p*end + q)
        else:
            raise ValueError

        def lag2nd(z):
            if start <= z <= end:
                return s*(z**2 + p*z + q)
            else:
                return 0

        def lag2nd_dz(z):
            if z == start and not kwargs.get("left_border", False) or \
                        z == end and not kwargs.get("right_border", False):
                return .5*s*(2*z + p)
            if start <= z <= end:
                return s*(2*z + p)
            else:
                return 0

        def lag2nd_ddz(z):
            # if z == start or z == end:
            if z == start and not kwargs.get("left_border", False) or \
                        z == end and not kwargs.get("right_border", False):
                return s
            if start <= z <= end:
                return s*2
            else:
                return 0

        return lag2nd, lag2nd_dz, lag2nd_ddz

    @staticmethod
    def cure_hint(domain):
        """
        cure hint for Lag2nd
        :param domain:
        :return:
        """
        if len(domain) < 3 or len(domain) % 2 != 1:
            raise ValueError("node count has to be at least 3 and can only be odd for Lag2nd!")

        funcs = np.empty((len(domain),), dtype=LagrangeSecondOrder)

        # boundary special cases
        funcs[0] = LagrangeSecondOrder(domain[0], domain[1], domain[2],
                                       curvature="concave", half="left", left_border=True)
        funcs[-1] = LagrangeSecondOrder(domain[-3], domain[-2], domain[-1],
                                        curvature="concave", half="right", right_border=True)

        # interior
        for idx in range(1, len(domain)-1):
            if idx % 2 != 0:
                funcs[idx] = LagrangeSecondOrder(domain[idx-1], domain[idx], domain[idx+1], curvature="convex",
                                                 left_border=True if idx == 1 else False,
                                                 right_border=True if idx == len(domain)-2 else False,
                                                 )
            else:
                funcs[idx] = LagrangeSecondOrder(domain[idx-2], domain[idx], domain[idx+2], curvature="concave",
                                                 left_border=True if idx == 2 else False,
                                                 right_border=True if idx == len(domain)-3 else False,
                                                 )

        return domain, funcs

    '''
    def __init__(self, start, top, end, max_element_length):
        self._element_length = end - start
        if not start <= top <= end or start == end or (
            not np.isclose(self._element_length, max_element_length) and not np.isclose(self._element_length,
                                                                                        max_element_length / 2)):
            raise ValueError("Input data is nonsense, see Definition.")

        self._start = start
        self.top = top
        self._end = end
        self._e_2 = max_element_length / 4

        if start == top:
            self._gen_left_top_poly()
            Function.__init__(self, self._lagrange2nd_border_left, nonzero=(start, end),
                              derivative_handles=[self._der_lagrange2nd_border_left,
                                                  self._dder_lagrange2nd_border_left])
        elif top == end:
            self._gen_right_top_poly()
            Function.__init__(self, self._lagrange2nd_border_right, nonzero=(start, end),
                              derivative_handles=[self._der_lagrange2nd_border_right,
                                                  self._dder_lagrange2nd_border_right])
        elif np.isclose(end - start, max_element_length):
            self._gen_left_top_poly()
            self._gen_right_top_poly()
            Function.__init__(self, self._lagrange2nd_interior, nonzero=(start, end),
                              derivative_handles=[self._der_lagrange2nd_interior,
                                                  self._dder_lagrange2nd_interior])
        elif np.isclose(end - start, max_element_length / 2):
            self._gen_mid_top_poly()
            Function.__init__(self, self._lagrange2nd_interior_half, nonzero=(start, end),
                              derivative_handles=[self._der_lagrange2nd_interior_half,
                                                  self._dder_lagrange2nd_interior_half])
        else:
            raise ValueError("Following arguments do not meet the specs from LagrangeSecondOrder: start, end")



    def _gen_left_top_poly(self):
        left_top_poly = npoly.Polynomial(npoly.polyfromroots((self._e_2, 2 * self._e_2)))
        self._left_top_poly = npoly.Polynomial(left_top_poly.coef / left_top_poly(0))

    def _gen_right_top_poly(self):
        right_top_poly = npoly.Polynomial(npoly.polyfromroots((0, self._e_2)))
        self._right_top_poly = npoly.Polynomial(right_top_poly.coef / right_top_poly(2 * self._e_2))

    def _gen_mid_top_poly(self):
        mid_top_poly = npoly.Polynomial(npoly.polyfromroots((0, 2 * self._e_2)))
        self._mid_top_poly = npoly.Polynomial(mid_top_poly.coef / mid_top_poly(self._e_2))

    def _lagrange2nd_border_left(self, z, der_order=0):
        """
        left border equation for lagrange 2nd order
        """
        if z < self.top or z > self._end:
            return 0
        else:
            return self._left_top_poly.deriv(der_order)(z)

    def _lagrange2nd_border_right(self, z, der_order=0):
        """
        right border equation for lagrange 2nd order
        """
        if z < self._start or z > self._end:
            return 0
        else:
            return self._right_top_poly.deriv(der_order)(z - self._start)

    def _lagrange2nd_interior(self, z, der_order=0):
        """
        wide (d) interior equations for lagrange 2nd order
        """
        if z < self._start or z > self._end:
            return 0
        elif z == self.top and der_order > 0:
            return 0
        elif self._start <= z <= self.top:
            return self._right_top_poly.deriv(der_order)(z - self._start)
        else:
            return self._left_top_poly.deriv(der_order)(z - self.top)

    def _lagrange2nd_interior_half(self, z, der_order=0):
        """
        small (d/2) interior equations for lagrange 2nd order
        """
        if z < self._start or z > self._end:
            return 0
        else:
            return self._mid_top_poly.deriv(der_order)(z - self._start)

    def _der_lagrange2nd_border_left(self, z):
        return self._lagrange2nd_border_left(z, der_order=1)

    def _der_lagrange2nd_border_right(self, z):
        return self._lagrange2nd_border_right(z, der_order=1)

    def _der_lagrange2nd_interior(self, z):
        return self._lagrange2nd_interior(z, der_order=1)

    def _der_lagrange2nd_interior_half(self, z):
        return self._lagrange2nd_interior_half(z, der_order=1)

    def _dder_lagrange2nd_border_left(self, z):
        return self._lagrange2nd_border_left(z, der_order=2)

    def _dder_lagrange2nd_border_right(self, z):
        return self._lagrange2nd_border_right(z, der_order=2)

    def _dder_lagrange2nd_interior(self, z):
        return self._lagrange2nd_interior(z, der_order=2)

    def _dder_lagrange2nd_interior_half(self, z):
        return self._lagrange2nd_interior_half(z, der_order=2)
    '''


def cure_interval(shapefunction_class, interval, node_count=None, node_distance=None):
    """
    Use test functions to cure an interval with either node_count nodes or nodes with node_node_distance.

    :param shapefunction_class: class to cure the interval (e.g. py:LagrangeFirstOrder)
    :param interval: tuple of limits that constrain the interval
    :param node_count: amount of nodes to use
    :param node_distance: distance of nodes

    :return: tuple of nodes and functions
    """
    if not issubclass(shapefunction_class, Function):
        raise TypeError("test_function_class must be a SubClass of Function.")

    # TODO move these into "cure_hint" method of core.Function
    if shapefunction_class not in {LagrangeFirstOrder, LagrangeSecondOrder}:
        raise TypeError("LagrangeFirstOrder and LagrangeSecondOrder supported as test_function_class for now.")

    domain = Domain(bounds=interval, step=node_distance, num=node_count)

    if not hasattr(shapefunction_class, "cure_hint"):
        raise TypeError("given function class {} offers no cure_hint!".format(shapefunction_class))

    return shapefunction_class.cure_hint(domain)
