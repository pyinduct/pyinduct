import numpy as np
from numpy.polynomial import polynomial as npoly

from .core import Function
from .simulation import Domain
"""
This module contains all shape functions that come with PyInduct. Furthermore helper methods
for curing can be found here.
"""


class LagrangeFirstOrder(Function):
    """
    Implementation of a lagrangian initial function of order 1::

      ^
    1-|         ^
      |        /|\
      |       / | \
      |      /  |  \
    0-|-----/   |   \-------------> z
            |   |   |

    start,top,end

    :param start: start node
    :param top: top node, where :math:`f(x) = 1`
    :param start: end node
    """

    def __init__(self, start, top, end):
        if not start <= top <= end or start == end:
            raise ValueError("Input data is nonsense, see Definition.")

        self._start = start
        self.top = top
        self._end = end

        # speed
        self._a = self.top - self._start
        self._b = self._end - self.top

        if start == top:
            Function.__init__(self, self._lagrange1st_border_left,
                              nonzero=(start, end), derivative_handles=[self._der_lagrange1st_border_left])
        elif top == end:
            Function.__init__(self, self._lagrange1st_border_right,
                              nonzero=(start, end), derivative_handles=[self._der_lagrange1st_border_right])
        else:
            Function.__init__(self, self._lagrange1st_interior,
                              nonzero=(start, end), derivative_handles=[self._der_lagrange1st_interior])

    def _lagrange1st_border_left(self, z):
        """
        left border equation for lagrange 1st order
        """
        if z < self.top or z >= self._end:
            return 0
        else:
            return (self.top - z) / self._b + 1

    def _lagrange1st_border_right(self, z):
        """
        right border equation for lagrange 1st order
        """
        if z <= self._start or z > self._end:
            return 0
        else:
            return (z - self._start) / self._a

    def _lagrange1st_interior(self, z):
        """
        interior equations for lagrange 1st order
        """
        if z < self._start or z > self._end:
            return 0
        elif self._start <= z <= self.top:
            return (z - self._start) / self._a
        else:
            return (self.top - z) / self._b + 1

    def _der_lagrange1st_border_left(self, z):
        """
        left border equation for lagrange 1st order
        """
        if z < self.top or z >= self._end:
            return 0
        else:
            return -1 / self._b

    def _der_lagrange1st_border_right(self, z):
        """
        right border equation for lagrange 1st order
        """
        if z <= self._start or z > self._end:
            return 0
        else:
            return 1 / self._a

    def _der_lagrange1st_interior(self, z):
        """
        interior equations for lagrange 1st order
        """
        if z < self._start or z > self._end or z == self.top:
            return 0
        elif self._start <= z < self.top:
            return 1 / self._a
        else:
            return -1 / self._b

            # @staticmethod
            # TODO implement correct one
            # def quad_int():
            #     return 2/3


class LagrangeSecondOrder(Function):
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
    :param top: top node, where :math:`f(x) = 1`
    :param end: end node
    :param max_element_length: value of the length d (see sketch)
    """

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


def cure_interval(test_function_class, interval, node_count=None, node_distance=None):
    """
    Use test functions to cure an interval with either node_count nodes or nodes with node_node_distance.

    :param test_function_class: class to cure the interval (e.g. py:LagrangeFirstOrder)
    :param interval: tuple of limits that constrain the interval
    :param node_count: amount of nodes to use
    :param node_distance: distance of nodes

    :return: tuple of nodes and functions
    """
    if not issubclass(test_function_class, Function):
        raise TypeError("test_function_class must be a SubClass of Function.")

    # TODO move these into "cure_hint" method of core.Function
    if test_function_class not in {LagrangeFirstOrder, LagrangeSecondOrder}:
        raise TypeError("LagrangeFirstOrder and LagrangeSecondOrder supported as test_function_class for now.")

    if test_function_class is LagrangeFirstOrder:
        domain = Domain(bounds=interval, step=node_distance, num=node_count)

        # interval boundaries
        test_functions = [LagrangeFirstOrder(domain[0], domain[0], domain[0] + domain.step),
                          LagrangeFirstOrder(domain[-1] - domain.step, domain[-1], domain[-1])]
        # interior case
        for node in domain[1:-1]:
            test_functions.insert(-1, LagrangeFirstOrder(node - domain.step,
                                                         node,
                                                         node + domain.step))
    elif test_function_class is LagrangeSecondOrder:
        # create extra nodes for easier handling
        inner_cnt = 2 * node_count - 1
        # inner_dist = node_distance / 2

        # domain = np.sort(np.concatenate((domain, domain[:-1] + np.diff(domain) / 2)))
        domain = Domain(interval, num=inner_cnt)
        max_element_length = 4 * domain.step

        test_functions = [LagrangeSecondOrder(domain[0], domain[0], domain[0] + 2 * domain.step, max_element_length),
                          LagrangeSecondOrder(domain[-1] - 2 * domain.step, domain[-1], domain[-1], max_element_length)]

        for i in range(1, inner_cnt - 1):
            if i % 2 != 0:
                test_functions.insert(-1, LagrangeSecondOrder(domain[i] - domain.step,
                                                              domain[i],
                                                              domain[i] + domain.step,
                                                              max_element_length))
            else:
                test_functions.insert(-1, LagrangeSecondOrder(domain[i] - 2 * domain.step,
                                                              domain[i],
                                                              domain[i] + 2 * domain.step,
                                                              max_element_length))

    return domain, np.asarray(test_functions)
