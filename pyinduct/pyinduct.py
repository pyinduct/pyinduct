# -*- coding: utf-8 -*-
from __future__ import division
import numpy as np
import scipy as sp

class EvalData:
    """
    convenience wrapper for function evaluation
    contains the input data that was used for evaluation and the results
    """

    def __init__(self, input_data, output_data):
        # check type and dimensions
        assert isinstance(input_data, list)
        assert isinstance(output_data, np.ndarray)
        assert len(input_data) == len(output_data.shape)

        for dim in range(len(output_data.shape)):
            assert len(input_data[dim]) == output_data.shape[dim]

        self.input_data = input_data
        self.output_data = output_data


class Function:
    """
    To ensure the accurateness of numerical handling, areas where it is nonzero have to be provided
    The user can choose between providing a (piecewise) analytical or pure numerical representation of the function
    """

    def __init__(self, function_data, domain=None, nonzero=None):
        if callable(function_data):
            self._analytic_handle = function_data
            if domain is None:
                raise ValueError("No domain given")
            self.domain = domain
            self.continuous = True
        elif isinstance(function_data, EvalData):
            self._numerical_data = function_data
            self.domain = function_data.input_data
            self.continuous = False
        else:
            raise TypeError("Given data type not supported")

        if nonzero is None:
            # pretty useless, though
            self._nonzero_area = [()]
        elif nonzero is True:
            # convenience since nonzero everywhere in domain
            self._nonzero_area = domain
        else:
            self._nonzero_area = nonzero

    @property
    def nonzero(self):
        """
        :return: list of tuples tha represent domains where phi(z) is nonzero
        """
        return self._nonzero_area

    def __call__(self, *args):
        """
        handle that is used to evaluate the function on a given point
        :param z: input location
        :return: function value
        """
        if hasattr(self, "_analytic_handle"):
            return self._analytic_handle(*args)
        elif not self.continuous:
            multi_idx = []
            for idx, arg in enumerate(args):
                value_idx = np.where(self._numerical_data.input_data[idx] == arg)
                if value_idx[0].size == 0:
                    raise ValueError("Value cannot be provided!")
                multi_idx.append(value_idx)

            return self._numerical_data.output_data[multi_idx]


def inner_product(first, second):
    """
    calculates the inner product of two functions
    :param first: function
    :param second: function
    :return: inner product
    """
    if not isinstance(first, Function) or not isinstance(second, Function):
        raise TypeError("Wrong type supplied must be a pyinduct.Function")

    # TODO remember not only 1d possible here!
    limits = domain_intersection(first.domain, second.domain)
    result = 0
    for lim in limits:
        if first.continuous and second.continuous:
            # use quad
            f = lambda z: first(z)*second(z)
            res = sp.integrate.quad(f, lim[0], lim[1])
        else:
            # use simpson (discrete)
            res = np.integrate.simps()
        result += res


def domain_intersection(first, second):
    """
    calculate intersection two domains
    :param first: domain
    :param second: domain
    :return: intersection
    """
    if isinstance(first, tuple):
        first = [first]
    if isinstance(second, tuple):
        second = [second]

    intersection = []
    start = None
    end = None
    first_idx = 0
    second_idx = 0
    last_first_idx = 0
    last_second_idx = 0
    last_first_upper = None
    last_second_upper = None

    while first_idx < len(first) and second_idx < len(second):
        if last_first_upper is not None and first_idx is not last_first_idx:
            if last_first_upper >= first[first_idx][0]:
                raise ValueError("Intervals not ordered!")
        if last_second_upper is not None and second_idx is not last_second_idx:
            if last_second_upper >= second[second_idx][0]:
                raise ValueError("Intervals not ordered!")

        if first[first_idx][0] > first[first_idx][1]:
            raise ValueError("Interval Boundaries given in wrong order")
        if second[second_idx][0] > second[second_idx][1]:
            raise ValueError("Interval Boundaries given in wrong order")

        # backup for interval order check
        last_first_idx = first_idx
        last_second_idx = second_idx
        last_first_upper = first[first_idx][1]
        last_second_upper = second[second_idx][1]

        # no common domain -> search
        if second[second_idx][0] <= first[first_idx][0] <= second[second_idx][1]:
            # common start found in 1st domain
            start = first[first_idx][0]
        elif first[first_idx][0] <= second[second_idx][0] <= first[first_idx][1]:
            # common start found in 2nd domain
            start = second[second_idx][0]
        else:
            # intervals have no intersection
            first_idx += 1
            continue

        # add end
        if first[first_idx][1] <= second[second_idx][1]:
            end = first[first_idx][1]
            first_idx += 1
        else:
            end = second[second_idx][1]
            second_idx += 1

        # complete domain found
        intersection.append((start, end))

    return intersection
