from __future__ import division
import numpy as np

from core import project_weights
from core import domain_intersection, integrate_function
from placeholder import EquationTerm, ScalarTerm, IntegralTerm, Scalars, ScalarFunctions, FieldVariable, get_scalar_target
from simulation import CanonicalForm

__author__ = 'Stefan Ecklebe'
"""
This module contains all classes and functions related to the creation of controllers as well as the implementation
for simulation purposes.
"""


class Controller(object):
    """
    wrapper class for all controllers that have to interact with the simulation environment

    :param control_handle: function handle that calculates the control output if provided with correct weights
    :param base_transformation: Matrix that transforms simulation weights into control weights
    """

    def __init__(self, control_handle, base_transformation=None):
        self._control_handle = control_handle
        self._base_transform = base_transformation

    def calculate_output(self, time, current_weights):
        """
        calculates the controller output based on the current_weights
        :param current_weights: current weights of the simulations system approximation
        :return: control output :math:`u`
        """
        if self._base_transform:
            weights = project_weights(current_weights, self._base_transform)
        else:
            weights = current_weights

        return self._control_handle(time, weights)


def approximate_control_law(control_law):
    """
    function that approximates the control law, given by a list of sum terms that equal u.
    the result is a function handle that contains pre-evaluated terms and only needs the current weights to be applied

    :param control_law: list of :py:cls:EquationTerm
    :return:
    """
    if not isinstance(control_law, list):
        raise TypeError("only list accepted.")

    scal_terms = []
    int_terms = []

    # sort terms
    for term in control_law:
        if not isinstance(term, EquationTerm):
            raise TypeError("only EquationTerm(s) accepted.")

        if isinstance(term, ScalarTerm):
            scal_terms.append(term)
        elif isinstance(term, IntegralTerm):
            int_terms.append(term)
        else:
            raise NotImplementedError

    coll_handle = _handle_collocated_terms(scal_terms)
    cont_handle = _handle_continuous_terms(int_terms)

    def eval_func(weights):
        return coll_handle(weights) + cont_handle(weights)

    return eval_func


def _handle_collocated_terms(terms):
    """
    processes the collocated terms inside a control law

    :param terms:
    :return:
    """
    cf = CanonicalForm()

    for term in terms:
        scalars = term.arg.get_arg_by_class(Scalars)
        if len(scalars) > 1:
            res = np.prod(np.array([scalars[0].data, scalars[1].data]), axis=0)
        else:
            res = scalars[0].data

        cf.add_to(get_scalar_target(scalars), res*term.scale)

    processed_terms = cf.get_terms()

    def eval_func(weights):
        """
        evaluation function
        :param weights:  np.ndarray of approximation weights (axis 0) and their time derivatives (axis 1)
        :return: result
        """
        result = 0
        if processed_terms[0] is not None:
            for temp_idx in range(processed_terms[0].shape[0]):
                result += np.dot(processed_terms[0][temp_idx], weights[:, temp_idx])

        if processed_terms[1] is not None:
            result += processed_terms[1][0]

        return result

    return eval_func


def _handle_continuous_terms(terms):
    """
    processes the continuous terms inside a control law.

    first, terms are sorted by interval to save computation time, then handles are generated to numerically integrate
    the terms. These handles are then grouped and returned as one evaluation handle.
    :param terms:
    :return:
    """
    int_handles = []
    cf = CanonicalForm()

    for term in terms:
        placeholders = dict([
            ("field_variables", term.arg.get_arg_by_class(FieldVariable)),
            ("scalars", term.arg.get_arg_by_class(Scalars)),
        ])
        if placeholders["field_variables"]:
            field_var = placeholders["field_variables"][0]
            temp_order = field_var.order[0]
            init_funcs = field_var.data

            factors = np.atleast_2d([integrate_function(func, domain_intersection(term.limits, func.nonzero))[0]
                                     for func in init_funcs])
            if placeholders["scalars"]:
                scales = placeholders["scalars"][0]
                res = np.prod(np.array([factors, scales]), axis=0)
            else:
                res = factors

            cf.add_to(("E", temp_order), res*term.scale)

        elif placeholders["scalars"]:
            # integral term with constant argument -> simple case
            scalars = placeholders["scalars"]
            if len(scalars) > 1:
                res = np.prod(np.array([scalars[0].data, scalars[1].data]), axis=0)
            else:
                res = scalars[0].data

            res = res*(term.limits[1] - term.limits[0])
            cf.add_to(get_scalar_target(scalars), res*term.scale)

        else:
            raise NotImplementedError

    processed_terms = cf.get_terms()

    def eval_func(weights):
        """
        evaluation function
        :param weights:  np.ndarray of approximation weights (axis 0) and their time derivatives (axis 1)
        :return: result
        """
        result = 0
        if processed_terms[0] is not None:
            for temp_idx in range(processed_terms[0].shape[0]):
                result += np.dot(processed_terms[0][temp_idx], weights[:, temp_idx])

        if processed_terms[1] is not None:
            result += processed_terms[1][0]

        return result

    return eval_func
