from __future__ import division
import numpy as np
from scipy.linalg import block_diag

from pyinduct import get_initial_functions
from core import calculate_base_projection, project_weights, domain_intersection, integrate_function
from placeholder import EquationTerm, ScalarTerm, IntegralTerm, Scalars, FieldVariable, get_scalar_target
# from simulation import CanonicalForms, SimulationInput
import simulation as sim
__author__ = 'Stefan Ecklebe'
"""
This module contains all classes and functions related to the creation of controllers as well as the implementation
for simulation purposes.
"""


class ControlLaw(object):
    """
    this class represents the approximated formulation of a control law.
    It can be initialized with several terms (see children of :py:class:`EquationTerm`).
    The equation is interpreted as term_0 + term_1 + ... + term_N = u, where u is the control output.

    :param terms: (list of) of object(s) of type EquationTerm
    """
    def __init__(self, terms, name=""):
        if isinstance(terms, EquationTerm):
            terms = [terms]
        if not isinstance(terms, list):
            raise TypeError("only (list of) {0} allowed".format(EquationTerm))

        for term in terms:
            if not isinstance(term, EquationTerm):
                raise TypeError("Only EquationTerm(s) are accepted.")

        self.terms = terms
        self.name = name


class Controller(sim.SimulationInput):
    """
    wrapper class for all controllers that have to interact with the simulation environment

    :param control_law: function handle that calculates the control output if provided with correct weights
    """

    def __init__(self, control_law):
        sim.SimulationInput.__init__(self)
        self._evaluator = approximate_control_law(control_law)

    def _calc_output(self, **kwargs):
        """
        calculates the controller output based on the current_weights
        :param current_weights: current weights of the simulations system approximation
        :return: control output :math:`u`
        """
        return self._evaluator(kwargs["weights"], kwargs["weight_lbl"])


def approximate_control_law(control_law):
    """
    function that approximates the control law, given by a list of sum terms that equal u.
    the result is a function handle that contains pre-evaluated terms and only needs the current weights (and their
    respective label) to be applied

    :param control_law: list of :py:cls:ControlLaw
    :return: evaluation handle
    """
    print("approximating control law")
    if not isinstance(control_law, ControlLaw):
        raise TypeError("only input of Type ControlLaw allowed!")

    approximated_forms = _parse_control_law(control_law)
    return LawEvaluator(approximated_forms)


def _parse_control_law(law):
    """
    parses the given control law by approximating given terms
    :param law:  list of equation terms
    :return: evaluation handle
    """
    # check terms
    for term in law.terms:
        if not isinstance(term, EquationTerm):
            raise TypeError("only EquationTerm(s) accepted.")

    cfs = sim.CanonicalForms(law.name)

    for term in law.terms:
        placeholders = dict([
            ("field_variables", term.arg.get_arg_by_class(FieldVariable)),
            ("scalars", term.arg.get_arg_by_class(Scalars)),
        ])
        if placeholders["field_variables"]:
            field_var = placeholders["field_variables"][0]
            temp_order = field_var.order[0]
            func_lbl = field_var.data["func_lbl"]
            weight_lbl = field_var.data["weight_lbl"]
            init_funcs = get_initial_functions(func_lbl, field_var.order[1])

            factors = np.atleast_2d([integrate_function(func, domain_intersection(term.limits, func.nonzero))[0]
                                     for func in init_funcs])

            if placeholders["scalars"]:
                scales = placeholders["scalars"][0]
                res = np.prod(np.array([factors, scales]), axis=0)
            else:
                res = factors

            cfs.add_to(weight_lbl, ("E", temp_order), res * term.scale)

        elif placeholders["scalars"]:
            # TODO make sure that all have the same target form!
            scalars = placeholders["scalars"]
            if len(scalars) > 1:
                res = np.prod(np.array([scalars[0].data, scalars[1].data]), axis=0)
            else:
                res = scalars[0].data

            cfs.add_to(scalars[0].target_form, get_scalar_target(scalars), res * term.scale)

        else:
            raise NotImplementedError

    return cfs


class LawEvaluator(object):
    """
    object that evaluates the control law approximation given by a CanonicalForms object
    """
    def __init__(self, cfs):
        self._cfs = cfs
        self._transformations = {}
        self._eval_vectors = {}

    @staticmethod
    def _build_transformation_matrix(src_lbl, tar_lbl, src_order, tar_order, use_eye=False):
        """
        constructs a transformation matrix from basis given by 'src' to basis given by 'tar' that transforms all
        temporal derivatives at once.

        :param src_lbl: label of source basis
        :param tar_lbl: label of target basis
        :param src_order: temporal derivative order of src basis
        :param tar_order: temporal derivative order of tar basis
        :param use_eye: use identity as block matrix element
        :return: transformation matrix as 2d np.ndarray
        """
        if src_order < tar_order:
            raise ValueError("higher derivative order needed than provided!")

        # build single transformation
        src_funcs = get_initial_functions(src_lbl, 0)
        tar_funcs = get_initial_functions(tar_lbl, 0)
        if use_eye:
            single_transform = np.eye(src_funcs.size)
        else:
            single_transform = calculate_base_projection(src_funcs, tar_funcs)

        # build block matrix
        part_trafo = block_diag(*[single_transform for i in range(tar_order+1)])
        trafo = np.hstack([part_trafo] + [np.zeros((part_trafo.shape[0], src_funcs.size))
                                          for i in range(src_order-tar_order)])
        return trafo

    @staticmethod
    def _build_eval_vector(terms):
        """
        build a vector that will compute the output by multiplication with the corresponding weight vector
        :param terms: coefficient vectors
        :return: evaluation vector
        """
        return np.hstack([vec for vec in terms[0]])

    def _transform_weights(self, weights, src_lbl, dst_lbl, src_order, dst_order):
        """
        evaluates the given term by transforming the given weights in dst_weights and multiplying them by the given
        vector.

        :param src_lbl:
        :param dst_lbl:
        :param term:
        :return:
        """
        # TODO move this special case one level up since it requires labels which should be dropped
        if src_lbl == dst_lbl:
            mat = self._build_transformation_matrix(src_lbl, dst_lbl, src_order, dst_order, use_eye=True)

            def trafo(weights):
                return np.dot(mat, weights)

        # TODO make use of caching again
        # if lbl not in self._transformations.keys():
        #     self._transformations[lbl] = transform


            dst_funcs = get_initial_functions(lbl, 0)
            src_order = int(weights.size / get_initial_functions(weight_label, 0).size) - 1
            dst_order = int(self._eval_vectors[lbl].size / dst_funcs.size) - 1
            # TODO use only hints
            if hasattr(dst_funcs[0], "transformation_hint") and False:
                transform = dst_funcs[0].transformation_hint(src_order, dst_order, weight_label, lbl)
                if transform:

            else:
                self._transformations[lbl] = self._build_transformation_matrix(weight_label, lbl, src_order,
                                                                               dst_order, use_eye=identical)

        target_weights = np.dot(self._transformations[lbl], weights)
        return trafo_handle(weights)


    def __call__(self, weights, weight_label):
        """
        evaluation function for approximated control law
        :param weights: 1d ndarray of approximation weights
        :param weight_label: string, label of functions the weights correspond to.
        :return: control output u
        """
        output = 0

        # add dynamic part
        for lbl, law in self._cfs.get_dynamic_terms().iteritems():
            if law[0] is not None:
                # build eval vector
                if lbl not in self._eval_vectors.keys():
                    self._eval_vectors[lbl] = self._build_eval_vector(law)

                dst_weights = self._transform_weights(weights, src_base, dst_base)
                return np.dot(self._eval_vectors[lbl], dst_weights)


        # add constant term
        static_terms = self._cfs.get_static_terms()
        if static_terms[1] is not None:
            output += static_terms[1][0]

        return output
