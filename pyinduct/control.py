
from itertools import chain
import numpy as np

from .registry import get_base
from .core import domain_intersection, integrate_function, \
    TransformationInfo, get_weight_transformation
from .placeholder import EquationTerm, ScalarTerm, IntegralTerm, Scalars, FieldVariable, get_common_target
from .simulation import SimulationInput, CanonicalForms
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


class Controller(SimulationInput):
    """
    wrapper class for all controllers that have to interact with the simulation environment

    :param control_law: function handle that calculates the control output if provided with correct weights
    """

    def __init__(self, control_law):
        SimulationInput.__init__(self, name=control_law.name)
        c_forms = approximate_control_law(control_law)
        self._evaluator = LawEvaluator(c_forms, self._value_storage)

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
    print("approximating control law {}".format(control_law.name))
    if not isinstance(control_law, ControlLaw):
        raise TypeError("only input of Type ControlLaw allowed!")

    return _parse_control_law(control_law)


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

    cfs = CanonicalForms(law.name)

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
            init_funcs = get_base(func_lbl, field_var.order[1])

            factors = np.atleast_2d([integrate_function(func, domain_intersection(term.limits, func.nonzero))[0]
                                     for func in init_funcs])

            if placeholders["scalars"]:
                scales = placeholders["scalars"][0]
                res = np.prod(np.array([factors, scales]), axis=0)
            else:
                res = factors

            # HACK! hardcoded exponent
            cfs.add_to(weight_lbl, dict(name="E", order=temp_order, exponent=1), res * term.scale)

        elif placeholders["scalars"]:
            # TODO make sure that all have the same target form!
            scalars = placeholders["scalars"]
            if len(scalars) > 1:
                # TODO if one of 'em is just a scalar and no array an error occurs
                res = np.prod(np.array([scalars[0].data, scalars[1].data]), axis=0)
            else:
                res = scalars[0].data

            cfs.add_to(scalars[0].target_form, get_common_target(scalars), res * term.scale)

        else:
            raise NotImplementedError

    return cfs


class LawEvaluator(object):
    """
    object that evaluates the control law approximation given by a CanonicalForms object
    """
    def __init__(self, cfs, storage=None):
        self._cfs = cfs
        self._transformations = {}
        self._eval_vectors = {}
        self._storage = storage

    @staticmethod
    def _build_eval_vector(terms):
        """
        build a set of vectors that will compute the output by multiplication with the corresponding
        power of the weight vector

        :param terms: coefficient vectors
        :return: evaluation vector
        """
        orders = set(terms["E"].keys())
        powers = set(chain.from_iterable([list(mat) for mat in terms["E"].values()]))
        dim = next(iter(terms["E"][max(orders)].values())).shape

        vectors = {}
        for power in powers:
            vector = np.hstack([terms["E"].get(order, {}).get(1, np.zeros(dim)) for order in range(max(orders)+1)])
            vectors.update({power: vector})

        return vectors

    def __call__(self, weights, weight_label):
        """
        evaluation function for approximated control law
        :param weights: 1d ndarray of approximation weights
        :param weight_label: string, label of functions the weights correspond to.
        :return: control output u
        """
        res = {}
        output = 0+0j

        # add dynamic part
        for lbl, law in self._cfs.get_dynamic_terms().items():
            dst_weights = [0]
            if "E" in law is not None:
                # build eval vector
                if lbl not in self._eval_vectors.keys():
                    self._eval_vectors[lbl] = self._build_eval_vector(law)

                # collect information
                info = TransformationInfo()
                info.src_lbl = weight_label
                info.dst_lbl = lbl
                info.src_base = get_base(weight_label, 0)
                info.dst_base = get_base(lbl, 0)
                info.src_order = int(weights.size / info.src_base.size) - 1
                info.dst_order = int(next(iter(self._eval_vectors[lbl].values())).size / info.dst_base.size) - 1

                # look up transformation
                if info not in self._transformations.keys():
                    # fetch handle
                    handle = get_weight_transformation(info)
                    self._transformations[info] = handle

                # transform weights
                dst_weights = self._transformations[info](weights)

                # evaluate
                vectors = self._eval_vectors[lbl]
                for p, vec in vectors.items():
                    output = output + np.dot(vec, np.power(dst_weights, p))

            res[lbl] = dst_weights
            # if self._storage is not None:
            #     entry = self._storage.get(info.dst_lbl, [])
            #     entry.append(dst_weights)
            #     self._storage[info.dst_lbl] = entry

        # add constant term
        static_terms = self._cfs.get_static_terms()
        if "f" in static_terms:
            output = output + static_terms["f"]

        # TODO: replace with the one from utils
        if abs(np.imag(output)) > np.finfo(np.complex128).eps * 100:
            print("Warning: Imaginary part of output is nonzero! out = {0}".format(output))

        out = np.real_if_close(output, tol=10000000)
        if np.imag(out) != 0:
            raise ValueError("calculated complex control output u={0},"
                                          " check for errors in control law!".format(out))

        res["output"] = out
        return res
