"""
This module contains all classes and functions related to the approximation of distributed control laws
as well as their implementation for simulation purposes.
"""

import numpy as np
from itertools import chain

from .core import real, get_weight_transformation, TransformationInfo
from .registry import get_base
from .simulation import SimulationInput, parse_weak_formulation

__all__ = ["Controller", "LawEvaluator"]

class Controller(SimulationInput):
    """
    Wrapper class for all controllers that have to interact with the simulation environment.

    Args:
        control_law (:py:class:`ControlLaw`): Function handle that calculates the control output if provided with
            correct weights.
    """

    def __init__(self, control_law):
        SimulationInput.__init__(self, name=control_law.name)
        ce = parse_weak_formulation(control_law, finalize=False)
        self._evaluator = LawEvaluator(ce, self._value_storage)

    def _calc_output(self, **kwargs):
        """
        Calculates the controller output based on the current_weights.

        Keyword Args:
            weights: Current weights of the simulations system approximation.
            weights_lbl (str): Corresponding label of :code:`weights`.

        Return:
            dict: Controller output :math:`u`.
        """
        return self._evaluator(kwargs["weights"], kwargs["weight_lbl"])


class LawEvaluator(object):
    """
    Object that evaluates the control law approximation given by a :py:class:`pyinduct.simulation.CanonicalEquations`
    object.

    Args:
        cfs (:py:class:`pyinduct.simulation.CanonicalEquation`): evaluation handle
    """

    def __init__(self, cfs, storage=None):
        self._cfs = cfs
        self._transformations = {}
        self._eval_vectors = {}
        self._storage = storage

    @staticmethod
    def _build_eval_vector(terms):
        """
        Build a set of vectors that will compute the output by multiplication with the corresponding
        power of the weight vector.

        Args:
            terms (dict): coefficient vectors
        Return:
            dict: evaluation vector
        """
        orders = set(terms["E"].keys())
        powers = set(chain.from_iterable([list(mat) for mat in terms["E"].values()]))
        dim = next(iter(terms["E"][max(orders)].values())).shape

        vectors = {}
        for power in powers:
            vector = np.hstack([terms["E"].get(order, {}).get(power, np.zeros(dim))[0, :]
                                for order in range(max(orders) + 1)])
            vectors.update({power: vector})

        return vectors

    def __call__(self, weights, weight_label):
        """
        Evaluation function for approximated control law.

        Args:
            weights (numpy.ndarray): 1d ndarray of approximation weights.
            weight_label (string): Label of functions the weights correspond to.

        Return:
            dict: control output :math:`u`
        """
        res = {}
        output = 0 + 0j

        # add dynamic part
        for lbl, law in self._cfs.get_dynamic_terms().items():
            dst_weights = [0]
            if "E" in law:
                # build eval vector
                if lbl not in self._eval_vectors.keys():
                    self._eval_vectors[lbl] = self._build_eval_vector(law)

                # collect information
                info = TransformationInfo()
                info.src_lbl = weight_label
                info.dst_lbl = lbl
                info.src_base = get_base(weight_label)
                info.dst_base = get_base(lbl)
                info.src_order = int(weights.size / info.src_base.fractions.size) - 1
                info.dst_order = int(next(iter(self._eval_vectors[lbl].values())).size
                                     / info.dst_base.fractions.size) - 1

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
                    output += np.dot(vec, np.power(dst_weights, p))

            res[lbl] = dst_weights

        # add constant term
        static_terms = self._cfs.get_static_terms()
        if "f" in static_terms:
            output = output + static_terms["f"]

        res["output"] = real(output)
        return res
