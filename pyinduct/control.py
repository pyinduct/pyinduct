"""
This module contains all classes and functions related to the approximation of
distributed controllers and observers as well as their implementation for
simulation purposes.
"""

import numpy as np
from itertools import chain

from .registry import get_base
from .core import (get_weight_transformation, get_transformation_info,
                   calculate_scalar_product_matrix)
from .simulation import SimulationInput, parse_weak_formulation

__all__ = ["Controller", "ObserverFeedback"]

class Controller(SimulationInput):
    """
    Wrapper class for all controllers that have to interact with the simulation
    environment.

    Args:
        control_law (:py:class:`.WeakFormulation`): Function handle that
            calculates the control output if provided with correct weights.
    """

    def __init__(self, control_law):
        SimulationInput.__init__(self, name=control_law.name)
        self.ce = parse_weak_formulation(control_law, finalize=False)
        self.feedback_gains = dict()

    def _calc_output(self, **kwargs):
        """
        Calculates the controller output based on the current_weights.

        Keyword Args:
            weights: Current weights of the simulations system approximation.
            weights_lbl (str): Corresponding label of :code:`weights`.

        Return:
            dict: Controller output :math:`u`.
        """

        # determine feedback gain
        if kwargs["weight_lbl"] not in self.feedback_gains:
            self.feedback_gains[kwargs["weight_lbl"]] = \
                evaluate_trafos(self.ce,
                                kwargs["weight_lbl"],
                                (1, len(kwargs["weights"])))

        # (state) feedback u = k^T * x
        res = self.feedback_gains[kwargs["weight_lbl"]] @ kwargs["weights"]

        # add constant term
        res += np.sum([st for st in self.ce.get_static_terms().values()])

        return dict(output=res)


class ObserverFeedback:
    """
    Wrapper class for all observer gains that have to interact with the
    simulation environment.

    Args:
        observer_gain (:py:class:`.WeakFormulation`): Observer gain projected
            on a set of test functions.
        output_error (:py:class:`.Controller`): Output error
    """

    def __init__(self, observer_gain, output_error):
        SimulationInput.__init__(self, name=observer_gain.name)
        self.ce = parse_weak_formulation(
            observer_gain, is_observer=True, finalize=False)
        self.feedback_gains = dict()
        self.output_error = output_error

    def _calc_output(self, **kwargs):
        """
        Calculates the observer error intrusion.

        Keyword Args:
            weights: Current weights of the simulations system approximation.
            weights_lbl (str): Corresponding label of :code:`weights`.

        Return:
            dict: Observer error intrusion.
        """

        # determine feedback gain
        if kwargs["obs_weight_lbl"] not in self.feedback_gains:
            self.feedback_gains[kwargs["obs_weight_lbl"]] = \
                evaluate_trafos(self.ce,
                                kwargs["obs_weight_lbl"],
                                (len(get_base(kwargs["obs_weight_lbl"])), 1),
                                is_observer=True)

        # calculate output error intrusion
        res = self.feedback_gains[kwargs["obs_weight_lbl"]] * self.output_error(
            **kwargs)

        # add constant term
        res += np.sum([st for st in self.ce.get_static_terms().values()])

        return dict(output=res)


def evaluate_trafos(ce, weight_label, vect_shape, is_observer=False):
    """
    Transform the different feedback/observer gains in `ce` to the basis
    `weight_label` and accumulate them to one gain vector.

    Args:
        ce (:py:class:`.CanonicalEquation`): Feedback/observer gain.
        weight_label (string): Label of functions the weights correspond to.
        vect_shape (tuple): Shape of the feedback vector.
        is_observer (bool): The argument `ce` is interpreted as
            feedback/observer if `observer` is False/True. Default: False

    Return:
        :class:`numpy.array`: Accumulated feedback/observer gain.
    """
    gain = np.zeros(vect_shape)
    identity = np.eye(max(vect_shape))

    for lbl, law in ce.get_dynamic_terms().items():
        if "E" in law:
            # build eval vector
            vectors = _build_eval_vector(law)
            if any([p != 1 for p in vectors]):
                raise NotImplementedError

            # collect information
            org_base = get_base(lbl)
            tar_base = get_base(weight_label)
            if is_observer:
                info = get_transformation_info(
                    lbl,
                    weight_label,
                    int(next(iter(vectors.values())).size
                        / org_base.fractions.size) - 1,
                    int(max(vect_shape) / len(tar_base)) - 1)
            else:
                info = get_transformation_info(
                    weight_label,
                    lbl,
                    int(max(vect_shape) / len(tar_base)) - 1,
                    int(next(iter(vectors.values())).size
                        / org_base.fractions.size) - 1)

            # fetch handle
            transformation = get_weight_transformation(info)

            # evaluate
            if is_observer:
                # map the available projections to the origin weights
                org_weights_trafo = calculate_scalar_product_matrix(
                    org_base.scalar_product_hint()[0], org_base, org_base)
                # map the desired projections to the target weights
                tar_weights_trafo = calculate_scalar_product_matrix(
                    tar_base.scalar_product_hint()[0], tar_base, tar_base)
                # map the availabel projections to the target projections
                gain += tar_weights_trafo @ transformation(
                    np.linalg.inv(org_weights_trafo) @ vectors[1])
            else:
                for i, iv in enumerate(identity):
                    gain[0, i] += np.dot(vectors[1], transformation(iv))

    return gain


def _build_eval_vector(terms):
    """
    Build a set of vectors that will compute the output by multiplication with
    the corresponding power of the weight vector.

    Args:
        terms (dict): coefficient vectors
    Return:
        dict: evaluation vector
    """
    orders = set(terms["E"].keys())
    powers = set(chain.from_iterable(
        [list(mat) for mat in terms["E"].values()]
    ))
    dim = next(iter(terms["E"][max(orders)].values())).shape

    vectors = {}
    for power in powers:
        vector = np.hstack([terms["E"].get(order, {}).get(power, np.zeros(dim))[:dim[0], :dim[1]]
                            for order in range(max(orders) + 1)])
        vectors.update({power: vector})

    return vectors
