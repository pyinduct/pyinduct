from pyinduct.simulation import SimulationInput

from pyinduct.control import approximate_feedback_law, LawEvaluator


class ObserverError(SimulationInput):
    """
    Wrapper class for all observer errors that have to interact with the simulation environment. The terms which
    have to approximated on the basis of the system weights have to provided through the argument :code:`sys_part`
    and the terms which have to approximated on the basis of the observer weights have to provided through the
    argument :code:`obs_part`. The observer error is provided as sum of the (:py:class:`FeedbackLaw`)'s
    :code:`sys_part` and :code:`obs_part`.

    Args:
        sys_part (:py:class:`FeedbackLaw`): Hold the terms which approximated from system weights.
        obs_part (:py:class:`FeedbackLaw`): Hold the terms which approximated from observer weights.
    """

    def __init__(self, sys_part, obs_part):
        SimulationInput.__init__(self, name="observer error: " + sys_part.name + " + " + obs_part.name)
        sys_c_forms = approximate_feedback_law(sys_part)
        self._sys_evaluator = LawEvaluator(sys_c_forms)
        obs_c_forms = approximate_feedback_law(obs_part)
        self._obs_evaluator = LawEvaluator(obs_c_forms)

    def _calc_output(self, **kwargs):
        """
        Calculates the observer error based on the system and the observer weights.

        Keyword Args:
            sys_weights: Current weights of the simulations system approximation.
            sys_weights_lbl (str): Corresponding label of :code:`sys_weights`.
            obs_weights: Current weights of the observer system approximation.
            obs_weights_lbl (str): Corresponding label of :code:`obs_weights`.

        Return:
            dict: Feedback under the key :code:`"output"`.
        """
        return self._sys_evaluator(kwargs["sys_weights"], kwargs["sys_weight_lbl"]) + \
               self._obs_evaluator(kwargs["obs_weights"], kwargs["obs_weight_lbl"])


def build_observer_from_state_space(self, state_space):
    """
    Return a :py:class:`Observer` object based on the given :py:class:`StateSpace` object.
    The method return :code:`None` if state_space.input is not a instance of
    :py:class:`pyinduct.control.ObserverError` or if self._input_function is a instance of
    :py:class:`SimulationInputSum` which not contain any :py:class:`pyinduct.control.ObserverError` instance.

    Returns:
        :py:class:`pyinduct.simulation.Observer` or None: See docstring.
    """
    pass

