"""
Simulation infrastructure with helpers and data structures for preprocessing of the given equations
and functions for postprocessing of simulation data.
"""

import warnings
from abc import ABCMeta, abstractmethod
from collections import OrderedDict
from collections.abc import Callable
from copy import copy
from itertools import chain

import numpy as np
from scipy.integrate import ode
from scipy.interpolate import interp1d
from scipy.linalg import block_diag

from .core import (Domain, Parameters, Function,
                   domain_intersection, integrate_function,
                   calculate_scalar_product_matrix,
                   vectorize_scalar_product, sanitize_input,
                   StackedBase, get_weight_transformation,
                   get_transformation_info,
                   EvalData, project_on_bases)
from .placeholder import (Scalars, TestFunction, Input, FieldVariable,
                          EquationTerm, get_common_target, get_common_form,
                          ObserverGain, ScalarTerm, IntegralTerm,
                          ScalarProductTerm)
from .registry import get_base, register_base

__all__ = ["SimulationInput", "SimulationInputSum", "WeakFormulation",
           "parse_weak_formulation",
           "create_state_space", "StateSpace", "simulate_state_space",
           "simulate_system", "simulate_systems",
           "get_sim_result", "evaluate_approximation",
           "parse_weak_formulations",
           "get_sim_results", "set_dominant_labels", "CanonicalEquation",
           "CanonicalForm", "SimulationInputVector"]


class SimulationInput(object, metaclass=ABCMeta):
    """
    Base class for all objects that want to act as an input for the time-step
    simulation.

    The calculated values for each time-step are stored in internal memory and
    can be accessed by :py:meth:`.get_results` (after the simulation is
    finished).

    Note:
        Due to the underlying solver, this handle may get called with time
        arguments, that lie outside of the specified integration domain. This
        should not be a problem for a feedback controller but might cause
        problems for a feedforward or trajectory implementation.
    """

    def __init__(self, name=""):
        self._time_storage = []
        self._value_storage = {}
        self.name = name
        self._res = np.array([0])

    def __call__(self, **kwargs):
        """
        handle that is used by the simulator to retrieve input.
        """
        out = self._calc_output(**kwargs)
        self._time_storage.append(kwargs["time"])
        for key, value in out.items():
            entries = self._value_storage.get(key, [])
            entries.append(copy(value))
            self._value_storage[key] = entries

        return np.atleast_1d(out["output"])

    @abstractmethod
    def _calc_output(self, **kwargs):
        """
        Handle that has to be implemented for output calculation.

        Keyword Args:
            time: The current simulation time.
            weights: The current weight vector.
            weight_lbl: The label of the weights used.

        Returns:
            dict: Dictionary with mandatory key ``output``.
        """
        return dict(output=self._res)

    def get_results(self, time_steps, result_key="output",
                    interpolation="nearest", as_eval_data=False):
        """
        Return results from internal storage for given time steps.

        Raises:
            Error: If calling this method before a simulation was run.

        Args:
            time_steps: Time points where values are demanded.
            result_key: Type of values to be returned.
            interpolation: Interpolation method to use if demanded time-steps
                are not covered by the storage, see
                :func:`scipy.interpolate.interp1d` for all possibilities.
            as_eval_data (bool): Return results as
                :py:class:`.EvalData` object for straightforward display.

        Return:
            Corresponding function values to the given time steps.
        """
        t_data = np.array(self._time_storage)
        res_data = np.array(self._value_storage[result_key])
        invalid_idxs = np.logical_not(np.isnan(res_data))
        mask = [np.all(a) for a in invalid_idxs]

        func = interp1d(t_data[mask],
                        res_data[mask],
                        kind=interpolation,
                        assume_sorted=False,
                        bounds_error=False,
                        fill_value=(res_data[mask][0], res_data[mask][-1]),
                        axis=0)
        values = func(time_steps)

        if as_eval_data:
            return EvalData([time_steps],
                            values,
                            name=".".join([self.name, result_key]),
                            fill_axes=True)

        return values

    def clear_cache(self):
        """
        Clear the internal value storage.

        When the same *SimulationInput* is used to perform various simulations,
        there is no possibility to distinguish between the different runs when
        :py:meth:`.get_results` gets called. Therefore this method can be used
        to clear the cache.
        """
        self._time_storage.clear()
        self._value_storage.clear()


class EmptyInput(SimulationInput):
    def __init__(self, dim):
        SimulationInput.__init__(self)
        self.dim = dim

    def _calc_output(self, **kwargs):
        return dict(output=np.zeros((len(np.atleast_1d(kwargs['time'])), self.dim)))


class SimulationInputSum(SimulationInput):
    """
    Helper that represents a signal mixer.
    """

    def __init__(self, inputs):
        SimulationInput.__init__(self)
        self.inputs = inputs

    def _calc_output(self, **kwargs):
        outs = [handle(**kwargs) for handle in self.inputs]
        return dict(output=np.sum(outs, axis=0))


class WeakFormulation(object):
    r"""
    This class represents the weak formulation of a spatial problem.
    It can be initialized with several terms (see children of
    :py:class:`.EquationTerm`).
    The equation is interpreted as

    .. math:: term_0 + term_1 + ... + term_N = 0.

    Args:
        terms (list): List of object(s) of type EquationTerm.
        name (string): Name of this weak form.
        dominant_lbl (string): Name of the variable that dominates this weak
            form.
    """

    def __init__(self, terms, name, dominant_lbl=None):
        self.terms = sanitize_input(terms, EquationTerm)
        self.name = name
        self.dominant_lbl = dominant_lbl


class StateSpace(object):
    r"""
    Wrapper class that represents the state space form of a dynamic system where

    .. math::
        \boldsymbol{\dot{x}}(t) &= \sum\limits_{k=0}^{L}\boldsymbol{A}_{k}
        \boldsymbol{x}^{p_k}(t)
        + \sum\limits_{j=0}^{V} \sum\limits_{k=0}^{L}\boldsymbol{B}_{j, k}
        \frac{\mathrm{d}^j u^{p_k}}{\mathrm{d}t^j}(t)
        + \boldsymbol{L}\tilde{\boldsymbol{y}}(t)\\
        \boldsymbol{y}(t) &= \boldsymbol{C}\boldsymbol{x}(t)
        + \boldsymbol{D}u(t)

    which has been approximated by projection on a base given by weight_label.

    Args:
        a_matrices (dict): State transition matrices
            :math:`\boldsymbol{A}_{p_k}` for the corresponding powers of
            :math:`\boldsymbol{x}`.
        b_matrices (dict): Cascaded dictionary for the input matrices
            :math:`\boldsymbol{B}_{j, k}` in the sequence: temporal derivative
            order, exponent.
        input_handle (:py:class:`.SimulationInput`): System input :math:`u(t)`.
        c_matrix: :math:`\boldsymbol{C}`
        d_matrix: :math:`\boldsymbol{D}`
    """

    def __init__(self, a_matrices, b_matrices, base_lbl=None,
                 input_handle=None, c_matrix=None, d_matrix=None,
                 obs_fb_handle=None):
        self.C = c_matrix
        self.D = d_matrix
        self.base_lbl = base_lbl
        self.observer_fb = obs_fb_handle

        # mandatory
        if isinstance(a_matrices, np.ndarray):
            self.A = {1: a_matrices}
        else:
            self.A = a_matrices
        if 0 not in self.A:
            # this is the constant term (power 0) aka the f-vector
            self.A[0] = np.zeros((self.A[1].shape[0],))

        # optional
        if isinstance(b_matrices, np.ndarray):
            # fake import order and power for backward compatibility
            self.B = {0: {1: b_matrices}}
        else:
            self.B = b_matrices

        # TODO calculate available order
        available_power = 1
        if self.B is None:
            self.B = {0: {available_power: np.zeros((self.A[available_power].shape[0], available_power))}}
        if self.C is None:
            self.C = np.zeros((available_power, self.A[available_power].shape[1]))
        if self.D is None:
            self.D = np.zeros((self.C.shape[0], np.atleast_2d(self.B[0][available_power]).T.shape[1]))

        if input_handle is None:
            self.input = EmptyInput(self.B[0][available_power].shape[1])
        elif isinstance(input_handle, SimulationInput):
            self.input = input_handle
        else:
            raise NotImplementedError

    # TODO export cython code?
    def rhs(self, _t, _q):
        r"""
        Callback for the integration of the dynamic system, described by this object.

        Args:
            _t (float): timestamp
            _q (array): weight vector

        Returns:
            (array): :math:`\boldsymbol{\dot{x}}(t)`
        """
        state_part = self.A[0]
        for power, a_mat in self.A.items():
            state_part = state_part + a_mat @ np.power(_q, power)

        input_part = np.zeros_like(state_part)
        inputs = np.atleast_2d(
            self.input(time=_t, weights=_q, weight_lbl=self.base_lbl))
        for der_order, power_dict in self.B.items():
            for power, b_mat in power_dict.items():
                for idx, col in enumerate(b_mat.T):
                    input_part = input_part + col * inputs[idx][der_order]

        q_t = state_part + input_part

        if self.observer_fb is not None:
            q_t = q_t + self.observer_fb(
                time=_t, weights=_q, weight_lbl=self.base_lbl)

        return q_t


def simulate_system(weak_form, initial_states,
                    temporal_domain, spatial_domain,
                    derivative_orders=(0, 0), settings=None):
    r"""
    Convenience wrapper for :py:func:`.simulate_systems`.

    Args:
        weak_form (:py:class:`.WeakFormulation`): Weak formulation of the system
            to simulate.
        initial_states (numpy.ndarray): Array of core.Functions for
            :math:`x(t=0, z), \dot{x}(t=0, z), \dotsc, x^{(n)}(t=0, z)`.
        temporal_domain (:py:class:`.Domain`): Domain object holding information
            for time evaluation.
        spatial_domain (:py:class:`.Domain`): Domain object holding information
            for spatial evaluation.
        derivative_orders (tuple): tuples of derivative orders (time, spat) that
            shall be evaluated additionally as values
        settings: Integrator settings, see :py:func:`.simulate_state_space`.
    """
    ics = sanitize_input(initial_states, Function)
    initial_states = {weak_form.name: ics}
    spatial_domains = {weak_form.name: spatial_domain}
    derivative_orders = {weak_form.name: derivative_orders}
    res = simulate_systems([weak_form], initial_states, temporal_domain, spatial_domains, derivative_orders, settings)
    return res


def simulate_systems(weak_forms, initial_states, temporal_domain,
                     spatial_domains, derivative_orders=None, settings=None,
                     out=list()):
    """
    Convenience wrapper that encapsulates the whole simulation process.

    Args:
        weak_forms ((list of) :py:class:`.WeakFormulation`): (list of) Weak
            formulation(s) of the system(s) to simulate.
        initial_states (dict, numpy.ndarray): Array of core.Functions for
            :math:`x(t=0, z), \dot{x}(t=0, z), \dotsc, x^{(n)}(t=0, z)`.
        temporal_domain (:py:class:`.Domain`): Domain object holding
            information for time evaluation.
        spatial_domains (dict): Dict with :py:class:`.Domain` objects holding
            information for spatial evaluation.
        derivative_orders (dict): Dict, containing tuples of derivative orders
            (time, spat) that shall be evaluated additionally as values
        settings: Integrator settings, see :py:func:`.simulate_state_space`.
        out (list): List from user namespace, where the following intermediate
            results will be appended:

            - canonical equations (list of types: :py:class:`.CanocialEquation`)
            - state space object (type: :py:class:`.StateSpace`)
            - initial weights (type: :py:class:`numpy.array`)
            - simulation results/weights (type: :py:class:`numpy.array`)

    Note:
        The *name* attributes of the given weak forms must be unique!

    Return:
        list: List of :py:class:`.EvalData` objects, holding the results for the
        FieldVariable and demanded derivatives.
    """
    if derivative_orders is None:
        derivative_orders = dict([(lbl, (0, 0))for lbl in spatial_domains])

    weak_forms = sanitize_input(weak_forms, WeakFormulation)
    print("simulate systems: {}".format([f.name for f in weak_forms]))

    print(">>> parse weak formulations")
    canonical_equations = parse_weak_formulations(weak_forms)
    out.append(canonical_equations)

    print(">>> create state space system")
    state_space_form = create_state_space(canonical_equations)
    out.append(state_space_form)

    print(">>> derive initial conditions")
    q0 = project_on_bases(initial_states, canonical_equations)
    out.append(q0)

    print(">>> perform time step integration")
    sim_domain, q = simulate_state_space(state_space_form, q0, temporal_domain,
                                         settings=settings)
    out.append(q)

    print(">>> perform postprocessing")
    results = get_sim_results(sim_domain, spatial_domains, q, state_space_form,
                              derivative_orders=derivative_orders)

    print(">>> finished simulation")
    return results


def get_sim_result(weight_lbl, q, temp_domain, spat_domain, temp_order, spat_order, name=""):
    """
    Create handles and evaluate at given points.

    Args:
        weight_lbl (str): Label of Basis for reconstruction.
        temp_order: Order or temporal derivatives to evaluate additionally.
        spat_order: Order or spatial derivatives to evaluate additionally.
        q: weights
        spat_domain (:py:class:`.Domain`): Domain object providing values for
            spatial evaluation.
        temp_domain (:py:class:`.Domain`): Time steps on which rows of q are
            given.
        name (str): Name of the WeakForm, used to generate the data set.
    """
    data = []

    # temporal
    ini_funcs = get_base(weight_lbl).fractions
    for der_idx in range(temp_order + 1):
        name = "{0}{1}".format(name, "_" + "".join(["d" for x in range(der_idx)] + ["t"]) if der_idx > 0 else "")
        data.append(evaluate_approximation(weight_lbl, q[:, der_idx * ini_funcs.size:(der_idx + 1) * ini_funcs.size],
                                           temp_domain, spat_domain, name=name))

    # spatial (0th derivative is skipped since this is already handled above)
    for der_idx in range(1, spat_order + 1):
        name = "{0}{1}".format(name, "_" + "".join(["d" for x in range(der_idx)] + ["z"]) if der_idx > 0 else "")
        data.append(
            evaluate_approximation(weight_lbl, q[:, :ini_funcs.size], temp_domain, spat_domain, der_idx, name=name))

    return data


def get_sim_results(temp_domain, spat_domains, weights, state_space, names=None,
                    derivative_orders=None):
    """
    Convenience wrapper for :py:func:`.get_sim_result`.

    Args:
        temp_domain (:py:class:`.Domain`): Time domain
        spat_domains (dict): Spatial domain from all subsystems which belongs to
            *state_space* as values and name of the systems as keys.
        weights (numpy.array): Weights gained through simulation. For example
            with :py:func:`.simulate_state_space`.
        state_space (:py:class:`.StateSpace`): Simulated state space instance.
        names: List of names of the desired systems. If not given all available
            subssystems will be processed.
        derivative_orders (dict): Desired derivative orders.

    Returns:
        List of :py:class:`.EvalData` objects.
    """
    ss_base = get_base(state_space.base_lbl)
    if names is None:
        if isinstance(ss_base, StackedBase):
            labels = ss_base.base_lbls
            names = ss_base.system_names
        else:
            names = list(spat_domains)
            labels = [state_space.base_lbl]
    else:
        if isinstance(ss_base, StackedBase):
            labels = [ss_base.base_lbls[ss_base.system_names.index(name)]
                      for name in names]
        else:
            labels = [state_space.base_lbl]

    if derivative_orders is None:
        derivative_orders = dict([(name, (0, 0)) for name in names])

    results = []
    for nm, lbl in zip(names, labels):
        # if derivative_orders[n] is None derivatives of the
        # corresponding variables are not provided
        if derivative_orders[nm][0] is None:
            derivative_orders[nm][0] = 0
        if derivative_orders[nm][1] is None:
            derivative_orders[nm][1] = 0

        # acquire a transformation into the original weights
        src_order = int(weights.shape[1] / ss_base.fractions.size) - 1
        info = get_transformation_info(state_space.base_lbl,
                                       lbl,
                                       src_order,
                                       derivative_orders[nm][0])
        transformation = get_weight_transformation(info)

        # project back
        data = get_sim_result(info.dst_lbl,
                              np.apply_along_axis(transformation, 1, weights),
                              temp_domain,
                              spat_domains[nm],
                              info.dst_order,
                              derivative_orders[nm][1],
                              name=nm)
        results += data

    return results


class CanonicalForm(object):
    """
    The canonical form of an nth order ordinary differential equation system.
    """

    def __init__(self, name=None):
        self.name = name
        self.matrices = {}
        # self._max_idx = dict(E=0, f=0, G=0)
        self._weights = None
        self._input_function = None
        self._observer_feedback = list()
        self._finalized = False
        self.powers = None
        self.max_power = None
        self.max_temp_order = None
        self.dim_u = 0
        self.dim_x = None
        self.dim_xb = None
        self.e_n_pb = None
        self.e_n_pb_inv = None
        self.singular = True

    # @staticmethod
    # def _build_name(term):
    #     return "_" + term[0] + str(term[1])

    # def __add__(self, other):
    #     for name, names in other._matrices.items():
    #         for der, derivatives in names.items():
    #             for p, pow in derivatives.items():
    #                 self._matrices[name][der][p] += pow

    @property
    def input_function(self):
        return self._input_function

    def set_input_function(self, func):
        if not isinstance(func, SimulationInput):
            raise TypeError("Inputs must be of type `SimulationInput`.")

        if self._input_function is None:
            self._input_function = func
        elif self._input_function is not func:
            raise ValueError("already defined input is overridden!")

    # @property
    # def weights(self):
    #     return self._weights
    #
    # @weights.setter
    # def weights(self, weight_lbl):
    #     if not isinstance(weight_lbl, str):
    #         raise TypeError("only string allowed as weight label!")
    #     if self._weights is None:
    #         self._weights = weight_lbl
    #     if self._weights != weight_lbl:
    #         raise ValueError("already defined target weights are overridden!")

    def add_to(self, term, value, column=None):
        """
        Adds the value :py:obj:`value` to term :py:obj:`term`. :py:obj:`term` is a dict that describes which
        coefficient matrix of the canonical form the value shall be added to.

        Args:
            term (dict): Targeted term in the canonical form h.  It has to contain:

                - name: Type of the coefficient matrix: 'E', 'f', or 'G'.
                - order: Temporal derivative order of the assigned weights.
                - exponent: Exponent of the assigned weights.
            value (:py:obj:`numpy.ndarray`): Value to add.
            column (int): Add the value only to one column of term (useful if only one dimension of term is known).
        """
        if self._finalized:
            raise RuntimeError("Object has already been finalized, you are trying some nasty stuff there.")

        if term["name"] == "L":
            self._observer_feedback.append(value)
            return

        if not isinstance(value, np.ndarray):
            raise TypeError("val must be numpy.ndarray")
        if column and not isinstance(column, int):
            raise TypeError("column index must be int")

        # get entry
        if term["name"] == "f":
            if ("order" in term) \
                or ("exponent" in term
                    and term["exponent"] != 0):
                warnings.warn("order and exponent are ignored for f_vector!")
            f_vector = self.matrices.get("f", np.zeros_like(value))
            self.matrices["f"] = value + f_vector
            return

        type_group = self.matrices.get(term["name"], {})
        derivative_group = type_group.get(term["order"], {})
        target_matrix = derivative_group.get(term["exponent"],
                                             np.zeros_like(value))

        if target_matrix.shape != value.shape and column is None:
            msg = "{0}{1}{2} was already initialized with dimensions {3} but " \
                  "value to add has dimension {4}".format(term["name"],
                                                          term["order"],
                                                          term["exponent"],
                                                          target_matrix.shape,
                                                          value.shape)
            raise ValueError(msg)

        if column is not None:
            # check whether the dimensions fit or if the matrix must be extended
            if column >= target_matrix.shape[1]:
                new_target_matrix = np.zeros((target_matrix.shape[0],
                                              column + 1))
                new_target_matrix[
                :target_matrix.shape[0],
                :target_matrix.shape[1]
                ] = target_matrix
                target_matrix = new_target_matrix

            target_matrix[:, column:column + 1] += value
        else:
            target_matrix += value

        # store changes
        derivative_group[term["exponent"]] = target_matrix
        type_group[term["order"]] = derivative_group
        self.matrices[term["name"]] = type_group

    def finalize(self):
        """
        Finalizes the object.
        This method must be called after all terms have been added by
        :py:meth:`.add_to` and before :py:meth:`.convert_to_state_space` can be
        called. This functions makes sure that the formulation can be converted
        into state space form (highest time derivative only comes in one power)
        and collects information like highest derivative order, it's power and
        the sizes of current and state-space state vector (`dim_x` resp.
        `dim_xb`). Furthermore, the coefficient matrix of the highest derivative
        order `e_n_pb` and it's inverse are made accessible.
        """
        if self._finalized:
            return

        # get highest power
        self.powers = set(chain.from_iterable([list(mat) for mat in self.matrices["E"].values()]))
        self.max_power = max(self.powers)

        # check whether the system can be formulated in an explicit form
        self.max_temp_order = max(self.matrices["E"])

        if len(self.matrices["E"][self.max_temp_order]) > 1:
            # more than one power of the highest derivative -> implicit formulation
            raise NotImplementedError

        pb = next(iter(self.matrices["E"][self.max_temp_order]))
        if pb != 1:
            # TODO raise the resulting last blocks to 1/pb
            raise NotImplementedError

        self.e_n_pb = self.matrices["E"][self.max_temp_order][pb]
        self.dim_x = self.e_n_pb.shape[0]  # length of the weight vector
        rank_e_n_pb = np.linalg.matrix_rank(self.e_n_pb)
        if rank_e_n_pb != max(self.e_n_pb.shape) or self.e_n_pb.shape[0] != self.e_n_pb.shape[1]:
            # this form cannot be used as dominant form
            self.singular = True
        else:
            self.singular = False
            self.e_n_pb_inv = np.linalg.inv(self.e_n_pb)

        self.dim_xb = self.max_temp_order * self.dim_x  # dimension of the new system

        # input
        for derivatives in self.matrices.get("G", {}).values():
            for power in derivatives.values():
                self.dim_u = max(self.dim_u, power.shape[1])

    def get_terms(self):
        """
        Return all coefficient matrices of the canonical formulation.

        Return:
            Cascade of dictionaries: Structure: Type > Order > Exponent.
        """
        return self.matrices

    def convert_to_state_space(self):
        """
        Convert the canonical ode system of order n a into an ode system of
        order 1.

        Note:
            This will only work if the highest derivative order of the given
            form can be isolated. This is the case if the highest order is only
            present in one power and the equation system can therefore be
            solved for it.

        Return:
            :py:class:`.StateSpace` object:
        """
        if not self._finalized:
            self.finalize()

        # system matrices A_*
        a_matrices = {}
        for p in self.powers:
            a_mat = np.zeros((self.dim_xb, self.dim_xb))

            # add integrator chain
            a_mat[:-self.dim_x:, self.dim_x:] = block_diag(
                *[np.eye(self.dim_x) for a in range(self.max_temp_order - 1)])

            # add "block-line" with feedback entries
            a_mat[-self.dim_x:, :] = -self._build_feedback("E",
                                                           p,
                                                           self.e_n_pb_inv)
            a_matrices.update({p: a_mat})

        # input matrices B_*
        if "G" in self.matrices:
            max_temp_input_order = max(iter(self.matrices["G"]))
            input_powers = set(chain.from_iterable(
                [list(mat) for mat in self.matrices["G"].values()])
            )
            dim_u = next(iter(
                self.matrices["G"][max_temp_input_order].values())).shape[1]

            # generate nested dict of B_o_p matrices where o is
            # derivative order and p is power
            b_matrices = {}
            for order in range(max_temp_input_order + 1):
                if order in self.matrices["G"]:
                    b_powers = {}
                    for q in input_powers:
                        b_mat = np.zeros((self.dim_xb, dim_u))
                        # overwrite the last "block-line" in the matrices
                        # with input entries
                        b_mat[-self.dim_x:, :] = \
                            - self.e_n_pb_inv @ self.matrices["G"][order][q]
                        b_powers.update({q: b_mat})

                    b_matrices.update({order: b_powers})
        else:
            b_matrices = None

        # the f vector aka the A matrix corresponding to the power zero
        f_mat = np.zeros((self.dim_xb,))
        if "f" in self.matrices:
            f_mat[-self.dim_x:] = self.matrices["f"]

        a_matrices.update({0: f_mat})

        ss = StateSpace(a_matrices, b_matrices,
                        input_handle=self.input_function)
        return ss

    def _build_feedback(self, entry, power, product_mat):
        max_order = max(sorted(self.matrices[entry]))
        entry_shape = next(iter(self.matrices[entry][max_order].values())).shape
        if entry == "G":
            # include highest order for system input
            max_order += 1

        blocks = [np.dot(product_mat, self.matrices[entry].get(order, {}).get(power, np.zeros(entry_shape)))
                  for order in range(max_order)]
        return np.hstack(blocks)


class CanonicalEquation(object):
    """
    Wrapper object, holding several entities of canonical forms for different
    weight-sets that form an equation when summed up.
    After instantiation, this object can be filled with information by passing
    the corresponding coefficients to :py:meth:`.add_to`. When the parsing
    process is completed and all coefficients have been collected, calling
    :py:meth:`.finalize` is required to compute all necessary information for
    further processing. When finalized, this object provides access to the
    dominant form of this equation.

    Args:
        name (str): Unique identifier of this equation.
        dominant_lbl (str): Label of the variable that dominates this equation.
    """

    def __init__(self, name, dominant_lbl=None):
        self.name = name
        self.dominant_lbl = dominant_lbl
        self.dynamic_forms = {}
        self._static_form = CanonicalForm(self.name + "_static")
        self._finalized = False
        self._finalized_dynamic_forms = False

    def add_to(self, weight_label, term, val, column=None):
        """
        Add the provided *val* to the canonical form for *weight_label*,
        see :py:meth:`.CanonicalForm.add_to` for further information.

        Args:
            weight_label (str): Basis to add onto.
            term: Coefficient to add onto, see :py:func:`~CanonicalForm.add_to`.
            val: Values to add.
            column (int): passed to :py:func:`~CanonicalForm.add_to`.
        """
        if self._finalized:
            raise RuntimeError("Object has already been finalized, you are trying some nasty stuff there.")

        if term["name"] in "fGL":
            # hold f and g vector separately
            self._static_form.add_to(term, val, column)
            return

        if weight_label is None:
            raise ValueError("weight_label can only be none if target is f or G.")

        if weight_label not in list(self.dynamic_forms.keys()):
            self.dynamic_forms[weight_label] = CanonicalForm("_".join([self.name + weight_label]))

        self.dynamic_forms[weight_label].add_to(term, val)

    def finalize(self):
        """
        Finalize the Object.
        After the complete formulation has been parsed and all terms have been
        sorted into this Object via :py:meth:`.add_to` this function has to be
        called to inform this object about it. Furthermore, the f and G parts of
        the static_form will be copied to the dominant form for easier
        state-space transformation.

        Note:
            This function must be called to use the :py:attr:`dominant_form`
            attribute.

        """
        if self.dominant_lbl is None:
            raise ValueError("You have to set the dominant labels of the\n"
                             "canonical equation (weak form), for example\n"
                             "with pyinduct.simulation.set_dominant_labels().")

        if not self._finalized_dynamic_forms:
            self.finalize_dynamic_forms()

        if self.dynamic_forms[self.dominant_lbl].singular:
            raise ValueError("The form that has to be chosen is singular.")

        # copy static terms to dominant form to transform them correctly
        for letter in "fG":
            if letter in self._static_form.matrices:
                self.dynamic_forms[self.dominant_lbl].matrices.update({letter: self._static_form.matrices[letter]})

        self._finalized = True

    def finalize_dynamic_forms(self):
        """
        Finalize all dynamic forms. See method
        :py:meth:`.CanonicalForm.finalize`.
        """
        for lbl, form in self.dynamic_forms.items():
            form.finalize()
        self._finalized_dynamic_forms = True

    @property
    def static_form(self):
        """
        :py:class:`.WeakForm` that does not depend on any weights.
        :return:
        """
        return self._static_form

    @property
    def dominant_form(self):
        """
        direct access to the dominant :py:class:`.CanonicalForm`.

        Note:
            :py:meth:`.finalize` must be called first.

        Returns:
            :py:class:`.CanonicalForm`: the dominant canonical form
        """
        if self.dominant_lbl is None:
            raise RuntimeError("Dominant label is not defined! Use for\n"
                               "expample pyinduct.simulation."
                               "set_dominant_label or set it manually.")
        return self.dynamic_forms[self.dominant_lbl]

    def get_static_terms(self):
        """
        Return:
            Terms that do not depend on a certain weight set.
        """
        return self._static_form.get_terms()

    def get_dynamic_terms(self):
        """
        Return:
            dict: Dictionary of terms for each weight set.
        """
        return {label: val.get_terms() for label, val in self.dynamic_forms.items()}

    @property
    def input_function(self):
        """
        The input handles for the equation.
        """
        return self._static_form.input_function

    def set_input_function(self, func):
        self._static_form.set_input_function(func)


def create_state_space(canonical_equations):
    """
    Create a state-space system constituted by several
    :py:class:`.CanonicalEquations` (created by
    :py:func:`.parse_weak_formulation`)

    Args:
        canonical_equations: List of :py:class:`.CanonicalEquation`'s.

    Raises:
        ValueError: If compatibility criteria cannot be fulfilled

    Return:
        :py:class:`.StateSpace`: State-space representation of the approximated
        system
    """
    set_dominant_labels(canonical_equations)

    if isinstance(canonical_equations, CanonicalEquation):
        # backward compatibility
        canonical_equations = [canonical_equations]

    # check whether the formulations are compatible
    for eq in canonical_equations:
        for lbl, form in eq.dynamic_forms.items():
            coupling_order = form.max_temp_order

            # search corresponding dominant form in other equations
            for _eq in canonical_equations:
                # check uniqueness of name - dom_lbl mappings
                if eq.name != _eq.name and eq.dominant_lbl == _eq.dominant_lbl:
                    raise ValueError("A dominant form has to be unique over all given Equations")

                # identify coupling terms
                if lbl == eq.dominant_lbl:
                    break

                # identify corresponding dominant form
                if _eq.dominant_lbl != lbl:
                    continue

                dominant_order = _eq.dominant_form.max_temp_order
                if dominant_order <= coupling_order:
                    # dominant order has to be at least one higher than
                    # the coupling order
                    raise ValueError("Formulations are not compatible")

    # transform dominant forms into state-space representation
    # and collect information
    dominant_state_spaces = {}
    state_space_props = Parameters(size=0,
                                   parts=OrderedDict(),
                                   powers=set(),
                                   input_powers=set(),
                                   dim_u=0,
                                   input=None)
    for eq in canonical_equations:
        dom_lbl = eq.dominant_lbl
        dom_form = eq.dominant_form
        dom_ss = dom_form.convert_to_state_space()
        dominant_state_spaces.update({dom_lbl: dom_ss})

        # collect some information
        state_space_props.parts[dom_lbl] = dict(start=copy(state_space_props.size),
                                                orig_size=dom_form.dim_x,
                                                size=dom_form.dim_xb,
                                                order=dom_form.max_temp_order - 1,
                                                sys_name=eq.name)
        state_space_props.powers.update(dom_form.powers)
        state_space_props.size += dom_form.dim_xb
        state_space_props.dim_u = max(state_space_props.dim_u, dom_form.dim_u)

        # update input handles
        if state_space_props.input is None:
            state_space_props.input = eq.input_function
        elif eq.input_function is not None:
            if not state_space_props.input is eq.input_function:
                raise ValueError("Only one input object allowed.")

    # build new basis by concatenating the dominant bases of every equation
    if len(canonical_equations) == 1:
        new_name = next(iter(canonical_equations)).dominant_lbl
    else:
        base_info = copy(state_space_props.parts)
        base_lbls = state_space_props.parts.keys()
        for lbl in base_lbls:
            base_info[lbl].update({"base": get_base(lbl)})
        new_base = StackedBase(base_info)
        new_name = "_".join(base_lbls)
        register_base(new_name, new_base)

    # build new state transition matrices A_p_k for corresponding powers p_k of the state vector
    a_matrices = {}
    for p in state_space_props.powers:
        a_mat = np.zeros((state_space_props.size, state_space_props.size))
        for row_eq in canonical_equations:
            row_dom_lbl = row_eq.dominant_lbl
            row_dom_dim = state_space_props.parts[row_dom_lbl]["size"]
            row_dom_trans_mat = row_eq.dominant_form.e_n_pb_inv
            row_dom_sys_mat = dominant_state_spaces[row_dom_lbl].A.get(p, None)
            row_idx = state_space_props.parts[row_dom_lbl]["start"]

            for col_eq in canonical_equations:
                col_dom_lbl = col_eq.dominant_lbl

                # main diagonal
                if col_eq.name == row_eq.name:
                    if row_dom_sys_mat is not None:
                        a_mat[row_idx:row_idx + row_dom_dim, row_idx:row_idx + row_dom_dim] = row_dom_sys_mat
                    continue

                # coupling terms
                if col_dom_lbl in row_eq.dynamic_forms:
                    for order, mats in row_eq.dynamic_forms[col_dom_lbl].matrices["E"].items():
                        orig_mat = mats.get(p, None)
                        if orig_mat is not None:
                            # transform matrix with row-transformation matrix and add to last "row"
                            # since it's not the dominant entry, revert sign change
                            cop_mat = row_dom_trans_mat @ -orig_mat
                            v_idx = row_idx + row_dom_dim - state_space_props.parts[row_dom_lbl]["orig_size"]
                            col_idx = state_space_props.parts[col_dom_lbl]["start"]
                            h_idx = col_idx + order * state_space_props.parts[col_dom_lbl]["orig_size"]
                            a_mat[v_idx: v_idx + cop_mat.shape[0], h_idx: h_idx + cop_mat.shape[1]] = cop_mat

        a_matrices.update({p: a_mat})

    # build new state input matrices
    b_matrices = {}
    for name, dom_ss in dominant_state_spaces.items():
        for order, order_mats in dom_ss.B.items():
            b_order_mats = b_matrices.get(order, {})
            for p, power_mat in order_mats.items():
                b_power_mat = b_order_mats.get(p, np.zeros((state_space_props.size, state_space_props.dim_u)))

                # add entry to the last "row"
                r_idx = state_space_props.parts[name]["start"]  # - state_space_props.parts[name]["orig_size"]
                b_power_mat[r_idx: r_idx + power_mat.shape[0], :power_mat.shape[1]] = power_mat

                b_order_mats.update({p: b_power_mat})
            b_matrices.update({order: b_order_mats})

    # build observer feedback handle
    def observer_feedback(**kwargs):
        res = np.zeros(state_space_props.size)
        for ce in canonical_equations:
            for fb in ce._static_form._observer_feedback:
                idx_a = (state_space_props.parts[ce.dominant_lbl]["start"] +
                         state_space_props.parts[ce.dominant_lbl]["orig_size"] *
                         state_space_props.parts[ce.dominant_lbl]["order"])
                idx_b = (idx_a +
                         state_space_props.parts[ce.dominant_lbl]["orig_size"])

                kwargs.update(obs_weight_lbl=ce.dominant_lbl)
                res[idx_a: idx_b] += ce.dominant_form.e_n_pb_inv @ np.squeeze(
                    fb._calc_output(**kwargs)["output"], 1)

                kwargs.pop("obs_weight_lbl")

        return res

    dom_ss = StateSpace(a_matrices, b_matrices, base_lbl=new_name,
                        input_handle=state_space_props.input,
                        obs_fb_handle=observer_feedback)
    return dom_ss


def parse_weak_formulation(weak_form, finalize=False, is_observer=False):
    r"""
    Parses a :py:class:`.WeakFormulation` that has been derived by projecting a
    partial differential equation an a set of test-functions. Within this
    process, the separating approximation
    :math:`x^n(z, t) = \sum_{i=1}^n c_i^n(t) \varphi_i^n(z)` is plugged into
    the equation and the separated spatial terms are evaluated, leading to a
    ordinary equation system for the weights :math:`c_i^n(t)`.

    Args:
        weak_form: Weak formulation of the pde.
        finalize (bool): Default: False. If you have already defined the
            dominant labels of the weak formulations you can set this to True.
            See :py:meth:`.CanonicalEquation.finalize`


    Return:
        :py:class:`.CanonicalEquation`: The spatially approximated equation in
        a canonical form.
    """

    if not isinstance(weak_form, WeakFormulation):
        raise TypeError("Only able to parse WeakFormulation")

    ce = CanonicalEquation(weak_form.name, weak_form.dominant_lbl)

    # handle each term
    for term in weak_form.terms:
        # extract Placeholders
        placeholders = dict(
            scalars=term.arg.get_arg_by_class(Scalars),
            functions=term.arg.get_arg_by_class(TestFunction),
            field_variables=term.arg.get_arg_by_class(FieldVariable),
            observer_fb=term.arg.get_arg_by_class(ObserverGain),
            inputs=term.arg.get_arg_by_class(Input))

        if is_observer:
            if placeholders["observer_fb"]:
                raise ValueError(
                    "The weak formulation for an observer gain can not hold \n"
                    "the 'Placeholder' ObserverGain.")
            if placeholders["field_variables"]:
                raise ValueError(
                    "The weak formulation for an observer gain can not hold \n"
                    "the 'Placeholder' FieldVariable.")
            if placeholders["scalars"]:
                if any([plh.target_term["name"] == 'E'
                        for plh in placeholders["scalars"]]):
                    raise ValueError(
                        "The weak formulation for an observer gain can not \n"
                        "hold a 'Placeholder' Scalars with target_term == 'E'.")

        # field variable terms: sort into E_np, E_n-1p, ..., E_0p
        if placeholders["field_variables"]:
            assert isinstance(term, IntegralTerm)

            if len(placeholders["field_variables"]) != 1:
                raise NotImplementedError

            field_var = placeholders["field_variables"][0]
            if not field_var.simulation_compliant:
                msg = "Shape- and test-function labels of FieldVariable must " \
                      "match for simulation purposes."
                raise ValueError(msg)

            temp_order = field_var.order[0]
            exponent = field_var.data["exponent"]
            term_info = dict(name="E", order=temp_order, exponent=exponent)
            base = get_base(field_var.data["func_lbl"]).derive(field_var.order[1])
            shape_funcs = base.raise_to(exponent)

            if placeholders["inputs"]:
                # essentially, this means that parts of the state-transition
                # matrix will be time dependent
                raise NotImplementedError

            if placeholders["functions"]:
                # is the integrand a product?
                if len(placeholders["functions"]) != 1:
                    raise NotImplementedError
                func1 = placeholders["functions"][0]
                base1 = get_base(func1.data["func_lbl"]).derive(func1.order[1])
                result = calculate_scalar_product_matrix(base1, shape_funcs)
            else:
                # extract constant term and compute integral
                part1 = []
                for func1 in shape_funcs.fractions:
                    from pyinduct.core import ComposedFunctionVector
                    if isinstance(func1, ComposedFunctionVector):
                        res = 0
                        for f in func1.members["funcs"]:
                            area = domain_intersection(term.limits, f.nonzero)
                            r, err = integrate_function(f, area)
                            res += r
                        for s in func1.members["scalars"]:
                            res += s
                    else:
                        area = domain_intersection(term.limits, func1.nonzero)
                        res, err = integrate_function(func1, area)
                    part1.append(res)

                a = Scalars(np.atleast_2d(part1))

                if placeholders["scalars"]:
                    b = placeholders["scalars"][0]
                    result = _compute_product_of_scalars([a, b])
                else:
                    result = a.data

            ce.add_to(weight_label=field_var.data["weight_lbl"],
                      term=term_info,
                      val=result * term.scale)
            continue

        # TestFunctions or pre evaluated terms, those can end up in E, f or G
        if placeholders["functions"]:
            if not 1 <= len(placeholders["functions"]) <= 2:
                raise NotImplementedError
            func1 = placeholders["functions"][0]
            base1 = get_base(func1.data["func_lbl"]).derive(func1.order[1])
            prod = base1.scalar_product_hint()

            if len(placeholders["functions"]) == 1:
                # product of one function and something else, solve integral
                # first by faking 2nd factor
                base2 = [f.mul_neutral_element() for f in base1]
            else:
                func2 = placeholders["functions"][1]
                base2 = get_base(func2.data["func_lbl"]).derive(func2.order[1])

            # resolve equation term
            if isinstance(term, ScalarProductTerm):
                int_res = vectorize_scalar_product(base1, base2, prod)
            elif isinstance(term, IntegralTerm):
                from pyinduct.core import Base, ComposedFunctionVector
                # create base with multiplied fractions
                s_base = Base([f1.scale(f2) for f1, f2 in zip(base1, base2)])

                int_res = []
                for frac in s_base:
                    # WARN I don't think that this case actually makes sense.
                    if isinstance(frac, ComposedFunctionVector):
                        res = 0
                        for f in frac.members["funcs"]:
                            area = domain_intersection(term.limits, f.nonzero)
                            r, err = integrate_function(f, area)
                            res += r
                        for s in frac.members["scalars"]:
                            res += s
                    else:
                        area = domain_intersection(term.limits, frac.nonzero)
                        res, err = integrate_function(frac, area)
                    int_res.append(res)
            else:
                raise NotImplementedError()

            # create column vector
            int_res = np.atleast_2d(int_res).T * term.scale

            # integral of the product of two functions
            if len(placeholders["functions"]) == 2:
                term_info = dict(name="f", exponent=0)
                ce.add_to(weight_label=None,
                          term=term_info, val=int_res)
                continue

            if placeholders["scalars"]:
                a = placeholders["scalars"][0]
                b = Scalars(int_res)
                result = _compute_product_of_scalars([a, b])
                ce.add_to(weight_label=a.target_form,
                          term=get_common_target(placeholders["scalars"]),
                          val=result)
                continue

            if placeholders["inputs"]:
                if len(placeholders["inputs"]) != 1:
                    raise NotImplementedError
                input_var = placeholders["inputs"][0]
                input_func = input_var.data["input"]
                input_index = input_var.data["index"]
                input_exp = input_var.data["exponent"]
                input_order = input_var.order[0]
                term_info = dict(name="G", order=input_order, exponent=input_exp)

                ce.add_to(weight_label=None,
                          term=term_info,
                          val=int_res,
                          column=input_index)
                ce.set_input_function(input_func)
                continue

            if is_observer:
                result = np.vstack([integrate_function(func, func.nonzero)[0]
                                    for func in base1])
                ce.add_to(weight_label=func1.data["appr_lbl"],
                          term=dict(name="E", order=0, exponent=1),
                          val=result * term.scale)
                continue

        # pure scalar terms, sort into corresponding matrices
        if placeholders["scalars"]:
            assert isinstance(term, ScalarTerm)

            result = _compute_product_of_scalars(placeholders["scalars"])
            target = get_common_target(placeholders["scalars"])
            target_form = get_common_form(placeholders)

            if placeholders["inputs"]:
                input_var = placeholders["inputs"][0]
                input_func = input_var.data["input"]
                input_index = input_var.data["index"]
                input_exp = input_var.data["exponent"]
                input_order = input_var.order[0]

                term_info = dict(name="G",
                                 order=input_order,
                                 exponent=input_exp)

                if target["name"] == "E":
                    # this would mean that the input term should appear in a
                    # matrix like E1 or E2, again leading to a time dependant
                    # state transition matrix
                    raise NotImplementedError

                ce.add_to(weight_label=None, term=term_info,
                          val=result * term.scale, column=input_index)
                ce.set_input_function(input_func)
                continue

            if is_observer:
                ce.add_to(
                    weight_label=placeholders["scalars"][0].target_term["test_appr_lbl"],
                    term=dict(name="E", order=0, exponent=1),
                    val=result * term.scale)
            else:
                ce.add_to(weight_label=target_form, term=target, val=result * term.scale)
            continue

        if placeholders["observer_fb"]:
            ce.add_to(weight_label=None,
                      term=dict(name="L"),
                      val=placeholders["observer_fb"][0].data["obs_fb"])
            continue

    # inform object that the parsing process is complete
    if finalize:
        ce.finalize()

    return ce


def parse_weak_formulations(weak_forms):
    """
    Convenience wrapper for :py:func:`.parse_weak_formulation`.

    Args:
        weak_forms: List of :py:class:`.WeakFormulation`'s.

    Returns:
        List of :py:class:`.CanonicalEquation`'s.
    """
    canonical_equations = list()
    for form in weak_forms:
        print(">>> parse formulation {}".format(form.name))
        ce = parse_weak_formulation(form)
        if ce.name in [ceq.name for ceq in canonical_equations]:
            raise ValueError(("Name {} for CanonicalEquation already assigned, "
                              "names must be unique.").format(form.name))
        canonical_equations.append(ce)

    return canonical_equations


def _compute_product_of_scalars(scalars):
    """
    Compute products for scalar terms while paying attention to some  caveats

    Depending on how the data (coefficients for the lumped equations) of the
    terms were generated, it is either a column or a row vector.
    Special cases contain a simple scaling of all equations shape = (1, 1)
    and products of row and column vectors if two terms are provided.

    Args:
        scalars:

    Returns:

    """
    data_shape1 = scalars[0].data.shape
    if len(scalars) < 1 or len(scalars) > 2:
        raise NotImplementedError()
    if len(scalars) == 1:
        # simple scaling of all terms
        if sum(data_shape1) > (max(data_shape1) + 1):
            # print("Workaround 1: Summing up all entries")
            res = np.sum(scalars[0].data, axis=0, keepdims=True).T
        else:
            assert data_shape1[0] == 1 or data_shape1[1] == 1
            res = scalars[0].data
        return res

    # two arguments
    data_shape2 = scalars[1].data.shape
    if data_shape1 == data_shape2 and data_shape2[1] == 1:
        # element wise multiplication
        res = np.prod(np.array([scalars[0].data, scalars[1].data]), axis=0)
    elif data_shape1 == (1, 1) or data_shape2 == (1, 1):
        # a lumped term is present
        res = scalars[0].data * scalars[1].data
    else:
        # dyadic product
        try:
            if data_shape1[1] == 1:
                res = scalars[0].data @ scalars[1].data
            elif data_shape2[1] == 1:
                res = scalars[1].data @ scalars[0].data
            # TODO: handle dyadic product ComposedFunctionVector and Base in the same way
            elif data_shape1[1] == data_shape2[0]:
                # print("Workaround 2: Matrix product")
                res = np.transpose(scalars[1].data) @ np.transpose(scalars[0].data)
            else:
                raise NotImplementedError
        except ValueError as e:
            raise ValueError("provided entries do not form a dyadic product")

    return res


def simulate_state_space(state_space, initial_state, temp_domain, settings=None):
    r"""
    Wrapper to simulate a system given in state space form:

    .. math:: \dot{q} = A_pq^p + A_{p-1}q^{p-1} + \dotsb + A_0q + Bu.

    Args:
        state_space (:py:class:`.StateSpace`): State space formulation of the
            system.
        initial_state: Initial state vector of the system.
        temp_domain (:py:class:`.Domain`): Temporal domain object.
        settings (dict): Parameters to pass to the :py:func:`set_integrator`
            method of the :class:`scipy.ode` class, with the integrator name
            included under the key :obj:`name`.

    Return:
        tuple: Time :py:class:`.Domain` object and weights matrix.
    """
    # if not isinstance(state_space, StateSpace):
    #     raise TypeError

    q = [initial_state]
    t = [temp_domain[0]]

    r = ode(state_space.rhs)

    # TODO check for complex-valued matrices and use 'zvode'
    if settings:
        r.set_integrator(settings.pop("name"), **settings)
    else:
        # use some sane defaults
        r.set_integrator(
            "vode",
            max_step=temp_domain.step,
            method="adams",
            nsteps=1e3
        )

    r.set_initial_value(q[0], t[0])

    for t_step in temp_domain[1:]:
        qn = r.integrate(t_step)
        if not r.successful():
            warnings.warn("*** Error: Simulation aborted at t={} ***".format(r.t))
            break

        t.append(r.t)
        q.append(qn)

    # create results
    q = np.array(q)

    return Domain(points=np.array(t), step=temp_domain.step), q


def evaluate_approximation(base_label, weights, temp_domain, spat_domain, spat_order=0, name=""):
    """
    Evaluate an approximation given by weights and functions at the points given
    in spatial and temporal steps.

    Args:
        weights: 2d np.ndarray where axis 1 is the weight index and axis 0 the
            temporal index.
        base_label (str): Functions to use for back-projection.
        temp_domain (:py:class:`.Domain`): For steps to evaluate at.
        spat_domain (:py:class:`.Domain`): For points to evaluate at (or in).
        spat_order: Spatial derivative order to use.
        name: Name to use.

    Return:
        :py:class:`.EvalData`
    """
    funcs = get_base(base_label).derive(spat_order).fractions
    if weights.shape[1] != funcs.shape[0]:
        raise ValueError("weights (len={0}) have to fit provided functions "
                         "(len={1})!".format(weights.shape[1], funcs.size))

    # evaluate shape functions at given points
    shape_vals = np.array([func.evaluation_hint(spat_domain)
                           for func in funcs]).T

    if shape_vals.ndim == 2:
        res = weights @ shape_vals.T
    else:
        # get extra dims to the front in both arrays
        extra_axes = range(1, shape_vals.ndim - 1)
        axes_idxs = np.array(extra_axes)
        b_shape_vals = np.swapaxes(shape_vals, 0, -1)
        b_shape_vals = np.moveaxis(b_shape_vals, axes_idxs, axes_idxs-1)
        w_shape = (*np.array(shape_vals.shape)[axes_idxs], *weights.shape)
        b_weights = np.broadcast_to(weights, w_shape)
        b_res = b_weights @ b_shape_vals
        res = np.moveaxis(b_res, axes_idxs-1, axes_idxs+1)

    ed = EvalData([temp_domain.points, spat_domain.points], res,
                  name=name, fill_axes=True)
    return ed


def set_dominant_labels(canonical_equations, finalize=True):
    """
    Set the dominant label (*dominant_lbl*) member of all given canonical
    equations and check if the problem formulation is valid (see background
    section: http://pyinduct.readthedocs.io/en/latest/).

    If the dominant label of one or more :py:class:`.CanonicalEquation`
    is already defined, the function raise a UserWarning if the (pre)defined
    dominant label(s) are not valid.

    Args:
        canonical_equations: List of :py:class:`.CanonicalEquation` instances.
        finalize (bool): Finalize the equations? Default: True.
    """
    if isinstance(canonical_equations, CanonicalEquation):
        canonical_equations = [canonical_equations]

    # collect all involved labels
    labels = set(
        chain(*[list(ce.dynamic_forms.keys()) for ce in canonical_equations]))

    if len(labels) != len(canonical_equations):
        raise ValueError("The N defined canonical equations (weak forms)\n"
                         "must hold exactly N different weight labels!\n"
                         "But your {} canonical equation(s) (weak form(s))\n"
                         "hold {} weight label(s)!"
                         "".format(len(canonical_equations),
                                   len(labels)))

    max_orders = dict()
    for ce in canonical_equations:
        ce.finalize_dynamic_forms()
        for lbl in list(ce.dynamic_forms.keys()):
            max_order = dict(
                (("max_order", ce.dynamic_forms[lbl].max_temp_order),
                 ("can_eqs", [ce])))
            if lbl not in max_orders or \
                max_orders[lbl]["max_order"] < max_order["max_order"]:
                max_orders[lbl] = max_order
            elif max_orders[lbl]["max_order"] == max_order["max_order"]:
                max_orders[lbl]["can_eqs"].append(
                    max_order["can_eqs"][0])

    non_valid1 = [(lbl, max_orders[lbl])
                  for lbl in labels if len(max_orders[lbl]["can_eqs"]) > 1]
    if non_valid1:
        raise ValueError("The highest time derivative from a certain weight\n"
                         "label may only occur in one canonical equation. But\n"
                         "each of the canonical equations {} holds the\n"
                         "weight label '{}' with order {} in time."
                         "".format(non_valid1[0][1]["can_eqs"][0].name,
                                   non_valid1[0][0],
                                   non_valid1[0][1]["max_order"]))

    non_valid2 = [lbl for lbl in labels if max_orders[lbl]["max_order"] == 0]
    if non_valid2:
        raise ValueError("The defined problem leads to an differential\n"
                         "algebraic equation, since there is no time\n"
                         "derivative for the weights {}. Such problems are\n"
                         "not considered in pyinduct, yet."
                         "".format(non_valid2))

    # set/check dominant labels
    for lbl in labels:
        pre_lbl = max_orders[lbl]["can_eqs"][0].dominant_lbl
        max_orders[lbl]["can_eqs"][0].dominant_lbl = lbl

        if  pre_lbl is not None and pre_lbl != lbl:
            warnings.warn("\n Predefined dominant label '{}' from\n"
                          "canonical equation / weak form '{}' not valid!\n"
                          "It will be overwritten with the label '{}'."
                          "".format(pre_lbl,
                                    max_orders[lbl]["can_eqs"][0].name,
                                    lbl),
                          UserWarning)

    if finalize:
        for ce in canonical_equations:
            ce.finalize()


class SimulationInputVector(SimulationInput):
    """
    A simulation input which combines :py:class:`.SimulationInput` objects into
    a column vector.

    Args:
        input_vector (array_like): Simulation inputs to stack.
    """

    def __init__(self, input_vector):
        SimulationInput.__init__(self)
        self._input_vector = self._sanitize_input_vector(input_vector)

    def _sanitize_input_vector(self, input_vector):
        if hasattr(input_vector, "__len__") and len(input_vector) == 0:
            return list()
        else:
            return sanitize_input(input_vector, SimulationInput)

    def __iter__(self):
        return iter(self._input_vector)

    def __getitem__(self, item):
        return self._input_vector[item]

    def append(self, input_vector):
        """
        Add an input to the vector.
        """
        inputs = self._sanitize_input_vector(input_vector)
        self._input_vector = np.hstack((self._input_vector, inputs))

    def _calc_output(self, **kwargs):
        output = list()
        for input in self._input_vector:
            output.append(input(**kwargs))

        return dict(output=output)

