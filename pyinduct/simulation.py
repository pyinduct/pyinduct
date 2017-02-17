"""
Simulation infrastructure with helpers and data structures for preprocessing of the given equations
and functions for postprocessing of simulation data.
"""

import warnings
from abc import ABCMeta, abstractmethod
from collections import OrderedDict
from copy import copy
from itertools import chain

import numpy as np
from scipy.integrate import ode
from scipy.interpolate import interp1d
from scipy.linalg import block_diag

from .core import (Domain, Parameters, Function, integrate_function, calculate_scalar_product_matrix,
                   project_on_base, dot_product_l2, sanitize_input, StackedBase, TransformationInfo,
                   get_weight_transformation, EvalData)
from .placeholder import Scalars, TestFunction, Input, FieldVariable, EquationTerm, get_common_target
from .registry import get_base, register_base

__all__ = ["SimulationInput", "SimulationInputSum", "WeakFormulation", "parse_weak_formulation",
           "create_state_space", "StateSpace", "simulate_state_space", "simulate_system", "simulate_systems",
           "process_sim_data", "evaluate_approximation"]


class SimulationInput(object, metaclass=ABCMeta):
    """
    Base class for all objects that want to act as an input for the time-step simulation.

    The calculated values for each time-step are stored in internal memory and can be accessed by
    :py:func:`get_results` (after the simulation is finished).

    Note:
        Due to the underlying solver, this handle may get called with time arguments, that lie outside of the specified
        integration domain. This should not be a problem for a feedback controller but might cause problems for a
        feedforward or trajectory implementation.
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
            entries.append(value)
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

    def get_results(self, time_steps, result_key="output", interpolation="nearest", as_eval_data=False):
        """
        Return results from internal storage for given time steps.

        Raises:
            Error: If calling this method before a simulation was run.

        Args:
            time_steps: Time points where values are demanded.
            result_key: Type of values to be returned.
            interpolation: Interpolation method to use if demanded time-steps are not covered by the storage,
                see :func:`scipy.interpolate.interp1d` for all possibilities.
            as_eval_data (bool): Return results as :py:class:`pyinduct.visualization.EvalData`
                object for straightforward display.

        Return:
            Corresponding function values to the given time steps.
        """
        func = interp1d(np.array(self._time_storage), np.array(self._value_storage[result_key]),
                        kind=interpolation, assume_sorted=True, axis=0)
        values = func(time_steps)

        if as_eval_data:
            return EvalData([time_steps], values, name=".".join([self.name, result_key]))

        return values


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


class SimulationInputVector(SimulationInput):
    """
    Class that represent the input vector :math:`\\boldsymbol{u}\\in\\mathbb R^n` from a state space
    system (:py:class:`StateSpace`) like

    .. math::
        \\boldsymbol{\\dot{x}}(t) &= \\boldsymbol{A}\\boldsymbol{x}(t) + \\boldsymbol{B}\\boldsymbol{u}(t) \\\\
        \\boldsymbol{y}(t) &= \\boldsymbol{C}\\boldsymbol{x}(t) + \\boldsymbol{D}\\boldsymbol{u}(t).

    Args:
        input_vector (list): List which holds (in sum) :math:`n` :py:class:`SimulationInput` and/or
            :py:class:`ObserverError` instances.
    """

    def __init__(self, input_vector):
        SimulationInput.__init__(self)
        if not all([isinstance(input, SimulationInput) and not isinstance(input, SimulationInputVector)
                    for input in input_vector]):
            raise TypeError("A SimulationInputVector can only hold SimulationInputs's and can not nest.")

        self.input_vector = input_vector
        self.len = len(input_vector)
        self.indices = np.arange(self.len)
        self._sys_inputs = collections.OrderedDict()
        self._obs_errors = collections.OrderedDict()
        for index in self.indices:
            if isinstance(input_vector[index], ObserverError):
                self._obs_errors[index] = input_vector[index]
            else:
                self._sys_inputs[index] = input_vector[index]

    def _calc_output(self, **kwargs):
        output = list()
        if "obs_weights" in kwargs.keys():
            for obs_err in self._obs_errors.values():
                output.append(obs_err(**kwargs))
        else:
            for sys_input in self._sys_inputs.values():
                output.append(sys_input(**kwargs))

        return dict(output=np.matrix(np.vstack(output)))


class WeakFormulation(object):
    """
    This class represents the weak formulation of a spatial problem.
    It can be initialized with several terms (see children of :py:class:`pyinduct.placeholder.EquationTerm`).
    The equation is interpreted as

    .. math:: term_0 + term_1 + ... + term_N = 0.

    Args:
        terms (list): List of object(s) of type EquationTerm.
        name (string): name of this weak form
    """

    def __init__(self, terms, name):
        self.terms = sanitize_input(terms, EquationTerm)
        self.name = name


class StateSpace(object):
    r"""
    Wrapper class that represents the state space form of a dynamic system where

    .. math::
        \boldsymbol{\dot{x}}(t) &= \sum\limits_{k=0}^{L}\boldsymbol{A}_{k} \boldsymbol{x}^{p_k}(t)
        + \sum\limits_{j=0}^{V} \sum\limits_{k=0}^{L}\boldsymbol{B}_{j, k} \frac{\mathrm{d}^j u^{p_k}}{\mathrm{d}t^j}(t) \\
        \boldsymbol{y}(t) &= \boldsymbol{C}\boldsymbol{x}(t) + \boldsymbol{D}u(t)

    which has been approximated by projection on a base given by weight_label.

    Args:
        a_matrices (dict): State transition matrices :math:`\boldsymbol{A}_{p_k}`
            for the corresponding powers of :math:`\boldsymbol{x}`
        b_matrices (dict): Cascaded dictionary for the input matrices :math:`\boldsymbol{B}_{j, k}` in the sequence:
            temporal derivative order, exponent .
        input_handle:  function handle, returning the system input :math:`u(t)`
        c_matrix: :math:`\boldsymbol{C}`
        d_matrix: :math:`\boldsymbol{D}`
    """

    def __init__(self, a_matrices, b_matrices, base_lbl=None,
                 input_handle=None, c_matrix=None, d_matrix=None):
        self.C = c_matrix
        self.D = d_matrix
        self.base_lbl = base_lbl

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

        self.input = input_handle
        if self.input is None:
            self.input = EmptyInput(self.B[0][available_power].shape[1])

        if not callable(self.input):
            raise TypeError("input must be callable!")

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
        q_t = self.A[0]
        for p, a_mat in self.A.items():
            q_t = q_t + a_mat @ np.power(_q, p)

        # TODO make compliant with definition of temporal derived input
        u = self.input(time=_t, weights=_q, weight_lbl=self.base_lbl)
        for o, p_mats in self.B.items():
            for p, b_mat in p_mats.items():
                # q_t = q_t + (b_mat @ np.power(u, p)).flatten()
                q_t = q_t + b_mat @ np.power(u, p)

        return q_t


def simulate_system(weak_form, initial_states,
                    temporal_domain, spatial_domain,
                    derivative_orders=(0, 0), settings=None):
    """
    Convenience wrapper for :py:func:`simulate_systems`.

    Args:
        weak_form (:py:class:`WeakFormulation`): Weak formulation of the system to simulate.
        initial_states (numpy.ndarray): Array of core.Functions for
            :math:`x(t=0, z), \\dot{x}(t=0, z), \\dotsc, x^{(n)}(t=0, z)`.
        temporal_domain (:py:class:`core.Domain`): Domain object holding information for time evaluation.
        spatial_domain (:py:class:`core.Domain`): Domain object holding information for spatial evaluation.
        derivative_orders (tuple): tuples of derivative orders (time, spat) that shall be
            evaluated additionally as values
        settings: Integrator settings, see :py:func:`simulate_state_space`.
    """
    ics = sanitize_input(initial_states, Function)
    initial_states = {weak_form.name: ics}
    spatial_domains = {weak_form.name: spatial_domain}
    derivative_orders = {weak_form.name: derivative_orders}
    res = simulate_systems([weak_form], initial_states, temporal_domain, spatial_domains, derivative_orders, settings)
    return res


def simulate_systems(weak_forms, initial_states, temporal_domain, spatial_domains, derivative_orders=None,
                     settings=None):
    """
    Convenience wrapper that encapsulates the whole simulation process.

    Args:
        weak_forms ((list of) :py:class:`WeakFormulation`): (list of) Weak formulation(s) of the system(s) to simulate.
        initial_states (dict, numpy.ndarray): Array of core.Functions for
            :math:`x(t=0, z), \\dot{x}(t=0, z), \\dotsc, x^{(n)}(t=0, z)`.
        temporal_domain (:py:class:`core.Domain`): Domain object holding information for time evaluation.
        spatial_domains (dict) Dict with :py:class:`Domain` objects holding information for spatial evaluation.
        derivative_orders (dict): Dict, containing tuples of derivative orders (time, spat) that shall be
            evaluated additionally as values
        settings: Integrator settings, see :py:func:`simulate_state_space`.

    Note:
        The *name* attributes of the given weak forms must be unique!

    Return:
        list: List of :py:class:`pyinduct.visualization.EvalData` objects, holding the results for the FieldVariable
        and demanded derivatives.
    """
    weak_forms = sanitize_input(weak_forms, WeakFormulation)
    print("simulating systems: {}".format([f.name for f in weak_forms]))

    # parse input and create state space system
    canonical_equations = OrderedDict()
    for form in weak_forms:
        print(">>> parsing formulation {}".format(form.name))
        if form.name in canonical_equations:
            raise ValueError(("Name {} for CanonicalEquation already assigned,"
                              + "names must be unique.").format(form.name))
        canonical_equations.update({form.name: parse_weak_formulation(form)})

    print(">>> creating state space system")
    state_space_form = create_state_space(canonical_equations)

    # calculate initial state, assuming it will be constituted by the
    # dominant systems
    print(">>> deriving initial conditions")
    q0 = np.array([])
    for form in weak_forms:
        lbl = canonical_equations[form.name].dominant_lbl
        q0 = np.hstack(tuple([q0] + [project_on_base(initial_state, get_base(lbl))
                                     for initial_state in initial_states[form.name]]))

    # simulate
    print(">>> performing time step integration")
    sim_domain, q = simulate_state_space(state_space_form, q0, temporal_domain, settings=settings)

    # evaluate
    print(">>> performing postprocessing")
    results = []
    for form in weak_forms:
        # acquire a transformation into the original weights
        info = TransformationInfo()
        info.src_lbl = state_space_form.base_lbl
        info.src_base = get_base(info.src_lbl)
        info.src_order = initial_states[form.name].size - 1
        info.dst_lbl = canonical_equations[form.name].dominant_lbl
        info.dst_base = get_base(info.dst_lbl)
        info.dst_order = derivative_orders[form.name][0]
        transformation = get_weight_transformation(info)

        # project back
        data = process_sim_data(info.dst_lbl,
                                np.apply_along_axis(transformation, 1, q),
                                sim_domain,
                                spatial_domains[form.name],
                                info.dst_order,
                                derivative_orders[form.name][1],
                                name=form.name)
        results += data

    print("finished simulation.")
    return results


def process_sim_data(weight_lbl, q, temp_domain, spat_domain, temp_order, spat_order, name=""):
    """
    Create handles and evaluate at given points.

    Args:
        weight_lbl (str): Label of Basis for reconstruction.
        temp_order: Order or temporal derivatives to evaluate additionally.
        spat_order: Order or spatial derivatives to evaluate additionally.
        q: weights
        spat_domain (:py:class:`core.Domain`): Domain object providing values for spatial evaluation.
        temp_domain (:py:class:`core.Domain`): Time steps on which rows of q are given.
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

    @input_function.setter
    def input_function(self, func):
        if self._input_function is None:
            self._input_function = func
        if self._input_function != func:
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
        if not isinstance(value, np.ndarray):
            raise TypeError("val must be numpy.ndarray")
        if column and not isinstance(column, int):
            raise TypeError("column index must be int")

        # get entry
        if term["name"] == "f":
            if ("order" in term) or ("exponent" in term and term["exponent"] is not 0):
                warnings.warn("order and exponent are ignored for f_vector!")
            f_vector = self.matrices.get("f", np.zeros_like(value))
            self.matrices["f"] = value + f_vector
            return

        type_group = self.matrices.get(term["name"], {})
        derivative_group = type_group.get(term["order"], {})
        target_matrix = derivative_group.get(term["exponent"], np.zeros_like(value)).astype(complex)

        if target_matrix.shape != value.shape and column is None:
            raise ValueError("{0}{1}{2} was already initialized with dimensions {3} but value to add has "
                             "dimension {4}".format(term["name"], term["order"], term["exponent"],
                                                    target_matrix.shape, value.shape))

        if column is not None:
            # check whether the dimensions fit or if the matrix has to be extended
            if column >= target_matrix.shape[1]:
                new_target_matrix = np.zeros((target_matrix.shape[0], column + 1)).astype(complex)
                new_target_matrix[:target_matrix.shape[0], :target_matrix.shape[1]] = target_matrix
                target_matrix = new_target_matrix

            target_matrix[:, column:column + 1] += value
        else:
            target_matrix += value

        # store changes
        derivative_group[term["exponent"]] = np.real_if_close(target_matrix)
        type_group[term["order"]] = derivative_group
        self.matrices[term["name"]] = type_group

    def finalize(self):
        """
        Finalizes the object.
        This method must be called after all terms have been added by :py:func:`add_to` and before
        :py:func:`convert_to_state_space` can be called. This functions makes sure that the formulation can be converted
        into state space form (highest time derivative only comes in one power) and collects information like highest
        derivative order, it's power and the sizes of current and state-space state vector (`dim_x` resp. `dim_xb`).
        Furthermore, the coefficient matrix of the highest derivative order `e_n_pb` and it's inverse are made
        accessible.
        """
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
        Convert the canonical ode system of order n a into an ode system of order 1.

        Note:
            This will only work if the highest derivative order of the given form can be isolated. This is the case if
            the highest order is only present in one power and the equation system can therefore be solved for it.

        Return:
            :py:class:`StateSpace` object:
        """
        if not self._finalized:
            self.finalize()

        # system matrices A_*
        a_matrices = {}
        for p in self.powers:
            a_mat = np.zeros((self.dim_xb, self.dim_xb))

            # add integrator chain
            a_mat[:-self.dim_x:, self.dim_x:] = block_diag(*[np.eye(self.dim_x)
                                                             for a in range(self.max_temp_order - 1)])

            # add "block-line" with feedback entries
            a_mat[-self.dim_x:, :] = -self._build_feedback("E", p, self.e_n_pb_inv)
            a_matrices.update({p: a_mat})

        # input matrices B_*
        if "G" in self.matrices:
            max_temp_input_order = max(iter(self.matrices["G"]))
            input_powers = set(chain.from_iterable([list(mat) for mat in self.matrices["G"].values()]))
            dim_u = next(iter(self.matrices["G"][max_temp_input_order].values())).shape[1]

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
                        b_mat[-self.dim_x:, :] = - self.e_n_pb_inv @ self.matrices["G"][order][q]
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

        blocks = (np.dot(product_mat, self.matrices[entry].get(order, {}).get(power, np.zeros(entry_shape)))
                  for order in range(max_order))
        return np.hstack(blocks)


class CanonicalEquation(object):
    """
    Wrapper object, holding several entities of canonical forms for different weight-sets that form an equation when
    summed up.
    After instantiation, this object can be filled with information by passing the corresponding coefficients to
    :py:func:`add_to` . When the parsing process is completed and all coefficients have been collected, calling
    :py:func:`finalize` is required to compute all necessary information for further processing.
    When finalized, this object provides access to the dominant form of this equation.
    """

    def __init__(self, name):
        self.name = name
        self.dominant_lbl = None
        self.dynamic_forms = {}
        self._static_form = CanonicalForm(self.name + "_static")
        self._finalized = False

    def add_to(self, weight_label, term, val, column=None):
        """
        Add the provided *val* to the canonical form for *weight_label*, see :py:func:`CanonicalForm.add_to` for further
        information.

        Args:
            weight_label (str): Basis to add onto.
            term: Coefficient to add onto, see :py:func:`CanonicalForm.add_to`.
            val: Values to add.
            column (int): passed to :py:func:`CanonicalForm.add_to`.
        """
        if self._finalized:
            raise RuntimeError("Object has already been finalized, you are trying some nasty stuff there.")

        if term["name"] in "fG":
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
        After the complete formulation has been parsed and all terms have been sorted into this Object via
        :py:func:`add_to` this function has to be called to inform this object about it.
        When invoked, the :py:class:`CanonicalForm` that holds the highest temporal derivative order will be marked
        as dominant and can be accessed via :py:attr:`dominant_form`.
        Furthermore, the f and G parts of the static_form will be copied to the dominant form for easier
        state-space transformation.

        Raises:
            RuntimeError: If two different forms provide the highest derivative orders

        Note:
            This function must be called to use the :py:attr:`dominant_form` attribute.

        """
        highest_dict = {}
        highest_list = []
        # highest_orders = [(key, val.max_temp_order) for key, val in self._dynamic_forms]
        for lbl, form in self.dynamic_forms.items():
            # finalize dynamic forms
            form.finalize()
            # extract maximum derivative orders
            highest_dict[lbl] = form.max_temp_order
            highest_list.append(form.max_temp_order)

        max_order = max(highest_list)
        highest_list.remove(max_order)
        if max_order in highest_list:
            raise ValueError("Highest derivative order cannot be isolated.")

        self.dominant_lbl = next((label for label, order in highest_dict.items() if order == max_order), None)
        if self.dynamic_forms[self.dominant_lbl].singular:
            raise ValueError("The form that has to be chosen is singular.")

        # copy static terms to dominant form to transform them correctly
        for letter in "fG":
            if letter in self._static_form.matrices:
                self.dynamic_forms[self.dominant_lbl].matrices.update({letter: self._static_form.matrices[letter]})

        self._finalized = True

    @property
    def static_form(self):
        """
        :py:class:`WeakForm` that does not depend on any weights.
        :return:
        """
        return self._static_form

    @property
    def dominant_form(self):
        """
        direct access to the dominant :py:class:`CanonicalForm`.

        Note:
            :py:func:`finalize` must be called first.

        Returns:
            :py:class:`CanonicalForm`: the dominant canonical form
        """
        if not self._finalized:
            raise RuntimeError("Object has not yet been finalized!")
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
        The input handle for the equation.
        """
        return self._static_form.input_function

    @input_function.setter
    def input_function(self, func):
        self._static_form.input_function = func


def create_state_space(canonical_equations):
    """
    Create a state-space system constituted by several :py:class:`CanonicalEquations`
    (created by :py:func:`parse_weak_formulation`)

    Args:
        canonical_equations (:py:class:`CanonicalEquation` or dict): dict of name: py:class:`CanonicalEquation` pairs

    Raises:
        ValueError: If compatibility criteria cannot be fulfilled

    Return:
        :py:class:`StateSpace`: State-space representation of the approximated system,
    """
    if isinstance(canonical_equations, CanonicalEquation):
        # backward compatibility
        canonical_equations = dict(default=canonical_equations)

    # check whether the formulations are compatible
    for name, eq in canonical_equations.items():
        for lbl, form in eq.dynamic_forms.items():
            coupling_order = form.max_temp_order

            # search corresponding dominant form in other equations
            for _name, _eq in canonical_equations.items():
                # check uniqueness of name - dom_lbl mappings
                if name != _name and eq.dominant_lbl == _eq.dominant_lbl:
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
    for name, eq in canonical_equations.items():
        dom_lbl = eq.dominant_lbl
        dom_form = eq.dominant_form
        dom_ss = dom_form.convert_to_state_space()
        dominant_state_spaces.update({dom_lbl: dom_ss})

        # collect some information
        state_space_props.parts[dom_lbl] = dict(start=copy(state_space_props.size),
                                                orig_size=dom_form.dim_x,
                                                size=dom_form.dim_xb,
                                                order=dom_form.max_temp_order)
        state_space_props.powers.update(dom_form.powers)
        state_space_props.size += dom_form.dim_xb
        state_space_props.dim_u = max(state_space_props.dim_u, dom_form.dim_u)

        # update input handles
        if state_space_props.input is None:
            state_space_props.input = eq.input_function
        else:
            if eq.input_function is not None and state_space_props.input != eq.input_function:
                raise ValueError("Only one input object allowed.")

    # build new basis by concatenating the dominant bases of every equation
    if len(canonical_equations) == 1:
        new_name = next(iter(canonical_equations.values())).dominant_lbl
    else:
        members = state_space_props.parts.keys()
        new_name = "_".join(members)
        fracs = [frac for lbl in members for frac in get_base(lbl).fractions]
        new_base = StackedBase(fracs, state_space_props.parts)
        register_base(new_name, new_base)

    # build new state transition matrices A_p_k for corresponding powers p_k of the state vector
    a_matrices = {}
    for p in state_space_props.powers:
        a_mat = np.zeros((state_space_props.size, state_space_props.size))
        for row_name, row_eq in canonical_equations.items():
            row_dom_lbl = row_eq.dominant_lbl
            row_dom_dim = state_space_props.parts[row_dom_lbl]["size"]
            row_dom_trans_mat = row_eq.dominant_form.e_n_pb_inv
            row_dom_sys_mat = dominant_state_spaces[row_dom_lbl].A.get(p, None)
            row_idx = state_space_props.parts[row_dom_lbl]["start"]

            for col_name, col_eq in canonical_equations.items():
                col_dom_lbl = col_eq.dominant_lbl

                # main diagonal
                if col_name == row_name:
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

    dom_ss = StateSpace(a_matrices, b_matrices, base_lbl=new_name,
                        input_handle=state_space_props.input)
    return dom_ss


def parse_weak_formulation(weak_form, finalize=True):
    """
    Parses a :py:class:`WeakFormulation` that has been derived by projecting a partial differential equation an a set
        of test-functions. Within this process, the separating approximation :math:`x^n(z, t) = ` is plugged into the
        equation and the separated spatial terms are evaluated, leading to a ordinary equation system for the weights
        :math:`x_i(t)` .

    Args:
        weak_form: Weak formulation of the pde.
        finalize (bool): finalize the generated CanonicalEquation. see :py:func:`CanonicalEquation.finalize()`

    Return:
        :py:class:`CanonicalEquation`: The spatially approximated equation in a canonical form.
    """

    if not isinstance(weak_form, WeakFormulation):
        raise TypeError("Only able to parse WeakFormulation")

    ce = CanonicalEquation(weak_form.name)

    # handle each term
    for term in weak_form.terms:
        # extract Placeholders
        placeholders = dict(scalars=term.arg.get_arg_by_class(Scalars),
                            functions=term.arg.get_arg_by_class(TestFunction),
                            field_variables=term.arg.get_arg_by_class(FieldVariable),
                            inputs=term.arg.get_arg_by_class(Input))

        # field variable terms: sort into E_np, E_n-1p, ..., E_0p
        if placeholders["field_variables"]:
            if len(placeholders["field_variables"]) != 1:
                raise NotImplementedError

            field_var = placeholders["field_variables"][0]
            if not field_var.simulation_compliant:
                raise ValueError("Shape- and test-function labels of FieldVariable must match for simulation purposes.")

            temp_order = field_var.order[0]
            exponent = field_var.data["exponent"]
            term_info = dict(name="E", order=temp_order, exponent=exponent)
            base = get_base(field_var.data["func_lbl"]).derive(field_var.order[1])
            shape_funcs = base.raise_to(exponent)

            if placeholders["inputs"]:
                # essentially, this means that parts of the state-transition matrix will be time dependent
                raise NotImplementedError

            # is the integrand a product?
            if placeholders["functions"]:
                if len(placeholders["functions"]) != 1:
                    raise NotImplementedError
                func = placeholders["functions"][0]
                test_funcs = get_base(func.data["func_lbl"]).derive(func.order[1])
                result = calculate_scalar_product_matrix(dot_product_l2, test_funcs, shape_funcs)
            else:
                # extract constant term and compute integral
                a = Scalars(np.atleast_2d([integrate_function(func, func.nonzero)[0]
                                           for func in shape_funcs.fractions]))

                if placeholders["scalars"]:
                    b = placeholders["scalars"][0]
                else:
                    b = Scalars(np.ones_like(a.data.T))

                result = _compute_product_of_scalars([a, b])

            ce.add_to(weight_label=field_var.data["weight_lbl"], term=term_info, val=result * term.scale)
            continue

        # TestFunctions or pre evaluated terms, those can end up in E, f or G
        if placeholders["functions"]:
            if not 1 <= len(placeholders["functions"]) <= 2:
                raise NotImplementedError
            func = placeholders["functions"][0]
            test_funcs = get_base(func.data["func_lbl"]).derive(func.order[1]).fractions

            if len(placeholders["functions"]) == 2:
                # TODO this computation is nonsense. Result must be a vector containing int(tf1*tf2)
                raise NotImplementedError
                #
                # func2 = placeholders["functions"][1]
                # test_funcs2 = get_base(func2.data["func_lbl"], func2.order[2])
                # result = calculate_scalar_product_matrix(dot_product_l2, test_funcs, test_funcs2)
                # cf.add_to(("f", 0), result * term.scale)
                # continue

            if placeholders["scalars"]:
                a = placeholders["scalars"][0]
                b = Scalars(np.vstack([integrate_function(func, func.nonzero)[0] for func in test_funcs]))
                result = _compute_product_of_scalars([a, b])

                ce.add_to(weight_label=a.target_form,
                          term=get_common_target(placeholders["scalars"]),
                          val=result * term.scale)
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

                result = np.array([[integrate_function(func, func.nonzero)[0]] for func in test_funcs])

                ce.add_to(weight_label=None, term=term_info, val=result * term.scale, column=input_index)
                ce.input_function = input_func
                continue

        # pure scalar terms, sort into corresponding matrices
        if placeholders["scalars"]:
            result = _compute_product_of_scalars(placeholders["scalars"])
            target = get_common_target(placeholders["scalars"])
            target_form = placeholders["scalars"][0].target_form

            if placeholders["inputs"]:
                input_var = placeholders["inputs"][0]
                input_func = input_var.data["input"]
                input_index = input_var.data["index"]
                input_exp = input_var.data["exponent"]
                input_order = input_var.order[0]
                term_info = dict(name="G", order=input_order, exponent=input_exp)

                if input_order > 0:
                    # here we would need to provide derivative handles in the callable
                    raise NotImplementedError

                if target["name"] == "E":
                    # this would mean that the input term should appear in a matrix like E1 or E2
                    # the result would be a time dependant sate transition matrix
                    raise NotImplementedError

                ce.add_to(weight_label=None, term=term_info, val=result * term.scale, column=input_index)
                ce.input_function = input_func
                continue

            ce.add_to(weight_label=target_form, term=target, val=result * term.scale)
            continue

    # inform object that the parsing process is complete
    if finalize:
        ce.finalize()

    return ce


def _compute_product_of_scalars(scalars):
    if len(scalars) > 2:
        raise NotImplementedError

    if len(scalars) == 1:
        # simple scaling of all terms
        res = scalars[0].data
    elif scalars[0].data.shape == scalars[1].data.shape:
        # element wise multiplication
        res = np.prod(np.array([scalars[0].data, scalars[1].data]), axis=0)
    else:
        # dyadic product
        try:
            if scalars[0].data.shape[1] == 1:
                res = scalars[0].data @ scalars[1].data
            else:
                res = scalars[1].data @ scalars[0].data
        except ValueError as e:
            raise ValueError("provided entries do not form a dyadic product" + e.msg)

    return res


def simulate_state_space(state_space, initial_state, temp_domain, settings=None):
    """
    Wrapper to simulate a system given in state space form:

    .. math:: \\dot{q} = A_pq^p + A_{p-1}q^{p-1} + \\dotsb + A_0q + Bu.

    Args:
        state_space (:py:class:`StateSpace`): State space formulation of the system.
        initial_state: Initial state vector of the system.
        temp_domain (:py:class:`core.Domain`): Temporal domain object.
        settings (dict): Parameters to pass to the :func:`set_integrator` method of the :class:`scipy.ode` class, with
            the integrator name included under the key :obj:`name`.

    Return:
        tuple: Time :py:class:`Domain` object and weights matrix.
    """
    if not isinstance(state_space, StateSpace):
        raise TypeError

    input_handle = state_space.input

    if not isinstance(input_handle, SimulationInput):
        raise TypeError("only simulation.SimulationInput supported.")

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
    Evaluate an approximation given by weights and functions at the points given in spatial and temporal steps.

    Args:
        weights: 2d np.ndarray where axis 1 is the weight index and axis 0 the temporal index.
        base_label (str): Functions to use for back-projection.
        temp_domain (:py:class:`Domain`): For steps to evaluate at.
        spat_domain (:py:class:`Domain`): For points to evaluate at (or in).
        spat_order: Spatial derivative order to use.
        name: Name to use.

    Return:
        :py:class:`pyinduct.visualization.EvalData`
    """
    funcs = get_base(base_label).derive(spat_order).fractions
    if weights.shape[1] != funcs.shape[0]:
        raise ValueError("weights (len={0}) have to fit provided functions (len={1})!".format(weights.shape[1],
                                                                                              funcs.size))

    # evaluate shape functions at given points
    shape_vals = np.array([func.evaluation_hint(spat_domain) for func in funcs])

    def eval_spatially(weight_vector):
        return np.real_if_close(np.dot(weight_vector, shape_vals), 1000)

    data = np.apply_along_axis(eval_spatially, 1, weights)
    return EvalData([temp_domain.points, spat_domain.points], data, name=name)


"""
Control section
"""


class FeedbackLaw(object):
    """
    This class represents the approximated formulation of a control law or observer error.
    It can be initialized with several terms (see children of :py:class:`pyinduct.placeholder.EquationTerm`).
    The equation is interpreted as

    .. math::
        term_0 + term_1 + ... + term_N = u

    where :math:`u` is the control output.

    Args:
        terms (list): List with object(s) of type :py:class:`pyinduct.placeholder.EquationTerm`.
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


class Feedback(SimulationInput):
    """
    Wrapper class for all state feedbacks that have to interact with the simulation environment.

    Args:
        feedback_law (:py:class:`FeedbackLaw`): Function handle that calculates the state feedback if provided with
            correct weights.
    """

    def __init__(self, feedback_law):
        SimulationInput.__init__(self, name=feedback_law.name)
        c_forms = approximate_feedback_law(feedback_law)
        self._evaluator = LawEvaluator(c_forms, self._value_storage)

    def _calc_output(self, **kwargs):
        """
        Calculates the feedback based on the current_weights.

        Keyword Args:
            weights: Current weights of the simulations system approximation.
            weights_lbl (str): Corresponding label of :code:`weights`.

        Return:
            dict: Feedback under the key :code:`"output"`.
        """
        return self._evaluator(kwargs["weights"], kwargs["weight_lbl"], kwargs["fam_tree"])


class LawEvaluator(object):
    """
    Object that evaluates the feedback law approximation given by a :py:class:`pyinduct.simulation.CanonicalForms`
    object.

    Args:
        cfs (:py:class:`pyinduct.simulation.FeedbackCanonicalForms`): evaluation handle
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

        vectors = dict()
        for power in powers:
            vector = np.hstack([terms["E"].get(order, {}).get(1, np.zeros(dim)) for order in range(max(orders) + 1)])
            vectors.update({power: vector})

        return vectors

    def __call__(self, weights, weight_label, family_tree=None):
        """
        Evaluation function for approximated feedback law.

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

            if "E" in law is not None:
                # build eval vector
                if lbl not in self._eval_vectors.keys():
                    self._eval_vectors[lbl] = self._build_eval_vector(law)

                if family_tree is None:
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

                # TODO: Support transformation hints for nested weights, too!
                elif isinstance(family_tree, collections.OrderedDict):
                    if not lbl in family_tree:
                        raise ValueError("Transformation hints for nested weights not supported yet.")
                    s_start, s_stop = family_tree[lbl]["slice_indices"]
                    dst_weights = weights[s_start: s_stop]

                else:
                    raise NotImplementedError

                # evaluate
                vectors = self._eval_vectors[lbl]
                for p, vec in vectors.items():
                    output = output + np.dot(vec, np.power(dst_weights, p))

            res[lbl] = dst_weights

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
                             " check for errors in feedback law!".format(out))

        res["output"] = out
        return res


class FeedbackCanonicalForms(object):
    """
    Wrapper that holds several entities of canonical forms for different sets of weights.
    """

    def __init__(self, name):
        self.name = name
        self._dynamic_forms = {}
        self._static_form = CanonicalForm(self.name + "static")

    def add_to(self, weight_label, term, val):
        """
        Add val to the canonical form for weight_label, see :py:func:`CanonicalForm.add_to` for further information.

        Args:
            weight_label (str): Basis to add onto.
            term: Coefficient to add onto, see :py:func:`CanonicalForm.add_to`.
            val: Values to add.
        """
        if term["name"] in "fG":
            # hold f and g vector separately
            self._static_form.add_to(term, val)
            return

        if weight_label not in list(self._dynamic_forms.keys()):
            self._dynamic_forms[weight_label] = CanonicalForm("_".join([self.name + weight_label]))

        self._dynamic_forms[weight_label].add_to(term, val)

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
        return {label: val.get_terms() for label, val in self._dynamic_forms.items()}


def approximate_feedback_law(feedback_law):
    """
    Function that approximates the feedback law, given by a list of sum terms that equal u.
    The result is a function handle that contains pre-evaluated terms and only needs the current weights (and their
    respective label) to be applied.

    Args:
        feedback_law (:py:class:`FeedbackLaw`): Function handle that calculates the feedback law output if provided with
            correct weights.
    Return:
        :py:class:`pyinduct.simulation.FeedbackCanonicalForms`: evaluation handle
    """
    print("approximating feedback law {}".format(feedback_law.name))
    if not isinstance(feedback_law, FeedbackLaw):
        raise TypeError("only input of Type FeedbackLaw allowed!")

    return _parse_feedback_law(feedback_law)


def _parse_feedback_law(law):
    """
    Parses the given feedback law by approximating given terms.

    Args:
        law (list):  List of :py:class:`pyinduct.placeholders.EquationTerm`'s

    Return:
        :py:class:`pyinduct.simulation.FeedbackCanonicalForms`: evaluation handle
    """

    # check terms
    for term in law.terms:
        if not isinstance(term, EquationTerm):
            raise TypeError("only EquationTerm(s) accepted.")

    cfs = FeedbackCanonicalForms(law.name)

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

            cfs.add_to(scalars[0].target_weight_label, get_common_target(scalars)[1], res * term.scale)

        else:
            raise NotImplementedError

    return cfs


"""
Observer section
"""


class Observer(StateSpace):
    """
    Standard observer class which correspond structurally to the standard state space implementation
    :py:class:`StateSpace` (from which it is derived).

    .. math::
        \\dot{\\boldsymbol{x}}(t) &= \\boldsymbol{A}\\boldsymbol{x}(t) + \\boldsymbol{B}u(t) + \\boldsymbol{L} \\tilde{\\boldsymbol{y}} \\\\
        \\boldsymbol{y}(t) &= \\boldsymbol{C}\\boldsymbol{x}(t) + \\boldsymbol{D}u(t)

    Where :math:`\\tilde{\\boldsymbol{y}}` is the observer error.
    The corresponding infinite dimensional observer has been approximated by a base given by weight_label.

    Args:
        weight_label: Label that has been used for approximation.
        a_matrices: :math:`\\boldsymbol{A_p}, \\dotsc, \\boldsymbol{A_0},`
        b_matrices: :math:`\\boldsymbol{B_q}, \\dotsc, \\boldsymbol{B_0},`
        input_handle: :math:`u(t)`
        f_vector:
        c_matrix: :math:`\\boldsymbol{C}`
        d_matrix: :math:`\\boldsymbol{D}`
        family_tree:
        l_matrices:
    """

    def __init__(self, weight_label, a_matrices, b_matrices, input_handle=None, f_vector=None, c_matrix=None,
                 d_matrix=None, family_tree=None, l_matrices=None):
        StateSpace.__init__(self, weight_label, a_matrices, b_matrices, input_handle, f_vector, c_matrix, d_matrix,
                            family_tree)

        if isinstance(l_matrices, np.ndarray):
            self.L = {1: l_matrices}
        else:
            self.L = l_matrices
        if self.L is None:
            self.L = {1: np.zeros((self.A[1].shape[0], 1))}

    def rhs_hint(self, _t, _sys_q, _u, sys_ss, _obs_q, obs_ss):
        q_t = StateSpace.rhs_hint(self, _t, _obs_q, _u, obs_ss)

        obs_error = obs_ss.input(time=_t,
                                 sys_weights=_sys_q, sys_weights_lbl=sys_ss.weight_lbl, sys_fam_tree=sys_ss.family_tree,
                                 obs_weights=_obs_q, obs_weights_lbl=obs_ss.weight_lbl, obs_fam_tree=obs_ss.family_tree)
        for p, l_mat in obs_ss.L.items():
            q_t = q_t + np.asarray(np.dot(l_mat, np.power(obs_error, p))).flatten()

        return np.asarray(q_t)


class ObserverError(SimulationInput):
    """
    Wrapper class for all observer errors that have to interact with the simulation environment. The terms which
    have to approximated on the basis of the system weights have to provided through the argument :code:`sys_part`
    and the terms which have to approximated on the basis of the observer weights have to provided through the
    argument :code:`obs_part`. The observer error is provided as sum of the :py:class:`FeedbackLaw`'s
    :code:`sys_part` and :code:`obs_part`.

    Args:
        sys_part (:py:class:`FeedbackLaw`): Hold the terms which approximated from system weights.
        obs_part (:py:class:`FeedbackLaw`): Hold the terms which approximated from observer weights.

    Keyword Args:
        weighted_initial_error (numbers.Number): Time :math:`t^*` (default: None). If the kwarg is given
            the observer error will be weighted
            :math:`\\tilde{y}_{\\varphi}(t) = \\varphi(t)\\tilde{y}(t)` with the function

            .. math::
                :nowrap:

                \\begin{align*}
                    \\varphi(t) =  \\left\\{\\begin{array}{lll}
                        0 & \\forall & t < 0 \\\\
                        (1-\\cos\\pi\\frac{t}{t^*}) / 2 & \\forall & t \\in [0, 1] \\\\
                        1 & \\forall  & t > 1.
                    \\end{array}\\right.
                \\end{align*}
    """

    def __init__(self, sys_part, obs_part, weighted_initial_error=None):
        SimulationInput.__init__(self, name="observer error: " + sys_part.name + " + " + obs_part.name)
        sys_c_forms = approximate_feedback_law(sys_part)
        self._sys_evaluator = LawEvaluator(sys_c_forms)
        obs_c_forms = approximate_feedback_law(obs_part)
        self._obs_evaluator = LawEvaluator(obs_c_forms)
        self._weighted_initial_error = weighted_initial_error

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
        sp = self._sys_evaluator(kwargs["sys_weights"], kwargs["sys_weights_lbl"], kwargs["sys_fam_tree"])["output"]
        op = self._obs_evaluator(kwargs["obs_weights"], kwargs["obs_weights_lbl"], kwargs["obs_fam_tree"])["output"]
        if self._weighted_initial_error is None or np.isclose(self._weighted_initial_error, 0):
            return dict(output=sp + op)
        else:
            t = np.clip(kwargs["time"], 0, self._weighted_initial_error)
            phi = (1 - np.cos(t / self._weighted_initial_error * np.pi)) * .5
            return dict(output=phi * (sp + op))


def build_observer_from_state_space(state_space):
    """
    Return a :py:class:`Observer` object based on the given :py:class:`StateSpace` object.
    The method return :code:`None` if state_space.input is not a instance of
    :py:class:`ObserverError` or if self._input_function is a instance of
    :py:class:`SimulationInputSum` which not hold any :py:class:`ObserverError` instance.

    Args:
        state_space (:py:class:`StateSpace`): State space to be convert.

    Returns:
        :py:class:`pyinduct.simulation.Observer` or None: See docstring.
    """
    input_func = state_space.input
    if not isinstance(input_func, SimulationInput):
        raise TypeError("The input function of the given state space must be a instance from SimulationInput.")
    if not isinstance(input_func, ObserverError):
        if isinstance(input_func, SimulationInputVector):
            if not any([isinstance(i, ObserverError) for i in input_func.input_vector]):
                return None
        else:
            raise NotImplementedError
    else:
        return Observer(state_space.weight_lbl, state_space.A, None, state_space.input, state_space.f,
                        state_space.C, state_space.D, state_space.family_tree, l_matrices=state_space.B)

    obs_b_matrices = dict()
    obs_l_matrices = dict()
    for exp, mat in state_space.B.items():
        obs_b_matrices[exp] = np.hstack(
            tuple([state_space.B[exp][:, index] for index in state_space.input._sys_inputs.keys()])
        )
        obs_l_matrices[exp] = np.hstack(
            tuple([state_space.B[exp][:, index] for index in state_space.input._obs_errors.keys()])
        )

    return Observer(state_space.weight_lbl, state_space.A, obs_b_matrices, state_space.input, state_space.f,
                    state_space.C, state_space.D, state_space.family_tree, obs_l_matrices)
