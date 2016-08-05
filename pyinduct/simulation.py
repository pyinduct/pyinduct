"""
This module consist of three parts.

    Simulation:
        Simulation infrastructure with helpers and data structures for preprocessing of the given equations
        and functions for postprocessing of simulation data.

    `Control`_:
        All classes and functions related to the creation of controllers as well as the implementation
        for simulation purposes.

    `Observer`_:
        Some objects for observer implementation which are mostly a combination from the objects for
        simulation and control tasks.
"""

import warnings
from abc import ABCMeta, abstractmethod
from itertools import chain

import numpy as np
from scipy.integrate import ode
from scipy.interpolate import interp1d
from scipy.linalg import block_diag
import collections

from pyinduct.core import domain_intersection, TransformationInfo, get_weight_transformation

from .core import (Function, integrate_function, calculate_scalar_product_matrix,
                   project_on_base, dot_product_l2)
from .placeholder import Scalars, TestFunction, Input, FieldVariable, EquationTerm, get_common_target
from .registry import get_base
from .visualization import EvalData

"""
Simulation section
"""


class Domain(object):
    """
    Helper class that manages ranges for data evaluation, containing parameters.

    Args:
        bounds (tupel): Interval bounds.
        num (int): Number of points in interval.
        step (numbers.Number): Distance between points (if homogeneous).
        points (array_like): Points themselves.

    Note:
        If num and step are given, num will take precedence.
    """

    def __init__(self, bounds=None, num=None, step=None, points=None):
        if points is not None:
            # points are given, easy one
            self._values = np.atleast_1d(points)
            self._limits = (points.min(), points.max())
            self._num = points.size
            # TODO check for evenly spaced entries
            # for now just use provided information
            self._step = step
        elif bounds and num:
            self._limits = bounds
            self._num = num
            self._values, self._step = np.linspace(bounds[0], bounds[1], num, retstep=True)
            if step is not None and not np.isclose(self._step, step):
                raise ValueError("could not satisfy both redundant requirements for num and step!")
        elif bounds and step:
            self._limits = bounds
            # calculate number of needed points but save correct step size
            self._num = int((bounds[1] - bounds[0]) / step + 1.5)
            self._values, self._step = np.linspace(bounds[0], bounds[1], self._num, retstep=True)
            if np.abs(step - self._step) > 1e-1:
                warnings.warn("desired step-size {} doesn't fit to given interval,"
                              " changing to {}".format(step, self._step))
        else:
            raise ValueError("not enough arguments provided!")

    def __len__(self):
        return len(self._values)

    def __getitem__(self, item):
        return self._values[item]

    @property
    def step(self):
        return self._step

    @property
    def bounds(self):
        return self._limits


class SimulationInput(object, metaclass=ABCMeta):
    """
    Base class for all objects that want to act as an input for the time-step simulation.

    The calculated values for each time-step are stored in internal memory and can be accessed by
    :py:func:`get_results` (after the simulation is finished).
    """

    def __init__(self, name=""):
        self._time_storage = []
        self._value_storage = {}
        self.name = name

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

        return out["output"]

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
        return dict(output=0)

    def get_results(self, time_steps, result_key="output", interpolation="nearest", as_eval_data=False):
        """
        Return results from internal storage for given time steps.

        Raises:
            Error: If calling this method before a simulation was running.

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
        outs = np.array([handle(**kwargs) for handle in self.inputs])
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
        if not all([isinstance(input, SimulationInput) and not isinstance(input, SimulationInputVector)
                    for input in input_vector]):
            raise TypeError("A SimulationInputVector can only hold SimulationInputs's and can not nest.")

        self.input_vector = input_vector
        self.len = len(input_vector)
        self.indices = set(np.arange(self.len) + 1)
        self.obs_err_indices = set()
        for index in self.indices:
            if isinstance(input_vector[index - 1], ObserverError):
                self.obs_err_indices.add(index)
        self.input_indices = self.indices - self.obs_err_indices

    def __call__(self, **kwargs):
        output = list()
        if "obs_weights" in kwargs.keys():
            for index in self.obs_err_indices:
                output.append(self.input_vector[index](kwargs))
        else:
            for index in self.input_indices:
                output.append(self.input_vector[index](kwargs))

        return np.array(output)


class WeakFormulation(object):
    """
    This class represents the weak formulation of a spatial problem.
    It can be initialized with several terms (see children of :py:class:`pyinduct.placeholder.EquationTerm`).
    The equation is interpreted as

    .. math:: term_0 + term_1 + ... + term_N = 0.

    Args:
        terms (list): List of object(s) of type EquationTerm.
        dynamic_weights (str): Weights (weight label) which occur temporal derived. It is only one kind of weight
            labels allowed in the weak formulation if :code:`dynamic_weights` will not provided.
    """

    def __init__(self, terms, dynamic_weights=None, name=None):
        if isinstance(terms, EquationTerm):
            terms = [terms]
        if not isinstance(terms, list):
            raise TypeError("only (list of) {0} allowed".format(EquationTerm))

        for term in terms:
            if not isinstance(term, EquationTerm):
                raise TypeError("Only EquationTerm(s) are accepted.")

        self.terms = terms
        self.dynamic_weights = dynamic_weights
        self.name = name


class StateSpace(object):
    """
    Standard state space implementation for a dynamic system with

    .. math::
        \\boldsymbol{\\dot{x}}(t) &= \\boldsymbol{A}\\boldsymbol{x}(t) + \\boldsymbol{B}u(t) \\\\
        \\boldsymbol{y}(t) &= \\boldsymbol{C}\\boldsymbol{x}(t) + \\boldsymbol{D}u(t).

    The corresponding infinite dimensional system has been approximated by a base given by weight_label.

    Args:
        weight_label: Label that has been used for approximation.
        a_matrices: :math:`\\boldsymbol{A_p}, \\dotsc, \\boldsymbol{A_0},`
        b_matrices: :math:`\\boldsymbol{B_q}, \\dotsc, \\boldsymbol{B_0},`
        input_handle: :math:`u(t)`
        f_vector:
        c_matrix: :math:`\\boldsymbol{C}`
        d_matrix: :math:`\\boldsymbol{D}`
    """

    def __init__(self, weight_label, a_matrices, b_matrices, input_handle=None, f_vector=None, c_matrix=None,
                 d_matrix=None):
        self.weight_lbl = weight_label

        self.f = f_vector
        self.C = c_matrix
        self.D = d_matrix

        # mandatory
        if isinstance(a_matrices, np.ndarray):
            self.A = {1: a_matrices}
        else:
            self.A = a_matrices

        # optional
        # TODO change order: 1 to order that is guaranteed to be in.
        if isinstance(b_matrices, np.ndarray):
            self.B = {1: np.atleast_2d(b_matrices)}
        else:
            self.B = b_matrices
        if self.B is None:
            self.B = {1: np.zeros((self.A[1].shape[0], 1))}

        if self.f is None:
            self.f = np.zeros((self.A[1].shape[0],))
        if self.C is None:
            self.C = np.zeros((1, self.A[1].shape[1]))
        if self.D is None:
            self.D = np.zeros((1, np.atleast_2d(self.B[1]).T.shape[1]))

        if input_handle is None:
            self.input = EmptyInput(self.B[1].shape[1])
        else:
            self.input = input_handle
        if isinstance(self.input, SimulationInputVector):
            if not all([bi.shape[1] == self._input_function.num for bi in self.B.values()]):
                raise ValueError("Input vector has more elements than (at least) one of the B matrices has rows.")
        elif isinstance(self.input, SimulationInput):
            if not all([1 in np.atleast_2d(bi).shape for bi in self.B.values()]):
                raise ValueError("All B matrices must be column vectors.")
        elif not callable(self.input):
            raise TypeError("Input must be callable!")

    def rhs_hint(self, _t, _q, ss):
        q_t = ss.f
        for p, a_mat in ss.A.items():
            # np.add(q_t, np.dot(a_mat, np.power(_q, p)))
            q_t = q_t + np.dot(a_mat, np.power(_q, p))

        u = ss.input(time=_t, weights=_q, weight_lbl=ss.weight_lbl)
        for p, b_mat in ss.B.items():
            q_t = q_t + np.dot(b_mat, np.power(u, p)).flatten()

        return q_t


# TODO update signature
def simulate_systems(weak_forms, initial_states, time_interval, time_step, spatial_interval, spatial_step):
    """
    Convenience wrapper for simulate system, see :py:func:`simulate_system` for parameters.

    Args:
        weak_forms (:py:class:`WeakFormulation`):
        initial_states:
        time_interval:
        time_step:
        spatial_interval:
        spatial_step:
    """
    return [simulate_system(sys, initial_states, time_interval, time_step, spatial_interval, spatial_step)
            for sys in weak_forms]


def simulate_system(weak_form, initial_states, temporal_domain, spatial_domain, settings=None, der_orders=(0, 0)):
    """
    Convenience wrapper that encapsulates the whole simulation process.

    Args:
        weak_form (:py:class:`WeakFormulation`): Weak formulation of the system to simulate.
        initial_states (numpy.ndarray): Array of core.Functions for :math:`x(t=0, z), \\dot{x}(t=0, z), \\dotsc, x^{(n)}(t=0, z)`.
        temporal_domain (:py:class:`Domain`): Domain object holding information for time evaluation.
        spatial_domain (:py:class:`Domain`): Domain object holding information for spatial evaluation.
        der_orders (tuple): Tuple of derivative orders (time, spat) that shall be evaluated additionally.
        settings: Integrator settings, see :py:func:`simulate_state_space`.

    Return:
        list: List of :py:class:`pyinduct.visualization.EvalData` objects, holding the results for the FieldVariable and asked derivatives.
    """
    print(("simulating system: {0}".format(weak_form.name)))
    if not isinstance(weak_form, WeakFormulation):
        raise TypeError("only WeakFormulation accepted.")

    initial_states = np.atleast_1d(initial_states)
    if not isinstance(initial_states[0], Function):
        raise TypeError("only core.Function accepted as initial state")

    if not isinstance(temporal_domain, Domain) or not isinstance(spatial_domain, Domain):
        raise TypeError("domains must be given as Domain object")

    # parse input and create state space system
    print(">>> parsing formulation")
    canonical_form = parse_weak_formulation(weak_form)
    print(">>> creating state space system")
    state_space_form = canonical_form.convert_to_state_space()

    # calculate initial state
    print(">>> deriving initial conditions")
    q0 = np.array([project_on_base(initial_state, get_base(
        canonical_form.weights, 0)) for initial_state in
                   initial_states]).flatten()

    # simulate
    print(">>> performing time step integration")
    sim_domain, q = simulate_state_space(state_space_form, q0, temporal_domain, settings=settings)

    # evaluate
    print(">>> performing postprocessing")
    temporal_order = min(initial_states.size - 1, der_orders[0])
    data = process_sim_data(canonical_form.weights, q, sim_domain, spatial_domain, temporal_order, der_orders[1],
                            name=canonical_form.name)

    print("finished simulation.")
    return data


def process_sim_data(weight_lbl, q, temp_domain, spat_domain, temp_order, spat_order, name=""):
    """
    Create handles and evaluate at given points.

    Args:
        weight_lbl (str): Label of Basis for reconstruction.
        temp_order: Order or temporal derivatives to evaluate additionally.
        spat_order: Order or spatial derivatives to evaluate additionally.
        q: weights
        spat_domain (:py:class:`Domain`): Domain object providing values for spatial evaluation.
        temp_domain (:py:class:`Domain`): Timesteps on which rows of q are given.
        name (str): Name of the WeakForm, used to generate the dataset.
    """
    data = []

    # temporal
    ini_funcs = get_base(weight_lbl, 0)
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
    The canonical form of an ordinary differential equation system of order n.
    """

    def __init__(self, name=None):
        self.name = name
        self._matrices = {}
        self._max_order = dict(E=None, G=None)
        self._max_exponent = dict(E=None, G=None)
        self._weights = None
        self._input_function = None
        self._inverse_en_hash = None
        self._en_hash = None

    @staticmethod
    def _build_name(term):
        return "_" + term[0] + str(term[1])

    @property
    def input_function(self):
        return self._input_function

    @input_function.setter
    def input_function(self, func):
        if self._input_function is None:
            self._input_function = func
        if self._input_function != func:
            raise ValueError("Already defined input is overridden!")

    @property
    def weights(self):
        return self._weights

    @weights.setter
    def weights(self, weight_lbl):
        if not isinstance(weight_lbl, str) and not weight_lbl is None:
            raise TypeError("Only string allowed as weight label!")
        if self._weights is None:
            self._weights = weight_lbl
            if weight_lbl is not None:
                self._len_weights = len(get_base(self._weights, 0))
        if self._weights != weight_lbl:
            raise ValueError("Already defined target weights are overridden!")

    @property
    def inverse_e_n(self):
        if self._max_order["E"] is None:
            warnings.warn("There is no E matrix in this canonical form.")
            return None
        else:
            en = self._matrices["E"][self._max_order["E"]][self._max_exponent["E"]]
        if self._en_hash is None or not np.allclose(en, self._en_hash):
            self._en_hash = en
            if en.shape[0] != en.shape[1]:
                raise warnings.warn("CanonicalForm holds rectangle matrix. Request for inverse unintended?")
            self._inverse_en_hash = en.I
            return self._inverse_en_hash
        else:
            return self._inverse_en_hash

    def add_to(self, term, value, column=None):
        """
        Adds the value :py:obj:`value` to term :py:obj:`term`. :py:obj:`term` is a dict that describes which
        coefficient matrix of the canonical form the value shall be added to.

        Args:
            term (dict): Targeted term in the canonical form h.  It has to contain:

                - name: Type of the coefficient matrix: 'E', 'f', or 'G'.
                - order: Temporal derivative order of the assigned weights.
                - exponent: Exponent of the assigned weights.
            value (:py:obj:`numpy.ndarray`): Value to add. It is converted in numpy.matrix if it is not already a
                numpy.matrix instance.
            column (int): Add the value only to one column of term (useful if only one dimension of term is known).
        """
        if not isinstance(value, np.ndarray):
            raise TypeError("Argument val must be numpy.ndarray.")
        elif not isinstance(value, np.matrix):
            value = np.matrix(value)
        if column and not isinstance(column, int):
            raise TypeError("Argument column (index) must be int.")

        # get entry
        if term["name"] == "f":
            if "order" in term or "exponent" in term:
                warnings.warn("Values to the keys order and exponent are ignored for f_vector!")
            f_vector = self._matrices.get("f", np.zeros_like(value))
            self._matrices["f"] = value + f_vector
            return

        type_group = self._matrices.get(term["name"], {})
        derivative_group = type_group.get(term["order"], {})
        target_matrix = derivative_group.get(term["exponent"], np.zeros_like(value))

        if target_matrix.shape != value.shape and column is None:
            raise ValueError("{0}{1}{2} was already initialized with dimensions {3} but value to add has "
                             "dimension {4}".format(term["name"], term["order"], term["exponent"],
                                                    target_matrix.shape, value.shape))

        if column is not None:
            # check whether the dimensions fit or if the matrix has to be extended
            if column >= target_matrix.shape[1]:
                new_target_matrix = np.zeros((target_matrix.shape[0], column + 1))
                new_target_matrix[:target_matrix.shape[0], :target_matrix.shape[1]] = target_matrix
                target_matrix = new_target_matrix

            target_matrix[:, column:column + 1] += value
        else:
            target_matrix += value

        # store changes
        derivative_group[term["exponent"]] = target_matrix
        type_group[term["order"]] = derivative_group
        self._matrices[term["name"]] = type_group

        # store greatest temporal derivative orders and exponents for "E" and "G" matrices
        if term["name"] in ("E", "G"):
            if self._max_order[term["name"]] is None or term["order"] > self._max_order[term["name"]]:
                self._max_order[term["name"]] = term["order"]
            if self._max_exponent[term["name"]] is None or term["exponent"] > self._max_exponent[term["name"]]:
                self._max_exponent[term["name"]] = term["exponent"]

    def get_terms(self):
        """
        Return all coefficient matrices of the canonical formulation.

        Return:
            Cascade of dictionaries: Structure: Type > Order > Exponent.
        """
        return self._matrices

    def convert_to_state_space(self):
        """
        Convert the canonical ode system of order n a into an ode system of order 1.
        This will only work if the highest derivative of the given FieldVariable can be isolated!

        Return:
            :py:class:`StateSpace` object:
        """
        if "f" in self._matrices:
            # TODO add functionality to StateSpace and allow f
            raise NotImplementedError

        # system matrices A_*
        # check whether the system can be formulated in an explicit form
        max_order = self._max_order["E"]

        if len(self._matrices["E"][max_order]) > 1:
            # more than one power of the highest derivative -> implicit formulation
            raise NotImplementedError

        pb = next(iter(self._matrices["E"][max_order]))
        if pb != 1:
            # TODO raise the resulting last blocks to 1/pb
            raise NotImplementedError

        e_n_pb_inv = self.inverse_e_n
        dim_x = e_n_pb_inv.shape[0]  # length of the weight vector

        dim_xb = max_order * dim_x  # dimension of the new system

        # get highest power
        # max_power = max(list(chain.from_iterable([list(mat) for mat in self._matrices["E"].values()])))
        powers = set(chain.from_iterable([list(mat) for mat in self._matrices["E"].values()]))

        # system matrices A_*
        a_matrices = {}
        # for p in range(max_power, 0, -1):
        for p in powers:
            a_mat = np.zeros((dim_xb, dim_xb))
            # add integrator chain
            a_mat[:-dim_x:, dim_x:] = block_diag(*[np.eye(dim_x) for a in range(max_order - 1)])
            # add "block-line" with feedback entries
            a_mat[-dim_x:, :] = -self._build_feedback("E", p, e_n_pb_inv)
            a_matrices.update({p: a_mat})

        # input matrices B_*
        if "G" in self._matrices:
            max_input_order = max(iter(self._matrices["G"]))
            # max_input_power = max(list(chain.from_iterable([list(mat) for mat in self._matrices["G"].values()])))
            input_powers = set(chain.from_iterable([list(mat) for mat in self._matrices["G"].values()]))
            dim_u = next(iter(self._matrices["G"][max_input_order].values())).shape[1]
            dim_ub = (max_input_order + 1) * dim_u  # dimension of the new systems input

            b_matrices = {}
            for q in input_powers:
                b_mat = np.zeros((dim_xb, dim_ub))
                # overwrite the last "block-line" in the matrices with input entries
                b_mat[-dim_x:, :] = -self._build_feedback("G", q, e_n_pb_inv)
                b_matrices.update({q: b_mat})
        else:
            b_matrices = None

        # the f vector
        f_mat = np.zeros((dim_xb,))
        if "f" in self._matrices:
            f_mat[-dim_x:] = self._matrices["f"]

        ss = StateSpace(self.weights, a_matrices, b_matrices, input_handle=self.input_function)
        return ss

    def _build_feedback(self, entry, power, product_mat):
        max_order = max(sorted(self._matrices[entry]))
        entry_shape = next(iter(self._matrices[entry][max_order].values())).shape
        if entry == "G":
            # include highest order for system input
            max_order += 1

        blocks = (np.dot(product_mat, self._matrices[entry].get(order, {}).get(power, np.zeros(entry_shape)))
                  for order in range(max_order))
        return np.hstack(blocks)


class CanonicalForms(object):
    """
    Wrapper that holds several entities of canonical forms for different sets of weights.
    """

    def __init__(self, dynamic_weight_label):
        self.dynamic_form = CanonicalForm()
        self.dynamic_form.weights = dynamic_weight_label
        self.static_forms = dict()

    def add_to(self, weight_label, term, val, column=None):
        """
        Add val to the canonical form for weight_label, see :py:func:`CanonicalForm.add_to` for further information.

        Args:
            weight_label (str): Basis to add onto.
            term: Coefficient to add onto, see :py:func:`CanonicalForm.add_to`.
            val: Values to add.
        """

        if weight_label == self.dynamic_form.weights or self.dynamic_form.weights is None or weight_label is None:
            # if not val.shape[0] == self.dynamic_form._len_weights:
            #     raise ValueError("Row width must correspond to the number of weights.")
            # if term["name"] == "E" and val.shape[1] != self.dynamic_form._len_weights:
            #     raise ValueError("Column width of E matrix must correspond to the number of weights!")

            self.dynamic_form.add_to(term, val, column=column)
        else:
            # if not val.shape[0] == self.static_forms[weight_label]._len_weights:
            #     raise ValueError("Row width must correspond to the number of weights of dynamic form.")
            # if term["name"] == "E" and val.shape[1] != self.static_forms[weight_label]._len_weights:
            #     raise ValueError("Column width of E matrix must correspond to the number of weights!")
            if weight_label not in list(self.static_forms.keys()):
                self.static_forms[weight_label] = CanonicalForm(weight_label)
            elif not isinstance(weight_label, str):
                raise TypeError("Argument weight_label must provided as string.")

            self.static_forms[weight_label].add_to(term, val, column=column)

    def get_dynamic_terms(self):
        """
        Return:
            dict: Terms of the dynamic :py:class:`CanonicalForm`.
        """
        return self.dynamic_form.get_terms()

    def get_static_terms(self):
        """
        Return:
            dict: Dictionary of terms for each static :py:class:`CanonicalForm`.
        """
        return {label: val.get_terms() for label, val in self.static_forms.items()}


def convert_cfs_to_state_space(list_of_cfs):
    """
    Create :py:class:`StateSpace` from list :code:`list_of_cfs` with elements from type :py:class:`CanonicalForms`.
    In the common case the :math:`N` list elements are derived from :math:`N` :py:class:`WeakFormulation`s which
    represent :math:`N` coupled pde's with boundary conditions.

    Args:
        list_of_cfs (list): List with elements from type :py:class:`CanonicalForms`.

    Returns:
        :py:class:`StateSpace`: State space approximation for the time dynamic of (basically) coupled pde's.
    """
    value_error_string = "Problem formulation meets not the specification. \n\n"
    odict_info = collections.OrderedDict()
    # for label, order in [(cf.dynamic_form.weights, cf.dynamic_form._max_order["E"]) for cf in list_of_cfs]:
    for cfs in list_of_cfs:
        label = cfs.dynamic_form.weights
        order = cfs.dynamic_form._max_order["E"]
        if order is None:
            raise TypeError(value_error_string + "The dynamic_form of an CanonicalForms object must hold "
                                                 "temporal derived weights.")
        odict_info[label] = dict()
        odict_info[label]["max_order"] = order
        odict_info[label]["weights_length"] = len(get_base(label, 0))
        odict_info[label]["state_space"] = cfs.dynamic_form.convert_to_state_space()
        odict_info[label]["stat_weights"] = set(cfs.static_forms.keys())
        odict_info[label]["cfs"] = cfs

    if len(set(odict_info.keys())) != len(list(odict_info.keys())):
        raise ValueError("There are at least two CanonicalForms objects with the same dynamic weight label.")

    input_function_set = set(
        [cfs.dynamic_form.input_function for cfs in list_of_cfs if not cfs.dynamic_form.input_function is None]
    )
    if len(input_function_set) > 1:
        raise ValueError("All given CanonicalForms.dynamic_form's must hold the same input function (or None).")
    elif len(input_function_set) == 1:
        if not isinstance(list(input_function_set)[0], (SimulationInput, SimulationInputVector)):
            raise TypeError("Input function must be from type SimulationInput or SimulationInputVector.")
        else:
            input_function = input_function_set.pop()
    else:
        input_function = None

    for cfs in list_of_cfs:
        cfs_to_check = [cfs.dynamic_form] + list(cfs.static_forms.values())

        list_of_powers = np.array([list(cf._max_exponent.values()) for cf in cfs_to_check]).flatten().astype(float)
        if any([power > 1 for power in list_of_powers]):
            raise NotImplementedError("Exponents greater 1 not implemented yet.")

        if not all([cf.input_function == None for cf in cfs.static_forms.values()]):
            raise ValueError("Input functions in static forms not allowed.")

        if any(["f" in cf._matrices.keys() for cf in cfs_to_check]):
            raise ValueError("No matrix \"f\" allowed for now.")

        if cfs.dynamic_form.max_order["G"] > 1:
            raise ValueError("For now, only order 1 for input matrix \"G\" supported.")

    # check for valid problem formulation
    for dyn_label in odict_info.keys():
        for cfs in list_of_cfs:
            if dyn_label in cfs.static_forms.keys():
                if cfs.static_forms[dyn_label]._max_order["E"] >= odict_info[dyn_label]["max_order"]:
                    raise ValueError(value_error_string +
                                     "For a specific weight_label, the temporal order of the static_form"
                                     "can not be greater or equal as that from the corresponding dynamic form.")

    dim_x = np.sum([value["weights_length"] * value["max_order"] for value in odict_info.values()])
    if isinstance(input_function, SimulationInput):
        dim_u = 1
    elif isinstance(input_function, SimulationInputVector):
        dim_u = len(input_function.indices)

    A = np.nan * np.matrix(np.zeros((dim_x, dim_x)))
    B = np.nan * np.matrix(np.zeros((dim_x,)))
    C = np.matrix(np.zeros((dim_u, dim_x)))
    D = np.matrix(np.zeros((dim_u, dim_u)))

    list_of_labels = list(odict_info.keys())
    a = None
    for a_lbl in list_of_labels:
        row_dict = dict()
        a_fraction = odict_info[a_lbl]["state_space"].A[1]
        row_dict[list_of_labels.index(a_lbl)] = a_fraction
        list_without_a_lbl = list(odict_info.keys()).remove(a_lbl)
        for h_lbl in list_without_a_lbl:
            if h_lbl in odict_info[a_lbl]["stat_weights"]:
                for ord in range(odict_info[h_lbl]["max_order"] + 1):
                    if ord in odict_info[a_lbl]["cfs"].static_forms[h_lbl]._matrices["E"].keys():
                        h_fraction = - odict_info[a_lbl]["cfs"].dynamic_form.inverse_e_n * \
                                     odict_info[a_lbl]["cfs"].static_forms[h_lbl]._matrices["E"][ord]
                    else:
                        h_fraction = np.matrix(np.zeros((odict_info[a_lbl]["weights_length"] * odict_info[a_lbl]["max_order"],
                                                         odict_info[h_lbl]["weights_length"])))
                    if ord > 0:
                        h_block = np.hstack((h_block, h_fraction))
                    else:
                        h_block = h_fraction
                if odict_info[a_lbl]["max_order"] > 1:
                    h_block = np.vstack((np.zeros((odict_info[a_lbl]["weights_length"] * (odict_info[a_lbl]["max_order"] - 1),
                                                   odict_info[h_lbl]["weights_length"] * odict_info[h_lbl]["max_order"])),
                                         h_block))
            else:
                h_block = np.vstack((np.zeros((odict_info[a_lbl]["weights_length"] * (odict_info[a_lbl]["max_order"]),
                                               odict_info[h_lbl]["weights_length"] * odict_info[h_lbl]["max_order"])),
                                     h_block))
            row_dict[list_of_labels.index(h_lbl)] = h_block
        row_odict = collections.OrderedDict(sorted(row_dict.items(), key=lambda t: t[0]))
        row = np.hstack(tuple(row_dict.values()))
        if list_of_labels.index(a_lbl) == 0:
            matrix = row
        else:
            matrix = np.vstack((matrix, row))





def parse_weak_formulation(weak_form):
    """
    Creates an ode system for the weights x_i based on the weak formulation.

    Args:
        weak_form: Weak formulation of the pde.

    Return:
        :py:class:`CanonicalForm`: n'th-order ode system.
    """

    if not isinstance(weak_form, WeakFormulation):
        raise TypeError("Only able to parse WeakFormulation.")

    cfs = CanonicalForms(weak_form.dynamic_weights)

    # handle each term
    for term in weak_form.terms:
        # extract Placeholders
        placeholders = dict(scalars=term.arg.get_arg_by_class(Scalars),
                            functions=term.arg.get_arg_by_class(TestFunction),
                            field_variables=term.arg.get_arg_by_class(FieldVariable),
                            inputs=term.arg.get_arg_by_class(Input))

        # field variable terms, sort into E_np, E_n-1p, ..., E_0p
        if placeholders["field_variables"]:
            if len(placeholders["field_variables"]) != 1:
                raise NotImplementedError
            field_var = placeholders["field_variables"][0]
            temp_order = field_var.order[0]
            exponent = field_var.data["exponent"]
            init_funcs = get_base(field_var.data["func_lbl"], field_var.order[1])
            shape_funcs = np.array([func.raise_to(exponent) for func in init_funcs])

            # for now we use .startswith and .endswith, while the function label
            # will manipulated from placeholder.Product._simplify_product
            if not field_var.data["func_lbl"].startswith(field_var.data["weight_lbl"]):
                if not field_var.data["func_lbl"].endswith(field_var.data["weight_lbl"]):
                    raise ValueError("In the simulation infrastructure of pyinduct field variables with weight labels"
                                     "which differing from function labels not considered. Use this feature only for"
                                     "controller approximation.")

            if placeholders["inputs"]:
                # TODO think about this case, is it relevant?
                raise NotImplementedError

            # is the integrand a product?
            if placeholders["functions"]:
                if len(placeholders["functions"]) != 1:
                    raise NotImplementedError
                func = placeholders["functions"][0]
                test_funcs = get_base(func.data["func_lbl"], func.order[1])
                result = calculate_scalar_product_matrix(dot_product_l2, test_funcs, shape_funcs)
            else:
                # extract constant term and compute integral
                a = Scalars(np.atleast_2d([integrate_function(func, func.nonzero)[0] for func in shape_funcs]))

                if placeholders["scalars"]:
                    b = placeholders["scalars"][0]
                else:
                    b = Scalars(np.ones_like(a.data.T))

                result = _compute_product_of_scalars([a, b])

            if cfs.dynamic_form.weights is None:
                cfs.dynamic_form.weights = field_var.data["weight_lbl"]
            cfs.add_to(field_var.data["weight_lbl"],
                       dict(name="E", order=temp_order, exponent=exponent),
                       result * term.scale)
            continue

        # TestFunction or pre evaluated terms, those can end up in E, f or G
        if placeholders["functions"]:
            if not 1 <= len(placeholders["functions"]) <= 2:
                raise NotImplementedError
            func = placeholders["functions"][0]
            test_funcs = get_base(func.data["func_lbl"], func.order[1])

            if len(placeholders["functions"]) == 2:
                # TODO this computation is nonsense. Result must be a vector containing int(tf1*tf2)
                raise NotImplementedError

                func2 = placeholders["functions"][1]
                test_funcs2 = get_base(func2.data["func_lbl"], func2.order[2])
                result = calculate_scalar_product_matrix(dot_product_l2, test_funcs, test_funcs2)
                cfs.add_to(weak_form.dynamic_weights, ("f", 0), result * term.scale)
                continue

            if placeholders["scalars"]:
                a = placeholders["scalars"][0]
                b = Scalars(np.vstack([integrate_function(func, func.nonzero)[0]
                                       for func in test_funcs]))
                result = _compute_product_of_scalars([a, b])
                weight_lbl, target = get_common_target(placeholders["scalars"])
                if cfs.dynamic_form.weights is None:
                    cfs.dynamic_form.weights = weight_lbl
                cfs.add_to(weight_lbl, target, result * term.scale)
                continue

            if placeholders["inputs"]:
                if len(placeholders["inputs"]) != 1:
                    raise NotImplementedError
                input_var = placeholders["inputs"][0]
                input_func = input_var.data["input"]
                input_index = input_var.data["index"]
                input_exp = input_var.data["exponent"]

                # here we would need to provide derivative handles in the callable
                input_order = input_var.order[0]
                if input_order > 0:
                    raise NotImplementedError

                result = np.array([[integrate_function(func, func.nonzero)[0]] for func in test_funcs])
                cfs.add_to(weak_form.dynamic_weights,
                           dict(name="G", order=input_order, exponent=input_exp),
                           result * term.scale,
                           column=input_index)
                cfs.dynamic_form.input_function = input_func
                continue

        # pure scalar terms, sort into corresponding matrices
        if placeholders["scalars"]:
            result = _compute_product_of_scalars(placeholders["scalars"])
            weight_lbl, target = get_common_target(placeholders["scalars"])

            if placeholders["inputs"]:
                input_var = placeholders["inputs"][0]
                input_func = input_var.data["input"]
                input_index = input_var.data["index"]
                input_exp = input_var.data["exponent"]

                # here we would need to provide derivative handles in the callable
                input_order = input_var.order[0]
                if input_order > 0:
                    raise NotImplementedError

                # this would mean that the input term should appear in a matrix like E1 or E2
                if target["name"] == "E":
                    raise NotImplementedError

                cfs.add_to(weak_form.dynamic_weights,
                           dict(name="G", order=input_order, exponent=input_exp),
                           result * term.scale,
                           column=input_index)
                cfs.dynamic_form.input_function = input_func
                continue

            if cfs.dynamic_form.weights is None:
                cfs.dynamic_form.weights = weight_lbl
            cfs.add_to(weight_lbl, target, result * term.scale)
            continue

    if cfs.static_forms == dict():
        return cfs.dynamic_form
    else:
        return cfs


def _compute_product_of_scalars(scalars):
    if len(scalars) > 2:
        raise NotImplementedError

    if len(scalars) == 1:
        res = scalars[0].data
    elif scalars[0].data.shape == scalars[1].data.shape:
        res = np.prod(np.array([scalars[0].data, scalars[1].data]), axis=0)
    elif scalars[0].data.shape == scalars[1].data.T.shape:
        # guarantee dyadic product no matter in which order args are passed
        if scalars[0].data.shape[0] > scalars[0].data.shape[1]:
            res = np.dot(scalars[0].data, scalars[1].data)
        else:
            res = np.dot(scalars[1].data, scalars[0].data)
    else:
        raise NotImplementedError

    return res


def simulate_state_space(sys_ss, sys_init_state, temp_domain, obs_ss=None, obs_init_state=None, settings=None):
    """
    Wrapper to simulate a system given in state space form:

    .. math:: \\dot{q} = A_pq^p + A_{p-1}q^{p-1} + \\dotsb + A_0q + Bu.

    Args:
        sys_ss (:py:class:`StateSpace`): State space formulation of the system.
        init_state: Initial state vector of the system.
        temp_domain (:py:class:`Domain`): Temporal domain object.
        obs_ss (:py:class:`Observer`): State space formulation of the observer.
        settings (dict): Parameters to pass to the :func:`set_integrator` method of the :class:`scipy.ode` class, with the integrator
            name included under the key :obj:`name`.

    Return:
        tuple: Time :py:class:`Domain` object and weights matrix.
    """
    if not isinstance(sys_ss, StateSpace) or isinstance(sys_ss, Observer):
        raise TypeError

    input_handle = sys_ss.input

    if not isinstance(input_handle, SimulationInput):
        raise TypeError("Only pyinduct.simulation.SimulationInput supported.")
    if isinstance(input_handle, SimulationInputVector):
        if any([isinstance(input, ObserverError) for input in input_handle.input_vector]):
            raise TypeError

    t = [temp_domain[0]]
    q = [sys_init_state]

    # TODO export cython code?
    def _rhs(_t, _q, ss):
        q_t = ss.rhs_hint(_t, _q, ss)

        return q_t

    r = ode(_rhs)

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

    r.set_f_params(sys_ss)
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
    funcs = get_base(base_label, spat_order)
    if weights.shape[1] != funcs.shape[0]:
        raise ValueError("weights (len={0}) have to fit provided functions (len={1})!".format(weights.shape[1],
                                                                                              funcs.size))

    # evaluate shape functions at given points
    shape_vals = np.array([func.evaluation_hint(spat_domain) for func in funcs])

    def eval_spatially(weight_vector):
        return np.real_if_close(np.dot(weight_vector, shape_vals), 1000)

    data = np.apply_along_axis(eval_spatially, 1, weights)
    return EvalData([temp_domain, spat_domain], data, name=name)


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
        return self._evaluator(kwargs["weights"], kwargs["weight_lbl"])


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

        vectors = {}
        for power in powers:
            vector = np.hstack([terms["E"].get(order, {}).get(1, np.zeros(dim)) for order in range(max(orders) + 1)])
            vectors.update({power: vector})

        return vectors

    def __call__(self, weights, weight_label):
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

    Where :math:`\tilde{\\boldsymbol{y}}` is the observer error.
    The corresponding infinite dimensional observer has been approximated by a base given by weight_label.

    Args:
        weight_label: Label that has been used for approximation.
        a_matrices: :math:`\\boldsymbol{A_p}, \\dotsc, \\boldsymbol{A_0},`
        b_matrices: :math:`\\boldsymbol{B_q}, \\dotsc, \\boldsymbol{B_0},`
        input_handle: :math:`u(t)`
        f_vector:
        c_matrix: :math:`\\boldsymbol{C}`
        d_matrix: :math:`\\boldsymbol{D}`
        observer_error_indices (array_like): List of indices. The indices show which
            input variable is an observer error.
    """

    def __init__(self, weight_label, a_matrices, b_matrices, input_handle=None, l_matrices=None, obs_err_handle=None,
                 f_vector=None, c_matrix=None, d_matrix=None, observer_error_indices=None):
        StateSpace.__init__(self, weight_label, a_matrices, b_matrices, input_handle, f_vector, c_matrix, d_matrix)

        if isinstance(l_matrices, np.ndarray):
            self.L = {1: l_matrices}
        else:
            self.L = l_matrices
        if self.L is None:
            self.L = {1: np.zeros((self.A[1].shape[0], 1))}

        if obs_err_handle is None:
            self.obs_err = EmptyInput(self.L[1].shape[1])
        else:
            self.obs_err = obs_err_handle
        if not callable(self.obs_err):
            raise TypeError("observer error must be callable!")


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
    :py:class:`ObserverError` or if self._input_function is a instance of
    :py:class:`SimulationInputSum` which not contain any :py:class:`ObserverError` instance.

    Returns:
        :py:class:`pyinduct.simulation.Observer` or None: See docstring.
    """
    pass
