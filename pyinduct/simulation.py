"""
Simulation infrastructure with helpers and data structures for preprocessing of the given equations
and functions for postprocessing of simulation data.
"""

from abc import ABCMeta, abstractmethod
from collections import Iterable
import warnings
import numpy as np
from itertools import chain
from scipy.linalg import block_diag
from scipy.interpolate import interp1d
from scipy.integrate import ode

from .registry import get_base, is_registered
from .core import (Function, integrate_function, calculate_scalar_product_matrix, project_on_base, dot_product_l2)
from .placeholder import Scalars, TestFunction, Input, FieldVariable, EquationTerm, get_common_target
from .visualization import EvalData


class Domain(object):
    """
    Helper class that manages ranges for data evaluation, containing parameters.

    Args:
        bounds (tuple): Interval bounds.
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
        func = interp1d(np.array(self._time_storage), np.array(self._value_storage[result_key]), kind=interpolation,
                        assume_sorted=True, axis=0)
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


class WeakFormulation(object):
    """
    This class represents the weak formulation of a spatial problem.
    It can be initialized with several terms (see children of :py:class:`pyinduct.placeholder.EquationTerm`).
    The equation is interpreted as

    .. math:: term_0 + term_1 + ... + term_N = 0.

    Args:
        terms (list): List of object(s) of type EquationTerm.
    """

    def __init__(self, terms, name=None):
        if isinstance(terms, EquationTerm):
            terms = [terms]
        if not isinstance(terms, list):
            raise TypeError("only (list of) {0} allowed".format(EquationTerm))

        for term in terms:
            if not isinstance(term, EquationTerm):
                raise TypeError("Only EquationTerm(s) are accepted.")

        self.terms = terms
        self.name = name


class StateSpace(object):
    """
    Wrapper class that represents the state space form of a dynamic system where

    .. math::
        \\boldsymbol{\\dot{x}}(t) &= \\boldsymbol{A}\\boldsymbol{x}(t) + \\boldsymbol{B}u(t) \\\\
        \\boldsymbol{y}(t) &= \\boldsymbol{C}\\boldsymbol{x}(t) + \\boldsymbol{D}u(t)

    which has been approximated by projection on a base given by weight_label.

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
            self.B = {1: b_matrices}
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
        if not callable(self.input):
            raise TypeError("input must be callable!")


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
    return [simulate_system(sys, initial_states, time_interval, time_step, spatial_interval, spatial_step) for sys in
            weak_forms]


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
    q0 = np.array([project_on_base(initial_state, get_base(canonical_form.weights, 0)) for initial_state in
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
        self._max_idx = dict(E=0, f=0, G=0)
        self._weights = None
        self._input_function = None

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
            raise ValueError("already defined input is overridden!")

    @property
    def weights(self):
        return self._weights

    @weights.setter
    def weights(self, weight_lbl):
        if not isinstance(weight_lbl, str):
            raise TypeError("only string allowed as weight label!")
        if self._weights is None:
            self._weights = weight_lbl
        if self._weights != weight_lbl:
            raise ValueError("already defined target weights are overridden!")

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
        if not isinstance(value, np.ndarray):
            raise TypeError("val must be numpy.ndarray")
        if column and not isinstance(column, int):
            raise TypeError("column index must be int")

        # get entry
        if term["name"] == "f":
            if "order" in term or "exponent" in term:
                warnings.warn("order and exponent are ignored for f_vector!")
            f_vector = self._matrices.get("f", np.zeros_like(value))
            self._matrices["f"] = value + f_vector
            return

        type_group = self._matrices.get(term["name"], {})
        derivative_group = type_group.get(term["order"], {})
        target_matrix = derivative_group.get(term["exponent"], np.zeros_like(value))

        if target_matrix.shape != value.shape and column is None:
            raise ValueError("{0}{1}{2} was already initialized with dimensions {3} but value to add has "
                             "dimension {4}".format(term["name"], term["order"], term["exponent"], target_matrix.shape,
                                                    value.shape))

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
        max_order = max(self._matrices["E"])

        if len(self._matrices["E"][max_order]) > 1:
            # more than one power of the highest derivative -> implicit formulation
            raise NotImplementedError

        pb = next(iter(self._matrices["E"][max_order]))
        if pb != 1:
            # TODO raise the resulting last blocks to 1/pb
            raise NotImplementedError

        e_n_pb = self._matrices["E"][max_order][pb]
        rank_e_n_pb = np.linalg.matrix_rank(e_n_pb)
        if rank_e_n_pb != max(e_n_pb.shape) or e_n_pb.shape[0] != e_n_pb.shape[1]:
            raise ValueError("singular matrix provided")

        dim_x = e_n_pb.shape[0]  # length of the weight vector
        e_n_pb_inv = np.linalg.inv(e_n_pb)

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

        blocks = (np.dot(product_mat, self._matrices[entry].get(order, {}).get(power, np.zeros(entry_shape))) for order
                  in range(max_order))
        return np.hstack(blocks)


class CanonicalForms(object):
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


def parse_weak_formulation(weak_form):
    """
    Creates an ode system for the weights x_i based on the weak formulation.

    Args:
        weak_form: Weak formulation of the pde.

    Return:
        :py:class:`CanonicalForm`: n'th-order ode system.
    """

    if not isinstance(weak_form, WeakFormulation):
        raise TypeError("only able to parse WeakFormulation")

    cf = CanonicalForm(weak_form.name)

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

            cf.weights = field_var.data["weight_lbl"]
            cf.add_to(dict(name="E", order=temp_order, exponent=exponent), result * term.scale)
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
                cf.add_to(("f", 0), result * term.scale)
                continue

            if placeholders["scalars"]:
                a = placeholders["scalars"][0]
                b = Scalars(np.vstack([integrate_function(func, func.nonzero)[0] for func in test_funcs]))
                result = _compute_product_of_scalars([a, b])
                cf.add_to(get_common_target(placeholders["scalars"]), result * term.scale)
                continue

            if placeholders["inputs"]:
                if len(placeholders["inputs"]) != 1:
                    raise NotImplementedError
                input_var = placeholders["inputs"][0]
                input_func = input_var.data["input"]
                input_index = input_var.data["index"]
                input_exp = input_var.data["exponent"]
                input_order = input_var.order[0]

                result = np.array([[integrate_function(func, func.nonzero)[0]] for func in test_funcs])
                cf.add_to(dict(name="G", order=input_order, exponent=input_exp), result * term.scale,
                          column=input_index)
                cf.input_function = input_func
                continue

        # pure scalar terms, sort into corresponding matrices
        if placeholders["scalars"]:
            result = _compute_product_of_scalars(placeholders["scalars"])
            target = get_common_target(placeholders["scalars"])

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

                cf.add_to(dict(name="G", order=input_order, exponent=input_exp), result * term.scale,
                          column=input_index)
                cf.input_function = input_func
                continue

            cf.add_to(target, result * term.scale)
            continue

    return cf


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


def simulate_state_space(state_space, initial_state, temp_domain, settings=None):
    """
    Wrapper to simulate a system given in state space form:

    .. math:: \\dot{q} = A_pq^p + A_{p-1}q^{p-1} + \\dotsb + A_0q + Bu.

    Args:
        state_space (:py:class:`StateSpace`): State space formulation of the system.
        initial_state: Initial state vector of the system.
        temp_domain (:py:class:`Domain`): Temporal domain object.
        settings (dict): Parameters to pass to the :func:`set_integrator` method of the :class:`scipy.ode` class, with the integrator
            name included under the key :obj:`name`.

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

    # TODO export cython code?
    def _rhs(_t, _q, ss):
        q_t = ss.f
        for p, a_mat in ss.A.items():
            # np.add(q_t, np.dot(a_mat, np.power(_q, p)))
            q_t = q_t + np.dot(a_mat, np.power(_q, p))

        u = ss.input(time=_t, weights=_q, weight_lbl=ss.weight_lbl)
        for p, b_mat in ss.B.items():
            q_t = q_t + np.dot(b_mat, np.power(u, p)).flatten()

        return q_t

    r = ode(_rhs)

    # TODO check for complex-valued matrices and use 'zvode'
    if settings:
        r.set_integrator(settings.pop("name"), **settings)
    else:
        # use some sane defaults
        r.set_integrator("vode", max_step=temp_domain.step, method="adams", nsteps=1e3)

    r.set_f_params(state_space)
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
        raise ValueError(
            "weights (len={0}) have to fit provided functions (len={1})!".format(weights.shape[1], funcs.size))

    # evaluate shape functions at given points
    shape_vals = np.array([func.evaluation_hint(spat_domain) for func in funcs])

    def eval_spatially(weight_vector):
        return np.real_if_close(np.dot(weight_vector, shape_vals), 1000)

    data = np.apply_along_axis(eval_spatially, 1, weights)
    return EvalData([temp_domain, spat_domain], data, name=name)
