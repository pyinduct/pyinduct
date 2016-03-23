# -*- coding: utf-8 -*-

from abc import ABCMeta, abstractmethod
from collections import Iterable
import warnings
import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import ode

from .registry import get_base, is_registered
from .core import (Function, integrate_function, calculate_scalar_product_matrix,
                   project_on_base, dot_product_l2)
from .placeholder import Scalars, TestFunction, Input, FieldVariable, EquationTerm, get_scalar_target
from .utils import find_nearest_idx
from .visualization import EvalData


class Domain(object):
    """
    Helper class that manages ranges for data evaluation, containing:
    - interval bounds
    - number of points in interval
    - distance between points (if homogeneous)
    - points themselves

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
            self._num = int((bounds[1]-bounds[0])/step + 1.5)
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
    base class for all objects that want to act as an input for the time-step simulation.

    The calculated values for each time-step are stored in internal memory and can be accessed by
    py:func:`get_results` . After the simulation is finished.
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
        entries = self._value_storage.get("output", [])
        entries.append(out)
        self._value_storage["output"] = entries
        return out

    @abstractmethod
    def _calc_output(self, **kwargs):
        """
        handle that has to be implemented for output calculation.

        :param kwargs:
        -"time": the current simulation time
        -"weights": the current weight vector
        -"weight_lbl": the label of the weights used
        """
        pass

    def get_results(self, time_steps, result_key="output", interpolation="nearest", as_eval_data=False):
        """
        return results from internal storage for given time steps.
        .. warning:: calling this method before a simulation was run will result in an error.

        :param time_steps: time points where values are demanded
        :param result_key: type of values to be returned
        :param interpolation: interpolation method to use if demanded time-steps are not covered by the storage:
        -"nearest" use nearest point available in storage
        -"linear" interpolate between the 2 nearest points
        - see more in py:func`interp1d`
        :param as_eval_data: return results as EvalData object for straightforward display
        """
        func = interp1d(np.array(self._time_storage), np.array(self._value_storage[result_key]),
                        kind=interpolation, assume_sorted=True, axis=0)
        values = func(time_steps)

        if as_eval_data:
            # check if output was vectorial
            if len(values.shape) <= 2:
                return EvalData([time_steps], func(time_steps), name=".".join([self.name, result_key]))
            else:
                res = []
                for idx, val in enumerate(np.swapaxes(values, 0, 1)):
                    res.append(EvalData([time_steps], val[:, 0], name=".".join([self.name, result_key, str(idx)])))
                return res

        return func(time_steps)


class SimulationInputSum(SimulationInput):
    """
    helper that represents a signal mixer
    """
    def __init__(self, inputs):
        SimulationInput.__init__(self)
        self._inputs = inputs

    def _calc_output(self, **kwargs):
        outs = np.array([handle(**kwargs) for handle in self._inputs])
        return np.sum(outs)


class WeakFormulation(object):
    """
    this class represents the weak formulation of a spatial problem.
    It can be initialized with several terms (see children of :py:class:`EquationTerm`).
    The equation is interpreted as term_0 + term_1 + ... + term_N = 0

    :param terms: (list of) of object(s) of type EquationTerm
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
    wrapper class that represents the state space form of a dynamic system where
    :math:`\\boldsymbol{\\dot{x}}(t) = \\boldsymbol{A}\\boldsymbol{x}(t) + \\boldsymbol{B}u(t)` and
    :math:`\\boldsymbol{y}(t) = \\boldsymbol{C}\\boldsymbol{x}(t) + \\boldsymbol{D}u(t)`
    which has been approximated by projection on a base given by weight_label.

    :param weight_label: label that has been used for approximation
    :param a_matrix: :math:`\\boldsymbol{A}`
    :param b_matrix: :math:`\\boldsymbol{B}`
    :param c_matrix: :math:`\\boldsymbol{C}`
    :param d_matrix: :math:`\\boldsymbol{D}`
    """
    def __init__(self, weight_label, a_matrix, b_matrix, input_handle=None, c_matrix=None, d_matrix=None):
        self.weight_lbl = weight_label
        self.input = input_handle

        # TODO dimension checks
        self.A = a_matrix
        self.B = b_matrix
        self.C = c_matrix
        self.D = d_matrix


# TODO update signature
def simulate_systems(weak_forms, initial_states, time_interval, time_step, spatial_interval, spatial_step):
    """
    convenience wrapper for simulate system, see :ref:py:func:simulate_system for parameters
    :param weak_forms:
    :return:
    """
    return [simulate_system(sys, initial_states, time_interval, time_step, spatial_interval, spatial_step) for sys in
            weak_forms]


def simulate_system(weak_form, initial_states, temporal_domain, spatial_domain, der_orders=(0, 0)):
    """
    convenience wrapper that encapsulates the whole simulation process

    :param weak_form:
    :param initial_states: np.array of core.Functions for :math:`x(t=0, z), \\dot{x}(t=0, z), \\dotsc, x^{(n)}(t=0, z)`
    :param temporal_domain: sim.Domain object holding information for time evaluation
    :param spatial_domain: sim.Domain object holding information for spatial evaluation
    :param der_orders: tuple of derivative orders (time, spat) that shall be evaluated additionally

    :return: list of EvalData object, holding the results for the FieldVariable and asked derivatives
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
    sim_domain, q = simulate_state_space(state_space_form, q0, temporal_domain)

    # evaluate
    print(">>> performing postprocessing")
    temporal_order = min(initial_states.size-1, der_orders[0])
    data = process_sim_data(canonical_form.weights, q, sim_domain, spatial_domain, temporal_order, der_orders[1],
                            name=canonical_form.name)

    print("finished simulation.")
    return data


def process_sim_data(weight_lbl, q, temp_domain, spat_domain, temp_order, spat_order, name=""):
    """
    create handles and evaluate at given points
    :param weight_lbl: label of Basis for reconstruction
    :param temp_order: order or temporal derivatives to evaluate additionally
    :param spat_order: order or spatial derivatives to evaluate additionally
    :param q: weights
    :param spat_domain: sim.Domain object providing values for spatial evaluation
    :param temp_domain: timesteps on which rows of q are given
    :param name: name of the WeakForm, used to generate the dataset
    """
    data = []

    # temporal
    ini_funcs = get_base(weight_lbl, 0)
    for der_idx in range(temp_order+1):
        name = "{0}{1}".format(name, "_" + "".join(["d" for x in range(der_idx)] + ["t"]) if der_idx > 0 else "")
        data.append(evaluate_approximation(weight_lbl, q[:, der_idx * ini_funcs.size:(der_idx + 1) * ini_funcs.size],
                                           temp_domain, spat_domain, name=name))

    # spatial (0th derivative is skipped since this is already handled above)
    for der_idx in range(1, spat_order+1):
        name = "{0}{1}".format(name, "_" + "".join(["d" for x in range(der_idx)] + ["z"]) if der_idx > 0 else "")
        data.append(
            evaluate_approximation(weight_lbl, q[:, :ini_funcs.size], temp_domain, spat_domain, der_idx, name=name))

    return data


class CanonicalForm(object):
    """
    represents the canonical form of n ordinary differential equation system of order n
    """
    def __init__(self, name=None):
        self.name = name
        self._max_idx = dict(E=0, f=0, G=0)
        self._weights = None
        self._input_function = None

    @staticmethod
    def _build_name(term):
        return "_"+term[0]+str(term[1])

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

    def add_to(self, term, val, column=None):
        """
        adds the value val to term term

        :param term: tuple of name and index matrix(or vector) to add onto
        :param val: value to add
        :param column: add the value only to one column of term (useful if only one dimension of term is known)
        """
        if not isinstance(term, tuple):
            raise TypeError("term must be tuple.")
        if not isinstance(term[0], str) or term[0] not in "EfG":
            raise TypeError("term[0] must be a letter out of [E, f, G]")
        if not isinstance(term[1], int):
            raise TypeError("term index must be int")
        if not isinstance(val, np.ndarray):
            raise TypeError("val must be numpy.ndarray")
        if column and not isinstance(column, int):
            raise TypeError("column index must be int")

        name = self._build_name(term)

        # try to increment term
        try:
            entity = getattr(self, name)
            if entity.shape != val.shape and column is None:
                raise ValueError("{0} was already initialized with dimensions {1} but value to add has dimension {"
                                 "2}".format(name, entity.shape, val.shape))

            if column:
                # check whether the dimensions fit or if the matrix has to be extended
                if column >= entity.shape[1]:
                    new_entity = np.zeros((entity.shape[0], column+1))
                    new_entity[:entity.shape[0], :entity.shape[1]] = entity
                    setattr(self, name, np.copy(new_entity))

                entity = getattr(self, name)[:, column:column+1]

            # add new value
            entity += val

        except AttributeError as e:
            # no entry so far -> create entry
            setattr(self, name, np.copy(val))
        finally:
            self._max_idx[term[0]] = max(self._max_idx[term[0]], term[1])

    def get_terms(self):
        """
        construct a list of all terms that have indices and return tuple of lists

        :return: tuple of lists
        """
        terms = {}
        for entry in "EfG":
            term = []
            i = 0
            shape = None
            while i <= self._max_idx[entry]:
                name = self._build_name((entry, i))
                if name in list(self.__dict__.keys()):
                    val = self.__dict__[name]
                    if shape is None:
                        shape = val.shape
                    elif shape != val.shape:
                        raise ValueError("dimension mismatch between coefficient matrices")
                    term.append(val)
                else:
                    term.append(None)
                i += 1

            if not all(x is None for x in term):
                # fill empty places with good dimensions and construct output array
                # TODO F**K this shit. No better way for that?
                result_term = np.zeros(tuple([len(term)] + [dim for dim in shape]), dtype=np.complex)
                for idx, mat in enumerate(term):
                    if mat is None:
                        mat = np.zeros(shape)
                    result_term[idx, ...] = mat
            else:
                result_term = None

            terms.update({entry: np.real_if_close(result_term) if result_term is not None else None})

        return terms["E"], terms["f"], terms["G"]

    def convert_to_state_space(self):
        """
        convert the canonical ode system of order n a into an ode system of order 1.

        :return: py:class:StateSpace
        """
        e_mats, f, g_mats = self.get_terms()
        if f is not None:
            raise NotImplementedError
        if g_mats is not None:
            if g_mats.shape[0] > 1:
                # this would be temporal derivatives of the input
                raise NotImplementedError

        n = e_mats.shape[0]
        en_mat = e_mats[-1]
        rank_en_mat = np.linalg.matrix_rank(en_mat)
        if rank_en_mat != max(en_mat.shape) or en_mat.shape[0] != en_mat.shape[1]:
            raise ValueError("singular matrix provided")

        dim_x = en_mat.shape[0]  # length of the weight vector
        en_inv = np.linalg.inv(en_mat)

        new_dim = (n-1)*dim_x  # dimension of the new system
        a_mat = np.zeros((new_dim, new_dim))

        # compose new system matrix
        for idx, mat in enumerate(e_mats):
            if idx < n-1:
                if 0 < idx:
                    # add integrator chain
                    a_mat[(idx-1)*dim_x:idx*dim_x, idx*dim_x:(idx+1)*dim_x] = np.eye(dim_x)
                # add last row
                a_mat[-dim_x:, idx*dim_x:(idx+1)*dim_x] = np.dot(en_inv, -mat)

        # compose new input matrix
        if g_mats is not None:
            b_mat = np.zeros((new_dim, g_mats.shape[2]))
            for idx, mat in enumerate(g_mats):
                # build backwards since order of input derivatives is not constant
                b_mat[new_dim-(idx+1)*mat.shape[0]:new_dim-idx*mat.shape[0], :] = -np.dot(en_inv, mat)
        else:
            if self._input_function:
                raise ValueError("input function but no matrix, something wrong here!")

            # just an emtpy hull that makes no problems
            b_mat = np.zeros((new_dim, 1))

        return StateSpace(self.weights, a_mat, b_mat, input_handle=self.input_function)


class CanonicalForms(object):
    """
    wrapper that holds several entities of canonical forms for different sets of weights
    """
    def __init__(self, name):
        self.name = name
        self._dynamic_forms = {}
        self._static_form = CanonicalForm(self.name+"static")

    def add_to(self, weight_label, term, val):
        """
        add val to the canonical form for weight_label
        see add_to from :ref:py:class:CanonicalForm for details
        """
        if term[0] in "fg":
            # hold f and g vector separately
            self._static_form.add_to(term, val)
            return

        if weight_label not in list(self._dynamic_forms.keys()):
            self._dynamic_forms[weight_label] = CanonicalForm("_".join([self.name+weight_label]))

        self._dynamic_forms[weight_label].add_to(term, val)

    def get_static_terms(self):
        """
        return terms that do not depend on a certain weight set
        :return:
        """
        return self._static_form.get_terms()

    def get_dynamic_terms(self):
        """
        return dict of terms for each weight set
        :return:
        """
        return {label: val.get_terms() for label, val in self._dynamic_forms.items()}


def parse_weak_formulation(weak_form):
        """
        creates an ode system for the weights x_i based on the weak formulation.

        :return: simulation.ODESystem
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

            # field variable terms, sort into E_n, E_n-1, ..., E_0
            if placeholders["field_variables"]:
                if len(placeholders["field_variables"]) != 1:
                    raise NotImplementedError
                field_var = placeholders["field_variables"][0]
                temp_order = field_var.order[0]
                init_funcs = get_base(field_var.data["func_lbl"], field_var.order[1])

                if placeholders["inputs"]:
                    # TODO think about this case, is it relevant?
                    raise NotImplementedError

                # is the integrand a product?
                if placeholders["functions"]:
                    if len(placeholders["functions"]) != 1:
                        raise NotImplementedError
                    func = placeholders["functions"][0]
                    test_funcs = get_base(func.data["func_lbl"], func.order[1])
                    result = calculate_scalar_product_matrix(dot_product_l2, test_funcs, init_funcs)
                else:
                    # pull constant term out and compute integral
                    a = Scalars(np.atleast_2d([integrate_function(func, func.nonzero)[0] for func in init_funcs]))

                    if placeholders["scalars"]:
                        b = placeholders["scalars"][0]
                    else:
                        b = Scalars(np.ones_like(a.data.T))

                    result = _compute_product_of_scalars([a, b])

                cf.weights = field_var.data["weight_lbl"]
                cf.add_to(("E", temp_order), result*term.scale)
                continue

            # TestFunction Terms, those will end up in f
            if placeholders["functions"]:
                if not 1 <= len(placeholders["functions"]) <= 2:
                    raise NotImplementedError
                func = placeholders["functions"][0]
                test_funcs = get_base(func.data["func_lbl"], func.order[1])

                if len(placeholders["functions"]) == 2:
                    # TODO this computation is nonesense. Result must be a vektor conataining int of (tf1*tf2)
                    raise NotImplementedError

                    func2 = placeholders["functions"][1]
                    test_funcs2 = get_base(func2.data["func_lbl"], func2.order[2])
                    result = calculate_scalar_product_matrix(dot_product_l2, test_funcs, test_funcs2)
                    cf.add_to(("f", 0), result*term.scale)
                    continue

                if placeholders["scalars"]:
                    a = placeholders["scalars"][0]
                    b = Scalars(np.vstack([integrate_function(func, func.nonzero)[0]
                                           for func in test_funcs]))
                    result = _compute_product_of_scalars([a, b])
                    cf.add_to(get_scalar_target(placeholders["scalars"]), result*term.scale)
                    continue

                if placeholders["inputs"]:
                    if len(placeholders["inputs"]) != 1:
                        raise NotImplementedError
                    input_var = placeholders["inputs"][0]
                    input_func = input_var.data["input"]
                    input_index = input_var.data["index"]

                    # here we would need to provide derivative handles in the callable
                    input_order = input_var.order[0]
                    if input_order > 0:
                        raise NotImplementedError

                    result = np.array([integrate_function(func, func.nonzero)[0] for func in init_funcs])
                    cf.add_to(("G", input_order), result*term.scale, column=input_index)
                    cf.input_function = input_func
                    continue

            # pure scalar terms, sort into corresponding matrices
            if placeholders["scalars"]:
                result = _compute_product_of_scalars(placeholders["scalars"])
                target = get_scalar_target(placeholders["scalars"])

                if placeholders["inputs"]:
                    input_var = placeholders["inputs"][0]
                    input_func = input_var.data["input"]
                    input_index = input_var.data["index"]

                    # here we would need to provide derivative handles in the callable
                    input_order = input_var.order[0]
                    if input_order > 0:
                        raise NotImplementedError

                    # this would mean that the input term should appear in a matrix like E1 or E2
                    if target[0] == "E":
                        raise NotImplementedError

                    cf.add_to(("G", input_order), result*term.scale, column=input_index)
                    cf.input_function = input_func
                    continue

                cf.add_to(target, result*term.scale)
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


def simulate_state_space(state_space, initial_state, temp_domain):
    """
    wrapper to simulate a system given in state space form: :math:`\\dot{q} = Aq + Bu`

    :param state_space: state space formulation of the system
    :param initial_state: initial state vector of the system
    :param temp_domain: tuple of t_start and t_end
    :return:
    """
    if not isinstance(state_space, StateSpace):
        raise TypeError

    input_handle = state_space.input
    if input_handle is None:
        class EmptyInput(SimulationInput):
            def _calc_output(self, **kwargs):
                return np.zeros((state_space.B.shape[0], 1))
        input_handle = EmptyInput()
    if not isinstance(input_handle, SimulationInput):
        raise TypeError("only simulation.SimulationInput supported.")

    q = [initial_state]
    t = [temp_domain[0]]

    # TODO export cython code?
    def _rhs(_t, _q, a_mat, b_mat, u, lbl):
        q_t = np.dot(a_mat, _q) + np.dot(b_mat, u(time=_t, weights=_q, weight_lbl=lbl)).flatten()
        return q_t

    # TODO check for complex-valued matrices and use 'zvode'
    r = ode(_rhs).set_integrator("vode", max_step=temp_domain.step,
                                 method="adams",
                                 nsteps=1e3)

    r.set_f_params(state_space.A, state_space.B, input_handle, state_space.weight_lbl)
    r.set_initial_value(q[0], t[0])

    for t_step in temp_domain[1:]:
        if not r.successful():
            warnings.warn("*** Error: Simulation aborted at t={} ***".format(t_step))
            break

        t.append(t_step)
        q.append(r.integrate(t_step))

    # create results
    q = np.array(q)

    return Domain(points=np.array(t), step=temp_domain.step), q


def evaluate_approximation(base_label, weights, temp_domain, spat_domain, spat_order=0, name=""):
    """
    evaluate an approximation given by weights and functions at the points given in spatial and temporal steps

    :param weights: 2d np.ndarray where axis 1 is the weight index and axis 0 the temporal index
    :param base_label: functions to use for back-projection
    :param temp_domain: steps to evaluate at
    :param spat_domain: sim.Domain to evaluate at (or in)
    :param spat_order: spatial derivative order to use
    :param name: name to use
    :return: EvalData
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
