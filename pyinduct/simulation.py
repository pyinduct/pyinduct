# coding=utf-8
from __future__ import division
from abc import ABCMeta, abstractmethod
import numpy as np
from scipy.integrate import ode

from pyinduct import get_initial_functions, is_registered
from core import (Function, integrate_function, calculate_function_matrix,
                  project_on_initial_functions)
from placeholder import Scalars, TestFunction, Input, FieldVariable, EquationTerm, get_scalar_target
from utils import evaluate_approximation

__author__ = 'Stefan Ecklebe'


class SimulationInput(object):
    """
    base class for all objects that want to act as an input for the timestep simulation
    """
    __metaclass__ = ABCMeta

    def __init__(self):
        pass

    @abstractmethod
    def __call__(self, time, weights, **kwargs):
        """
        handle that will be used to retrieve input
        """
        pass


class Mixer(SimulationInput):
    """
    helper that represents a signal mixer
    """
    def __init__(self, inputs):
        SimulationInput.__init__(self)
        self._inputs = inputs

    def __call__(self, time, weights, **kwargs):
        outs = np.array([handle(time, weights) for handle in self._inputs])
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

    :param a_matrix: :math:`\\boldsymbol{A}`
    :param a_matrix: :math:`\\boldsymbol{B}`
    :param c_matrix: :math:`\\boldsymbol{C}`
    :param d_matrix: :math:`\\boldsymbol{D}`
    """
    def __init__(self, a_matrix, b_matrix, c_matrix=None, d_matrix=None):
        # TODO dimension checks
        self.A = a_matrix
        self.B = b_matrix
        self.C = c_matrix
        self.D = d_matrix


def simulate_systems(weak_forms, initial_states, time_interval, time_step, spatial_interval, spatial_step):
    """
    convenience wrapper for simulate system, see :ref:py:func:simulate_system for parameters
    :param weak_forms:
    :return:
    """
    return [simulate_system(sys, initial_states, time_interval, time_step, spatial_interval, spatial_step) for sys in
            weak_forms]


def simulate_system(weak_form, initial_states, time_interval, time_step, spatial_interval, spatial_step):
    """
    convenience wrapper that encapsulates the whole simulation process

    :param weak_form:
    :param initial_states: np.array of core.Functions for :math:`x(t=0, z), \\dot{x}(t=0, z), \\dotsc, x^{(n)}(t=0, z)`
    :param time_interval: tuple of (t_start and t_end)
    :return: tuple of integration time-steps and np.array of
    """
    print("simulating system: {0}".format(weak_form.name))
    if not isinstance(weak_form, WeakFormulation):
        raise TypeError("only WeakFormulation accepted.")

    initial_states = np.atleast_1d(initial_states)
    if not isinstance(initial_states[0], Function):
        raise TypeError("only core.Function accepted as initial state")

    if not isinstance(time_interval, tuple):
        raise TypeError("time_interval must be tuple")

    # parse input and create state space system
    print(">>> parsing formulation")
    canonical_form = parse_weak_formulation(weak_form)
    print(">>> creating state space system")
    state_space_form = canonical_form.convert_to_state_space()

    # calculate initial state
    print(">>> deriving initial conditions")
    q0 = np.array([project_on_initial_functions(initial_state, canonical_form.initial_functions) for initial_state in
                   initial_states]).flatten()

    # include boundary conditions
    # TODO

    # simulate
    print(">>> performing time step integration")
    t, q = simulate_state_space(state_space_form, canonical_form.input_function, q0, time_interval, time_step=time_step)

    # create handles and evaluate at given points
    # TODO also generate spatial derivatives here
    print(">>> performing postprocessing")
    data = []
    ini_funcs = get_initial_functions(canonical_form.initial_functions, 0)
    for der_idx in range(initial_states.size):
        data.append(evaluate_approximation(q[:, der_idx*ini_funcs.size:(der_idx+1)*ini_funcs.size],
                                           canonical_form.initial_functions,
                                           t, spatial_interval, spatial_step))
        data[-1].name = "{0}{1}".format(canonical_form.name,
                                        "_" + "".join(["d" for x in range(der_idx)] + ["t"]) if der_idx > 0 else "")

    # return results
    print("finished simulation.")
    return data


class CanonicalForm(object):
    """
    represents the canonical form of n ordinary differential equation system of order n
    """
    def __init__(self, name=None):
        self.name = name
        self._max_idx = dict(E=0, f=0, g=0)
        self._initial_functions = None
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
    def initial_functions(self):
        return self._initial_functions

    @initial_functions.setter
    def initial_functions(self, func_lbl):
        if not isinstance(func_lbl, str):
            raise TypeError("only string allowed as function label!")
        if self._initial_functions is None:
            self._initial_functions = func_lbl
        if self._initial_functions != func_lbl:
            raise ValueError("already defined initial functions are overridden!")

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

    def add_to(self, term, val):
        """
        adds the value val to term term
        :param term: tuple of name and index matrix(or vector) to add onto
        :param val: value to add
        :return:
        """
        if not isinstance(term, tuple):
            raise TypeError("term must be tuple.")
        if not isinstance(term[0], str) or term[0] not in "Efg":
            raise TypeError("term[0] must be string")
        if isinstance(term[1], int):
            name = self._build_name(term)
        else:
            raise TypeError("term index must be int")

        if not isinstance(val, np.ndarray):
            raise TypeError("val must be numpy.ndarray")

        # try to increment term
        try:
            entity = getattr(self, name)
            if entity.shape != val.shape:
                raise ValueError("{0} was already initialized with dimensions {1} but value to add has dimension {"
                                 "2}".format(name, entity.shape, val.shape))
            # add
            entity += val

        except AttributeError as e:
            # create entry
            setattr(self, name, np.copy(val))
        finally:
            self._max_idx[term[0]] = max(self._max_idx[term[0]], term[1])

    def get_terms(self):
        """
        construct a list of all terms that have indices and return tuple of lists
        :return: tuple of lists
        """
        terms = {}
        for entry in "Efg":
            term = []
            i = 0
            shape = None
            while i <= self._max_idx[entry]:
                name = self._build_name((entry, i))
                if name in self.__dict__.keys():
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
                result_term = np.zeros(tuple([len(term)] + [dim for dim in shape]))
                for idx, mat in enumerate(term):
                    if mat is None:
                        mat = np.zeros(shape)
                    result_term[idx, ...] = mat
            else:
                result_term = None

            terms.update({entry: result_term})

        return terms["E"], terms["f"], terms["g"]

    def convert_to_state_space(self):
        """
        takes a list of matrices that form a system of odes of order n and converts it into a ode system of order 1
        :return: tuple of (A, B)
        """
        e_mats, f, g = self.get_terms()
        if f is not None:
            raise NotImplementedError
        if g is not None:
            if g.shape[0] > 1:
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

        # compose new input vector
        b_vec = np.zeros((new_dim, 1))
        if g is not None:
            b_vec[-dim_x:] = np.dot(en_inv, -g[0])

        return StateSpace(a_mat, b_vec)


class CanonicalForms(object):
    """
    wrapper that holds several entities of canonical forms for different sets of weights
    """
    def __init__(self, name):
        self.name = name
        self._forms = {}

    def add_to(self, weight_label, term, val):
        """
        add val to the canonical form for weight_label
        see add_to from :ref:py:class:CanonicalForm for details
        """
        if weight_label not in self._forms.keys():
            self._forms[weight_label] = CanonicalForm("_".join([self.name+weight_label]))

        self._forms[weight_label].add_to(term, val)

    def get_terms(self):
        """
        return dict of terms for each weight set
        :return:
        """
        return {label: val.get_terms() for label, val in self._forms.iteritems()}


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
                init_funcs = get_initial_functions(field_var.data["func_lbl"], field_var.order[1])

                if placeholders["scalars"]:
                    a = Scalars(np.atleast_2d([integrate_function(func, func.nonzero)[0]
                                               for func in init_funcs]))
                    b = placeholders["scalars"][0]
                    result = _compute_product_of_scalars([a, b])

                elif placeholders["functions"]:
                    if len(placeholders["functions"]) != 1:
                        raise NotImplementedError
                    func = placeholders["functions"][0]
                    test_funcs = get_initial_functions(func.data["func_lbl"], func.order[1])
                    result = calculate_function_matrix(test_funcs, init_funcs)

                elif placeholders["inputs"]:
                    # TODO think about this
                    raise NotImplementedError

                else:
                    factors = np.atleast_2d([integrate_function(func, func.nonzero)[0] for func in init_funcs]).T
                    result = np.hstack(tuple([factors for i in range(factors.shape[0])]))

                cf.initial_functions = field_var.data["func_lbl"]
                cf.weights = field_var.data["weight_lbl"]
                cf.add_to(("E", temp_order), result*term.scale)
                continue

            if placeholders["functions"]:
                if not 1 <= len(placeholders["functions"]) <= 2:
                    raise NotImplementedError
                func = placeholders["functions"][0]
                test_funcs = get_initial_functions(func.data["func_lbl"], func.order[1])

                if len(placeholders["functions"]) == 2:
                    func2 = placeholders["functions"][1]
                    test_funcs2 = get_initial_functions(func2.data["func_lbl"], func2.order[2])
                    result = calculate_function_matrix(test_funcs, test_funcs2)
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
                    input_func = input_var.data
                    input_order = input_var.order

                    result = np.array([integrate_function(func, func.nonzero)[0] for func in init_funcs])
                    cf.add_to(("g", 0), result*term.scale)
                    cf.input_function = input_func
                    continue

            # pure scalar terms, sort into corresponding matrices
            if placeholders["scalars"]:
                result = _compute_product_of_scalars(placeholders["scalars"])
                target = get_scalar_target(placeholders["scalars"])

                if placeholders["inputs"]:
                    input_var = placeholders["inputs"][0]
                    input_func = input_var.data
                    input_order = input_var.order[0]

                    # this would mean that the input term should appear in a matrix like E1 or E2
                    if target[0] == "E":
                        raise NotImplementedError

                    cf.add_to(("g", 0), result*term.scale)
                    # TODO think of a more modular concept for input handling
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
        res = np.dot(scalars[1].data, scalars[0].data)
    else:
        raise NotImplementedError

    if res.shape[0] < res.shape[1]:
        return res.T
    return res


def simulate_state_space(state_space, input_handle, initial_state, time_interval, time_step=1e-2):
    """
    wrapper to simulate a system given in state space form: :math:`\\dot{q} = Aq + Bu`

    :param state_space: state space formulation of the system
    :param input_handle: function handle to evaluate input
    :param time_interval: tuple of t_start and t_end
    :return:
    """
    if not isinstance(state_space, StateSpace):
        raise TypeError
    if not isinstance(input_handle, SimulationInput):
        raise TypeError

    q = [initial_state]
    t = [time_interval[0]]

    def _rhs(t, q, a_mat, b_mat, u):
        q_t = np.dot(a_mat, q) + np.dot(b_mat, u(t, q)).flatten()
        return q_t

    r = ode(_rhs).set_integrator("vode", max_step=time_step)
    if input_handle is None:
        def input_handle(x):
            return 0

    r.set_f_params(state_space.A, state_space.B, input_handle)
    r.set_initial_value(initial_state, time_interval[0])

    precision = -int(np.log10(time_step))
    while r.successful() and np.round(r.t, precision) < time_interval[1]:
        t.append(r.t + time_step)
        q.append(r.integrate(r.t + time_step))

    # create results
    t = np.array(t)
    q = np.array(q)

    return t, q
