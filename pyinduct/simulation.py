from __future__ import division
from abc import ABCMeta, abstractmethod
import numpy as np
from scipy.integrate import ode
from core import Function, sanitize_input, calculate_function_matrix_differential

__author__ = 'Stefan Ecklebe'

class Placeholder(object):
    """
    class that works as an placeholder for terms that are later substituted
    """
    def __init__(self, order=0, location=None):
        """
        :param order how many derivations are to be applied before evaluation
        :param location to evaluate at before further computation
        """
        if not isinstance(order, int) or order < 0:
            raise ValueError("invalid derivative order.")
        self.order = order

        if location is not None:
            if location and not isinstance(location, (int, long, float)):
                raise TypeError("location must be a number")
        self.location = location


class Scalars(Placeholder):
    """
    placeholder for scalars that will be replaced later
    """
    def __init__(self, values):
        Placeholder.__init__(self)
        self.values = sanitize_input(values, (int, long, float))


class TestFunctions(Placeholder):
    """
    class that works as a placeholder for test-functions in an equation
    """
    def __init__(self, functions, order=0, location=None):
        Placeholder.__init__(self, order, location)
        self.functions = sanitize_input(functions, Function)


class Input(Placeholder):
    """
    class that works as a placeholder for the input of a system
    """
    def __init__(self, function_handle, order=0):
        Placeholder.__init__(self)
        if not callable(function_handle):
            raise TypeError("callable object has to be provided.")
        self.handle = function_handle


class FieldVariable(Placeholder):
    """
    class that represents terms of the systems field variable x(z, t).
    since differentiation may occur, order can provide information about which derivative of the field variable.
    """
    def __init__(self, initial_functions, order=(0, 0), location=None):
        """
        :param : order tuple of temporal_order and spatial_order
        :param : factor
        :param : location
        """
        self.initial_functions = sanitize_input(initial_functions, Function)

        if not isinstance(order, tuple) or len(order) > 2:
            raise TypeError("order mus be 2-tuple of int.")
        if any([True for n in order if n < 0]):
            raise ValueError("derivative orders must be positive")
        if sum(order) > 2:
            raise ValueError("only derivatives of order one and two supported")
        if location is not None:
            if location and not isinstance(location, (int, long, float)):
                raise TypeError("location must be a number")

        self.order = order
        self.location = location


class TemporalDerivedFieldVariable(FieldVariable):
    def __init__(self, inital_functions, order, location=None):
        FieldVariable.__init__(self, inital_functions, (order, 0), location=location)


class SpatialDerivedFieldVariable(FieldVariable):
    def __init__(self, initial_functions, order, location=None):
        FieldVariable.__init__(self, initial_functions, (0, order), location=location)


class MixedDerivedFieldVariable(FieldVariable):
    def __init__(self, initial_functions, location=None):
        FieldVariable.__init__(self, initial_functions, (1, 1), location=location)

class Product(object):
    """
    represents a product
    """
    def __init__(self, a, b=None):
        if not isinstance(a, Placeholder) or (b is not None and not isinstance(b, Placeholder)):
            raise TypeError("argument not allowed in product")
        self.args = [a, b]

    def get_arg_by_class(self, cls):
        """
        extract element from product that is an instance of cls
        :return:
        """
        return [elem for elem in self.args if isinstance(elem, cls)]

class WeakEquationTerm(object):
    """
    base class for all accepted terms in a weak formulation
    """
    __metaclass__ = ABCMeta

    def __init__(self, scale, arg):
        if not isinstance(scale, (int, long, float)):
            raise TypeError("only numbers allowed as scale.")
        if not isinstance(arg, Product):
            if isinstance(arg, Placeholder):
                arg = Product(arg)
            else:
                raise TypeError("argument not supported.")

        self.scale = scale
        self.arg = arg

class ScalarTerm(WeakEquationTerm):
    """
    class that represents a scalar term in a weak equation
    """
    def __init__(self, argument, scale=1.0):
        WeakEquationTerm.__init__(self, scale, argument)

class IntegralTerm(WeakEquationTerm):
    """
    Class that represents an integral term in a weak equation
    """
    def __init__(self, integrand, limits, scale=1.0):
        WeakEquationTerm.__init__(self, scale, integrand)

        if any([True for arg in integrand.args if arg.location is not None]):
            raise ValueError("cannot integrate if integrand has to be evaluated first.")
        if not isinstance(limits, tuple):
            raise TypeError("limits must be provided as tuple")
        self.limits = limits


class SpatialIntegralTerm(IntegralTerm):
    def __init__(self, integrand, limits, scale=1.0):
        IntegralTerm.__init__(self, integrand, limits, scale)


class WeakFormulation(object):
    """
    this class represents the weak formulation of a spatial problem.
    It can be initialized with several terms (see children of WeakEquationTerm).
    The equation is interpreted as term_0 + term_1 + ... + term_N = 0

    :param terms: single (or list) of object(s) of type WeakEquationTerm
    """
    def __init__(self, terms):
        if isinstance(terms, WeakEquationTerm):
            terms = [terms]
        if not isinstance(terms, list):
            raise TypeError("only (list of) {0} allowed".format(WeakEquationTerm))

        for term in terms:
            if not isinstance(term, WeakEquationTerm):
                raise TypeError("Only WeakEquationTerm(s) are accepted.")

        self.terms = terms


def simulate_system(weak_form, initial_state, time_interval):
    """
    convenience wrapper that encapsulates the whole simulation process
    :param weak_form:
    :param initial_state:
    :param time_interval:
    :return:
    """

def parse_weak_formulation(weak_form):
        """
        creates an ode system for the weights x_i based on the weak formulation.
        :return: simulation.ODESystem
        """
        if not isinstance(weak_form, WeakFormulation):
            raise TypeError("only able to parse WeakFormulation")

        # if self.init_funcs.shape != self.test_funcs.shape:
        #     raise ValueError("dimensions of init- and test-functions do not match.")

        u_vector = None
        f_vector = None
        e_matrices = []

        # self._f = np.zeros((dim,))
        # self._E = [np.zeros((dim, dim)) for i in range(3)]

        # handle each term
        for term in weak_form.terms:
            # extract Placeholders
            scalars = term.arg.get_arg_by_class(Scalars)
            functions = term.arg.get_arg_by_class(TestFunctions)
            field_variables = term.arg.get_arg_by_class(FieldVariable)
            inputs = term.arg.get_arg_by_class(Input)

            locations = dict(scalars=[], functions=[], field_variables=[], inputs=[])
            if isinstance(term, ScalarTerm):
                # can be handled same as integral term but locations have to be extracted first
                for place_type in [scalars, functions, field_variables, inputs]:
                    # did instances of this type occur in term
                    if place_type:
                        for elem in place_type:
                            locations[place_type].append(elem.location)

            if isinstance(term, IntegralTerm):
                locations = None

            # handle most common terms
            if scalars:
                if not functions and not field_variables and not inputs:
                    # only scalar terms
                    result = np.prod(np.array([[val for val in scalar] for scalar in scalars]))*term.scale
                    ss_mats.f_vector += result

            # TODO move cases from Product case into functions and add them below
            if isinstance(term.arg, Product):
                funcs = term.arg.get_arg_by_class(TestFunctions)
                ders = term.arg.get_arg_by_class(FieldVariable)
                ins = term.arg.get_arg_by_class(Input)

                if len(ders) == 1:
                    temp_order = ders[0].order[0]
                    spat_order = ders[0].order[1]
                    der_loc = ders[0].location
                    # TODO handle Input as well
                    if len(funcs) == 1:
                        func_loc = funcs[0].location
                        if der_loc is None or func_loc is None:
                            raise ValueError("scalar term mus be evaluated, should be an integral otherwise.")
                        test_der_order = funcs[0].order
                        result = calculate_function_matrix_differential(self.init_funcs, self.test_funcs,
                                                                        spat_order, test_der_order,
                                                                        locations=(der_loc, func_loc))
                    else:
                        raise NotImplementedError
                    self._E[temp_order] += result*term.scale
                elif len(ins) == 1:
                    # since product contains two elements and a FieldDerivative was not in, other one is TestFunc
                    assert len(funcs) == 1
                    func_loc = funcs[0].location
                    if func_loc is None:
                        raise ValueError("scalar term mus be evaluated, should be an integral otherwise.")
                    test_der_order = funcs[0].order
                    result = np.asarray([func.derivative(test_der_order)(func_loc) for func in self.test_funcs])
                    self._f += result*term.scale
                else:
                    raise NotImplementedError

            elif isinstance(term, IntegralTerm):
                # TODO move cases from Product case into functions and add them below
                if isinstance(term.arg, Product):
                    funcs = term.arg.get_arg_by_class(TestFunctions)
                    ders = term.arg.get_arg_by_class(FieldVariable)
                    ins = term.arg.get_arg_by_class(Input)

                    if len(ders) == 1:
                        temp_order = ders[0].order[0]
                        spat_order = ders[0].order[1]
                        # TODO handle Input as well
                        if len(funcs) == 1:
                            test_der_order = funcs[0].order
                            result = calculate_function_matrix_differential(self.init_funcs, self.test_funcs,
                                                                            spat_order, test_der_order)
                        else:
                            raise NotImplementedError
                        self._E[temp_order] += result*term.scale
                    else:
                        raise NotImplementedError

            if False:
                print("f:")
                print(self._f)
                print("EO:")
                print(self._E[0])
                print("E1:")
                print(self._E[1])
                print("E2:")
                print(self._E[2])


def convert_to_state_space(coefficient_matrices, input_matrix):
    """
    takes a list of matrices that form a system of odes of order n.
      converts it into a ode system of order 1
    :param coefficient_matrices: list of len n
    :param input_matrix: numpy.ndarray
    :return: tuple of (A, B)
    """
    n = len(coefficient_matrices)
    en_mat = coefficient_matrices[-1]
    rank_en_mat = np.linalg.matrix_rank(en_mat)
    if rank_en_mat != max(en_mat.shape) or en_mat.shape[0] != en_mat.shape[1]:
        raise ValueError("singular matrix provided")

    dim_x = en_mat.shape[0]  # length of the weight vector
    en_inv = np.linalg.inv(en_mat)

    new_dim = (n-1)*dim_x  # dimension of the new system
    a_mat = np.zeros((new_dim, new_dim))

    # compose new system matrix
    for idx, mat in enumerate(coefficient_matrices):
        if idx < n-1:
            if 0 < idx:
                # add integrator chain
                a_mat[(idx-1)*dim_x:idx*dim_x, idx*dim_x:(idx+1)*dim_x] = np.eye(dim_x)
            # add last row
            a_mat[-dim_x:, idx*dim_x:(idx+1)*dim_x] = np.dot(en_inv, -mat)

    b_vec = np.zeros((new_dim, ))
    b_vec[-dim_x:] = np.dot(en_inv, -input_matrix)

    return a_mat, b_vec

def simulate_system(system_matrix, input_vector, input_handle, initial_state, time_interval, t_step=1e-2):
    """
    wrapper to simulate a system given in state space form: q_dot = A*q + B*u
    :param system_matrix: A
    :param input_vector: B
    :param input_handle: function handle to evaluate input
    :param time_interval: tuple of t_start an t_end
    :return:
    """
    q = []
    t = []

    def _rhs(t, q, a_mat, b_vec, u):
        q_t = np.dot(a_mat, q) + np.dot(b_vec, u(t))

    r = ode(_rhs).set_integrator("vode", max_step=t_step)
    r.set_f_params(system_matrix, input_vector, input_handle)
    r.set_initial_value(initial_state, time_interval[0])

    while r.successful() and r.t < time_interval[1]:
        t.append(r.t)
        q.append(r.integrate(r.t + t_step))

    # create results
    t = np.array(t)
    q = np.array(q)

    return t, q
    q = q[:, :self._nodes.shape[0]]
