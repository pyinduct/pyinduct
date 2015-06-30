from __future__ import division
from abc import ABCMeta, abstractmethod
import numpy as np
from scipy.integrate import ode
from core import Function, sanitize_input, integrate_function, calculate_function_matrix_differential

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

        if any([True for arg in self.arg.args if getattr(arg, "location", None) is not None]):
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

class CanonicalForm(object):
    """
    represents the canonical form of n ordinary differential equation system of order n
    """
    def __init__(self):
        self._max_idx = 0

    def _build_name(self, term):
        return "_"+term[0]+str(term[1])

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
            raise TypeError("val must b numpy.ndarray")

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
            self._max_idx = max(self._max_idx, term[1])

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
            while i <= self._max_idx:
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
                result_term = np.zeros((len(term), shape[0], shape[1]))
                for idx, mat in enumerate(term):
                    if mat is None:
                        mat = np.zeros(shape)
                    result_term[idx, ...] = mat
            else:
                result_term = None

            terms.update({entry: result_term})

        return terms["E"], terms["f"], terms["g"]

def parse_weak_formulation(weak_form):
        """
        creates an ode system for the weights x_i based on the weak formulation.
        :return: simulation.ODESystem
        """
        if not isinstance(weak_form, WeakFormulation):
            raise TypeError("only able to parse WeakFormulation")

        cf = CanonicalForm()

        # handle each term
        for term in weak_form.terms:
            # extract Placeholders
            placeholders = dict(scalars=term.arg.get_arg_by_class(Scalars),
                                functions=term.arg.get_arg_by_class(TestFunctions),
                                field_variables=term.arg.get_arg_by_class(FieldVariable),
                                inputs=term.arg.get_arg_by_class(Input))

            # field variable terms, sort into E_n, E_n-1, ..., E_0
            if placeholders["field_variables"]:
                if len(placeholders["field_variables"]) != 1:
                    raise NotImplementedError
                field_var = placeholders["field_variables"][0]
                field_loc = field_var.location
                temp_order = field_var.order[0]
                spat_order = field_var.order[1]
                init_funcs = np.asarray([func.derivative(spat_order) for func in field_var.initial_functions])
                result = None

                if placeholders["scalars"]:
                    factors = _compute_product_of_scalars(placeholders["scalars"])
                    if field_loc:
                        column = np.multiply(np.array([func(field_loc) for func in init_funcs]),
                                             factors)
                    else:
                        column = np.multiply(np.array([integrate_function(func, func.nonzero)[0] for func in
                                                       init_funcs]),
                                             factors)
                    # cvt to matrix
                    result = np.array([column, ]*init_funcs.shape[0]).transpose()

                elif placeholders["functions"]:
                    if len(placeholders["functions"]) != 1:
                        raise NotImplementedError
                    func = placeholders["function"][0]
                    test_funcs = func.test_functions
                    func_loc = func.location
                    func_order = func.order
                    if field_loc:
                        if func_loc:
                            factors = np.array([func(func_loc) for func in test_funcs])
                            column = np.multiply(np.array([func(field_loc) for func in init_funcs]), factors)
                            result1 = np.array([column, ]*test_funcs.shape[0]).transpose()
                            result2 = calculate_function_matrix_differential(init_funcs, test_funcs,
                                                                             0, func_order,
                                                                             locations=(field_loc, func_loc))
                            # lets see
                            assert np.allclose(resul1, result2)
                            result = result2
                    else:
                        if func_loc:
                            factors = np.array([func(func_loc) for func in test_funcs])
                            column = np.multiply(np.array([integrate_function(func, func.nonzero)[0] for func in
                                                           init_funcs]),
                                                 factors)
                            result = np.array([column, ]*test_funcs.shape[0]).transpose()
                        else:
                            result = calculate_function_matrix_differential(init_funcs, test_funcs,
                                                                            0, func_order)
                elif placeholders["inputs"]:
                    raise NotImplementedError

                    # TODO think about this
                    if len(placeholders["inputs"]) != 1:
                        raise NotImplementedError
                    input_var = placeholders["inputs"][0]
                    input_func = input_var.handle
                    input_order = input_var.order
                    if field_loc:
                        result = np.array([func(field_loc) for func in init_funcs])
                    else:
                        result = np.array([integrate_function(func, func.nonzero)[0] for func in init_funcs])
                else:
                    if field_loc:
                        column = np.array([func(field_loc) for func in init_funcs])
                    else:
                        column = np.array([integrate_function(func, func.nonzero)[0] for func in init_funcs])

                    # result = np.zeros((init_funcs.shape[0], init_funcs.shape[0])) + column
                    result = np.array([column, ]*init_funcs.shape[0]).transpose()

                cf.add_to(("E", temp_order), result*term.scale)

            # TODO non-field variable terms
            continue

        return cf

            # # TODO move cases from Product case into functions and add them below
            # if isinstance(term.arg, Product):
            #     funcs = term.arg.get_arg_by_class(TestFunctions)
            #     ders = term.arg.get_arg_by_class(FieldVariable)
            #     ins = term.arg.get_arg_by_class(Input)
            #
            #     if len(ders) == 1:
            #         temp_order = ders[0].order[0]
            #         spat_order = ders[0].order[1]
            #         der_loc = ders[0].location
            #         # TODO handle Input as well
            #         if len(funcs) == 1:
            #             func_loc = funcs[0].location
            #             if der_loc is None or func_loc is None:
            #                 raise ValueError("scalar term mus be evaluated, should be an integral otherwise.")
            #             test_der_order = funcs[0].order
            #             result = calculate_function_matrix_differential(self.init_funcs, self.test_funcs,
            #                                                             spat_order, test_der_order,
            #                                                             locations=(der_loc, func_loc))
            #         else:
            #             raise NotImplementedError
            #         self._E[temp_order] += result*term.scale
            #     elif len(ins) == 1:
            #         # since product contains two elements and a FieldDerivative was not in, other one is TestFunc
            #         assert len(funcs) == 1
            #         func_loc = funcs[0].location
            #         if func_loc is None:
            #             raise ValueError("scalar term mus be evaluated, should be an integral otherwise.")
            #         test_der_order = funcs[0].order
            #         result = np.asarray([func.derivative(test_der_order)(func_loc) for func in self.test_funcs])
            #         self._f += result*term.scale
            #     else:
            #         raise NotImplementedError
            #
            # elif isinstance(term, IntegralTerm):
            #     # TODO move cases from Product case into functions and add them below
            #     if isinstance(term.arg, Product):
            #         funcs = term.arg.get_arg_by_class(TestFunctions)
            #         ders = term.arg.get_arg_by_class(FieldVariable)
            #         ins = term.arg.get_arg_by_class(Input)
            #
            #         if len(ders) == 1:
            #             temp_order = ders[0].order[0]
            #             spat_order = ders[0].order[1]
            #             # TODO handle Input as well
            #             if len(funcs) == 1:
            #                 test_der_order = funcs[0].order
            #                 result = calculate_function_matrix_differential(self.init_funcs, self.test_funcs,
            #                                                                 spat_order, test_der_order)
            #             else:
            #                 raise NotImplementedError
            #             self._E[temp_order] += result*term.scale
            #         else:
            #             raise NotImplementedError
            #
            # if False:
            #     print("f:")
            #     print(self._f)
            #     print("EO:")
            #     print(self._E[0])
            #     print("E1:")
            #     print(self._E[1])
            #     print("E2:")
            #     print(self._E[2])

def _compute_product_of_scalars(scalars):
    if len(scalars) == 1:
        return scalars[0].values
    values = [[val for val in scalar.values] for scalar in scalars]
    return np.prod(np.array(values))

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
