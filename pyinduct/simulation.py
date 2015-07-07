# coding=utf-8
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
    def __init__(self, data, order=(0, 0), location=None):
        """
        :param order how many derivations are to be applied before evaluation (t, z)
        :param location to evaluate at before further computation
        """
        self.data = data
        if not isinstance(order, tuple) or any([not isinstance(o, int) or o < 0 for o in order]):
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
    def __init__(self, values, target_term=None):
        values = np.atleast_2d(values)
        Placeholder.__init__(self, sanitize_input(values, (int, long, float)))
        self.target_term = target_term


class TestFunctions(Placeholder):
    """
    class that works as a placeholder for test-functions in an equation
    """
    def __init__(self, functions, order=0, location=None):
        Placeholder.__init__(self, sanitize_input(functions, Function), (0, order), location)


class Input(Placeholder):
    """
    class that works as a placeholder for the input of a system
    """
    def __init__(self, function_handle, order=0):
        if not callable(function_handle):
            raise TypeError("callable object has to be provided.")
        Placeholder.__init__(self, function_handle, order=(order, 0))


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
        if not isinstance(order, tuple) or len(order) > 2:
            raise TypeError("order mus be 2-tuple of int.")
        if any([True for n in order if n < 0]):
            raise ValueError("derivative orders must be positive")
        if sum(order) > 2:
            raise ValueError("only derivatives of order one and two supported")
        if location is not None:
            if location and not isinstance(location, (int, long, float)):
                raise TypeError("location must be a number")

        Placeholder.__init__(self, sanitize_input(initial_functions, Function), order=order, location=location)


class TemporalDerivedFieldVariable(FieldVariable):
    def __init__(self, initial_functions, order, location=None):
        FieldVariable.__init__(self, initial_functions, (order, 0), location=location)


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
        if b is None:  # multiply by one as Default
            if isinstance(a, Input):
                b = Scalars(np.ones(1))
            if isinstance(a, Scalars):
                if a.target_term[0] == "E":
                    b = Scalars(np.ones(a.data.T.shape))
                else:
                    b = Scalars(np.ones(a.data.shape))

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
        self.scale = scale

        # convenience: convert single argument
        if not isinstance(arg, Product):
            if isinstance(arg, Placeholder):
                # arg = Product(arg)
                arguments = [arg]
            else:
                raise TypeError("argument not supported.")
        else:
            arguments = arg.args

        # evaluate all terms that can be evaluated
        new_args = []
        for idx, arg in enumerate(arguments):
            if getattr(arg, "location", None) is not None:
                # evaluate term and add scalar
                # print("WARNING: converting Placeholder that is to be evaluated into 'Scalars' object.")
                new_args.append(_evaluate_placeholder(arg))
            else:
                new_args.append(arg)

        self.arg = Product(*new_args)

class ScalarTerm(WeakEquationTerm):
    """
    class that represents a scalar term in a weak equation
    """
    def __init__(self, argument, scale=1.0):
        WeakEquationTerm.__init__(self, scale, argument)

        if any([True for arg in self.arg.args if isinstance(arg, (FieldVariable, TestFunctions))]):
            raise ValueError("cannot leave z dependency. specify location to evaluate expression.")


class IntegralTerm(WeakEquationTerm):
    """
    Class that represents an integral term in a weak equation
    """
    def __init__(self, integrand, limits, scale=1.0):
        WeakEquationTerm.__init__(self, scale, integrand)

        if not any([isinstance(arg, (FieldVariable, TestFunctions)) for arg in self.arg.args]):
            raise ValueError("nothing to integrate")
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
    pass

class CanonicalForm(object):
    """
    represents the canonical form of n ordinary differential equation system of order n
    """
    def __init__(self):
        self._max_idx = dict(E=0, f=0, g=0)

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
                # F**K this shit. No better way for that?
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
        takes a list of matrices that form a system of odes of order n.
          converts it into a ode system of order 1
        :return: tuple of (A, B)
        """
        e_mats, f, g = self.get_terms()
        if f is not None:
            raise NotImplementedError
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

        b_vec = np.zeros((new_dim, 1))
        b_vec[-dim_x:] = np.dot(en_inv, -g[0])

        return a_mat, b_vec


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
                temp_order = field_var.order[0]
                spat_order = field_var.order[1]
                init_funcs = np.asarray([func.derivative(spat_order) for func in field_var.data])
                result = None

                if placeholders["scalars"]:
                    # TODO move into seperate function
                    a = Scalars(np.atleast_2d(np.array([integrate_function(func, func.nonzero)[0]
                                                        for func in init_funcs])).T)
                    b = placeholders["scalars"][0]
                    result = _compute_product_of_scalars([a, b])

                elif placeholders["functions"]:
                    if len(placeholders["functions"]) != 1:
                        raise NotImplementedError
                    func = placeholders["functions"][0]
                    test_funcs = func.data
                    func_order = func.order[1]
                    result = calculate_function_matrix_differential(init_funcs, test_funcs, 0, func_order)

                elif placeholders["inputs"]:
                    raise NotImplementedError

                    # TODO think about this
                    # if len(placeholders["inputs"]) != 1:
                    #     raise NotImplementedError
                    # input_var = placeholders["inputs"][0]
                    # input_func = input_var.handle
                    # input_order = input_var.order
                    # if field_loc:
                    #     result = np.array([func(field_loc) for func in init_funcs])
                    # else:
                    #     result = np.array([integrate_function(func, func.nonzero)[0] for func in init_funcs])
                else:
                    factors = np.atleast_2d([integrate_function(func, func.nonzero)[0] for func in init_funcs]).T
                    result = np.hstack(tuple([factors for i in range(factors.shape[0])]))

                cf.add_to(("E", temp_order), result*term.scale)
                continue

            # TODO non-field variable terms
            if placeholders["functions"]:
                if not 1 <= len(placeholders["functions"]) <= 2:
                    raise NotImplementedError
                func = placeholders["functions"][0]
                func_order = func.order[0]
                test_funcs = np.asarray([func.derivative(func_order) for func in func.data])

                if len(placeholders["functions"]) == 2:
                    func2 = placeholders["functions"][1]
                    test_funcs2 = func.data
                    func_order2 = func.order[1]
                    result = calculate_function_matrix_differential(test_funcs, test_funcs2,
                                                                    0, func_order2)
                    cf.add_to(("f", 0), result*term.scale)
                    continue

                if placeholders["scalars"]:
                    a = placeholders["scalars"][0]
                    b = Scalars(np.atleast_2d(np.array([integrate_function(func, func.nonzero)[0]
                                                        for func in test_funcs])))
                    result = _compute_product_of_scalars([a, b])

                    # factors = _compute_product_of_scalars(placeholders["scalars"])
                    # column = np.multiply(np.array([integrate_function(func, func.nonzero)[0] for func in test_funcs]),
                    #                      factors)
                    # result = np.array([column, ]*test_funcs.shape[0]).transpose()
                    cf.add_to(_get_scalar_target(placeholders["scalars"]), result*term.scale)
                    continue

                if placeholders["inputs"]:
                    if len(placeholders["inputs"]) != 1:
                        raise NotImplementedError
                    input_var = placeholders["inputs"][0]
                    # TODO think were to put input handle
                    input_func = input_var.handle
                    input_order = input_var.order

                    result = np.array([integrate_function(func, func.nonzero)[0] for func in init_funcs])
                    cf.add_to(("g", 0), result*term.scale)
                    continue

            # pure scalar terms, sort into corresponding matrices
            if placeholders["scalars"]:
                result = _compute_product_of_scalars(placeholders["scalars"])
                target = _get_scalar_target(placeholders["scalars"])

                if placeholders["inputs"]:
                    if target[0] == "E":
                        # this would mean that the input term should appear in a matrix like E1 or E2
                        raise NotImplementedError
                    cf.add_to(("g", 0), result*term.scale)
                    continue

                cf.add_to(target, result*term.scale)
                continue

        return cf

def _get_scalar_target(scalars):
    """
    extract target fro list of scalars.
    makes sure that targets are equivalent.
    :param scalars:
    :return:
    """
    targets = [elem for elem in [getattr(ph, "target_term", None) for ph in scalars] if elem]
    if targets:
        if targets[1:] != targets[:-1]:
            # since scalars are evaluated separately prefer E for f
            residual = filter(lambda x: x[0] != "f", targets)
            if len(residual) > 1:
                # different temporal derivatives of state -> not supported
                raise ValueError("target_term of scalars in product must be identical")
        return targets[0]

    return None

def _evaluate_placeholder(placeholder):
    """
    evaluates a placeholder object and returns a Scalars object
    :param placeholder:
    :return:
    """
    if not isinstance(placeholder, Placeholder):
        raise TypeError("only placeholders supported")
    if isinstance(placeholder, (Scalars, Input)):
        raise TypeError("provided type cannot be evaluated")

    functions = np.asarray([func.derivative(placeholder.order[1]) for func in placeholder.data])
    location = placeholder.location
    values = np.atleast_2d([func(location) for func in functions])

    if isinstance(placeholder, FieldVariable):
        return Scalars(values.T, target_term=("E", placeholder.order[0]))
    elif isinstance(placeholder, TestFunctions):
        return Scalars(values, target_term=("f", 0))
    else:
        raise NotImplementedError

def _compute_product_of_scalars(scalars):
    if len(scalars) > 2:
        raise NotImplementedError

    if len(scalars) == 1:
        res = scalars[0].data
    elif scalars[0].data.shape == scalars[1].data.shape:
        res = np.prod(np.array([scalars[0].data, scalars[1].data]), axis=0)
    elif scalars[0].data.shape == scalars[1].data.T.shape:
        res = np.dot(scalars[0].data, scalars[1].data)

    if res.shape[0] < res.shape[1]:
        return res.T
    return res

def _simulate_system(system_matrix, input_vector, input_handle, initial_state, time_interval, t_step=1e-2):
    """
    wrapper to simulate a system given in state space form: q_dot = A*q + B*u
    :param system_matrix: A
    :param input_vector: B
    :param input_handle: function handle to evaluate input
    :param time_interval: tuple of t_start and t_end
    :return:
    """
    q = []
    t = []

    def _rhs(t, q, a_mat, b_vec, u):
        q_t = np.dot(a_mat, q) + np.dot(b_vec, u(t))
        return q_t

    r = ode(_rhs).set_integrator("vode", max_step=t_step)
    r.set_f_params(system_matrix, input_vector.flatten(), input_handle)
    r.set_initial_value(initial_state, time_interval[0])

    while r.successful() and r.t < time_interval[1]:
        t.append(r.t)
        q.append(r.integrate(r.t + t_step))

    # create results
    t = np.array(t)
    q = np.array(q)

    return t, q
