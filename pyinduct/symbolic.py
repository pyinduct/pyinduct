import warnings
import sympy as sp
import numpy as np
import sys
from tqdm import tqdm
import collections
import pyinduct as pi
from pyinduct.core import domain_intersection, integrate_function
from pyinduct.simulation import simulate_state_space, SimulationInput
from sympy.utilities.lambdify import implemented_function

__all__ = ["VariablePool"]


class VariablePool:
    registry = dict()

    def __init__(self, description):
        if description in self.registry:
            raise ValueError("Variable pool '{}' already exists.".format(description))

        self.registry.update({description: self})
        self.description = description
        self.variables = dict()
        self.categories = dict()
        self.categories.update({None: list()})

    def __getitem__(self, item):
        if len(self.categories[item]) == 0:
            return None

        elif len(self.categories[item]) == 1:
            return self.categories[item][0]

        else:
            return self.categories[item]

    def _new_variable(self, name, dependency, implementation, category, **kwargs):
        assert isinstance(name, str)

        if name in self.variables:
            raise ValueError("Name '{}' already in variable pool.".format(name))

        if dependency is None and implementation is None:
            variable = sp.Symbol(name, **kwargs)

        elif implementation is None:
            assert isinstance(dependency, collections.Iterable)
            variable = sp.Function(name, **kwargs)(*dependency)

        elif callable(implementation):
            variable = implemented_function(name, implementation, **kwargs)(*dependency)

        else:
            raise NotImplementedError

        self.variables.update({name: variable})

        if category not in self.categories:
            self.categories.update({category: list()})

        self.categories[category].append(variable)

        return variable

    def _new_variables(self, names, dependencies, implementations, category, **kwargs):
        assert isinstance(names, collections.Iterable)

        if dependencies is None:
            dependencies = [None] * len(names)
        if implementations is None:
            implementations = [None] * len(names)

        assert len(names) == len(dependencies)
        assert len(names) == len(implementations)

        variables = list()
        for name, dependency, implementation in zip(names, dependencies, implementations):
            variables.append(self._new_variable(name, dependency, implementation, category, **kwargs))

        return variables

    def new_symbol(self, name, category, **kwargs):
        return self._new_variable(name, None, None, category, **kwargs)

    def new_symbols(self, names, category, **kwargs):
        return self._new_variables(names, None, None, category, **kwargs)

    def new_function(self, name, dependency, category, **kwargs):
        return self._new_variable(name, dependency, None, category, **kwargs)

    def new_functions(self, names, dependencies, category, **kwargs):
        return self._new_variables(names, dependencies, None, category, **kwargs)

    def new_implemented_function(self, name, dependency, implementation, category, **kwargs):
        return self._new_variable(name, dependency, implementation, category, **kwargs)

    def new_implemented_functions(self, names, dependencies, implementations, category, **kwargs):
        return self._new_variables(names, dependencies, implementations,
                                   category, **kwargs)


global_variable_pool = VariablePool("GLOBAL")

dummy_counter = 0


def new_dummy_variable(dependcy, implementation, **kwargs):
    global dummy_counter

    name = "_pyinduct_dummy{}".format(dummy_counter)
    dummy_counter += 1

    return global_variable_pool._new_variable(
        name, dependcy, implementation, "dummies", **kwargs)


def new_dummy_variables(dependcies, implementations, **kwargs):
    dummies = list()

    for dependcy, implementation in zip(dependcies, implementations):
        dummies.append(new_dummy_variable(dependcy, implementation, **kwargs))

    return dummies


def pprint(expr, description=None, n=None, limit=4, num_columns=180,
           discard_small_values=False, tolerance=1e-6):
    """
    Wraps sympy.pprint, adds description to the console output
    (if given) and the availability of hiding the output if
    the approximation order exceeds a given limit.

    Args:
        expr (sympy.Expr or array-like): Sympy expression or list of sympy
            expressions to pprint.
        description (str): Description of the sympy expression to pprint.
        n (int): Current approximation order, default None, means
            :code:`limit` will be ignored.
        limit (int): Limit approximation order, default 4.
        num_columns (int): Kwarg :code:`num_columns` of sympy.pprint,
            default 180.
        discard_small_values (bool): If true: round numbers < tolerance to 0.
            Default: false.
        tolerance (float): Applies when discard_small_values is true.
            Default: 1e-6.
    """
    if n is not None and n > limit:
        return

    else:
        if description is not None:
            print("\n>>> {}".format(description))

        if discard_small_values:
            # this is not clever or perfomant, but short
            expr = sp.nsimplify(expr, tolerance=tolerance, rational=True).n()

        sp.pprint(expr, num_columns=num_columns)


class SimulationInputWrapper:
    """
    Wraps a :py:class:`.SimulationInput` into a callable, for further use
    as sympy implemented function (input function) and call during
    the simulation, see :py:class:`.simulate_system`.
    """
    def __init__(self, sim_input):
        assert isinstance(sim_input, SimulationInput)

        self._sim_input = sim_input

    def __call__(self, kwargs):
        return self._sim_input(**kwargs)


# TODO: find a better name for this class
class SymbolicFeedback(SimulationInput):

    def __init__(self, expression, base_weights_info, name=str(), args=None):
        SimulationInput.__init__(self, name=name)

        self.feedback_gains = dict()
        for lbl, vec in base_weights_info.items():
            gain, _expression = sp.linear_eq_to_matrix(sp.Matrix([expression]), list(vec))
            expression = (-1) * _expression
            self.feedback_gains.update({lbl: np.array(gain).astype(float)})

        self.remaining_terms = None
        if not expression == expression * 0:
            if args is None:
                raise ValueError("The feedback law holds variables, which "
                                 "could not be sort into the linear feedback "
                                 "gains. Provide the weights variable 'weights' "
                                 "and the time variabel 't' as tuple over the "
                                 "'args' argument.")

            # TODO: check that 'expression' depends only on 'args'
            elif False:
                pass

            else:
                self.remaining_terms = sp.lambdify(args, expression, "numpy")


    def _calc_output(self, **kwargs):
        """
        Calculates the controller output based on the current_weights and time.

        Keyword Args:
            weights: Current weights of the simulations system approximation.
            weights_lbl (str): Corresponding label of :code:`weights`.
            time (float): Current simulation time.

        Return:
            dict: Controller output :math:`u`.
        """

        # linear feedback u = k^T * x
        res = self.feedback_gains[kwargs["weight_lbl"]] @ kwargs["weights"]

        # add constant, nonlinear and other crazy terms
        if self.remaining_terms is not None:
            res += self.remaining_terms(kwargs["weights"], kwargs["time"])

        return dict(output=res)


def simulate_system(rhs, funcs, init_conds, base_label, input_syms,
                    time_sym, temp_domain, settings=None):
    """
    Simulate finite dimensional ode according to the provided
    right hand side (:code:`rhs`)

    .. math:: \partial_t c(t) = f(c(t), u(t))

    Args:
        rhs (sympy.Matrix): Vector :math:`f(c(t), u(t))`
        funcs (sympy.Matrix): Vector: :math:`c(t)`
        init_conds (array-like): Vector:
            :math:`c(t_0), \quad t_0 = \text{temp_domain[0]}`
        base_label (str): Label of a finite dimension base
            :math:`\varphi_i, i=1,...,n` which is registered with the module
            :py:mod:`pyinduct.registry`.
        input_syms (array-like): List of system input symbols/
            implemented functions :math:`u(t)`, see
            :py:class:`.SimulationInputWrapper`.
        time_sym (sympy.Expr): Symbol the variable :math:`t`.
        temp_domain (.Domain): Temporal domain.
        **settings: Kwargs will be passed through to scipy.integrate.ode.

    Returns:
        See :py:func:`.simulate_state_space`.
    """
    # argument for the dictionary by a pyinuct simulation input call
    input_arg = input_syms[0].args[0]

    # dictionary / kwargs for the pyinuct simulation input call
    _input_arg = dict(time=0, weights=init_conds, weight_lbl=base_label)

    # check if all simulation input symbols have only one
    # depended variable and uniqueness of it
    assert all([len(sym.args) == 1 for sym in input_syms])
    assert all([input_arg == sym.args[0] for sym in input_syms])

    # check length of args
    n = len(pi.get_base(base_label))
    assert all([n == len(it) for it in [init_conds, funcs]])

    # check if all inputs holds an SimulationInputWrapper as implementation
    assert all(isinstance(inp._imp_, SimulationInputWrapper) for inp in list(input_syms))

    # derive callable from the symbolic expression of the right hand side
    print("\n>>> lambdify right hand side")
    rhs_lam = sp.lambdify((funcs, time_sym, input_arg), rhs, modules="numpy")

    # check callback
    assert len(rhs_lam(init_conds, 0, _input_arg)) == n

    def _rhs(_t, _q):
        _input_arg["time"] = _t
        _input_arg["weights"] = _q

        return rhs_lam(_q, _t, _input_arg)

    return simulate_state_space(_rhs, init_conds, temp_domain, settings)


def evaluate_implemented_functions(expression):

    der_replace_dict = dict()
    for der in expression.atoms(sp.Derivative):

        # only derivatives will be processed which holds
        # exact one sympy function
        if len(der.atoms(sp.Function)) != 1:
            continue

        # skip if the function is not the only argument of the derivative
        func = der.atoms(sp.Function).pop()
        if der.args[0] != func:
            continue

        # skip if the function has no implementation
        if not hasattr(func, "_imp_"):
            continue

        # skip if the function has more or less than one dependent variable
        if len(func.args) != 1:
            continue

        # determine derivative order
        der_order = get_derivative_order(der)

        imp = func._imp_
        if isinstance(imp, pi.Function):
            new_imp = imp.derive(der_order)
            dummy_der = new_dummy_variable(func.args, new_imp)
            der_replace_dict.update({der: dummy_der})

        elif callable(imp):
            raise NotImplementedError(
                "Only derivatives of a pyinduct.Function "
                "can be aquired.")

        else:
            raise NotImplementedError

    # replace all derived implemented pyinduct functions
    # with a dummy function which holds the derivative as implementation
    expr_without_derivatives = expression.xreplace(der_replace_dict)

    # evaluate if possible
    evaluated_expression = expr_without_derivatives.doit().n()

    # undo replace if the derivative could not be evaluated
    reverse_replace = dict([(v, k) for k, v in der_replace_dict.items()])

    return evaluated_expression.xreplace(reverse_replace)


def evaluate_integrals(expression):
    expr_expand = expression.expand()

    replace_dict = dict()
    for integral in tqdm(expr_expand.atoms(sp.Integral), file=sys.stdout,
                         desc="\n>>> evaluate integrals"):

        if not len(integral.args[1]) == 3:
            raise ValueError(
                "Only the evaluation of definite integrals is implemented.")

        integrand = integral.args[0]
        dependent_var, limit_a, limit_b = integral.args[1]
        all_funcs = integrand.atoms(sp.Function)
        impl_funcs = {func for func in all_funcs if hasattr(func, "_imp_")}

        if len(impl_funcs) == 0:
            replace_dict.update({integral: integral.doit()})

        elif isinstance(integrand, (sp.Mul, sp.Function, sp.Derivative)):

            constants = list()
            dependents = list()
            if isinstance(integrand, sp.Mul):
                for arg in integrand.args:
                    if dependent_var in arg.free_symbols:
                        dependents.append(arg)

                    else:
                        constants.append(arg)

            elif isinstance(integrand, (sp.Function, sp.Derivative)):
                dependents.append(integrand)

            else:
                raise NotImplementedError

            assert len(dependents) != 0
            assert np.prod([sym for sym in constants + dependents]) == integrand

            # collect numeric implementation of all
            # python and pyinduct functions
            py_funcs = list()
            pi_funcs = list()
            prove_integrand = sp.Integer(1)
            domain = {(float(limit_a), float(limit_b))}
            prove_replace = dict()
            for func in dependents:

                # check: maximal one free symbol
                free_symbol = func.free_symbols
                assert len(free_symbol) <= 1

                # check: free symbol is the integration variable
                if len(free_symbol) == 1:
                    assert free_symbol.pop() == dependent_var

                # if this term is not a function try to lambdify the function
                # and implement the lambdified function
                if len(func.atoms(sp.Function)) == 0:
                    lam_func = sp.lambdify(dependent_var, func)
                    orig_func = func
                    func = new_dummy_variable((dependent_var,), lam_func)
                    prove_replace.update({func: orig_func})

                # check: only one sympy function in expression
                _funcs = func.atoms(sp.Function)
                assert len(_funcs) == 1

                # check: only one dependent variable
                _func = _funcs.pop()
                assert len(_func.args) == 1

                # check: correct dependent variable
                assert _func.args[0] == dependent_var

                # determine derivative order
                if isinstance(func, sp.Derivative):
                    der_order = get_derivative_order(func)

                else:
                    der_order = 0

                # for a semantic check
                prove_integrand *= sp.diff(_func, dependent_var, der_order)

                # categorize _imp_ in python and pyinduct functions
                implementation = _func._imp_
                if isinstance(implementation, pi.Function):
                    domain = domain_intersection(domain, implementation.nonzero)
                    pi_funcs.append((implementation, int(der_order)))

                elif callable(implementation):
                    if der_order != 0:
                        raise NotImplementedError(
                            "Only derivatives of a pyinduct.Function "
                            "can be aquired.")

                    py_funcs.append(implementation)

                else:
                    raise NotImplementedError

            # check if things will be processed correctly
            prove_integrand = np.prod(
                [sym for sym in constants + [prove_integrand]])
            assert sp.Integral(
                prove_integrand, (dependent_var, limit_a, limit_b)
            ).xreplace(prove_replace) == integral

            # function to integrate
            def _integrand(z, py_funcs=py_funcs, pi_funcs=pi_funcs):
                mul = ([f(z) for f in py_funcs] +
                       [f.derive(ord)(z) for f, ord in pi_funcs])

                return np.prod(mul)

            _integral = integrate_function(_integrand, domain)[0]
            result = np.prod([sym for sym in constants + [_integral]])

            replace_dict.update({integral: result})

        else:
            raise NotImplementedError

    return expr_expand.xreplace(replace_dict)


def derive_first_order_representation(expression, funcs, input_,
                                      mode="sympy.solve",
                                      interim_results=None):

    # make sure funcs depends on one varialble only
    assert len(funcs.free_symbols) == 1
    depvar = funcs.free_symbols.pop()

    if mode == "sympy.solve":
        # use sympy solve for rewriting
        print("\n>>> rewrite  as c' = f(c,u)")
        sol = sp.solve(expression, sp.diff(funcs, depvar))
        rhs = sp.Matrix([sol[it] for it in sp.diff(funcs, depvar)])

        return rhs

    elif mode == "sympy.linear_eq_to_matrix":
        # rewrite expression as E1 * c' + E0 * c + G * u = 0
        print("\n>>> rewrite as E1 c' + E0 c + G u = 0")
        E1, _expression = sp.linear_eq_to_matrix(expression,
                                                 list(sp.diff(funcs, depvar)))
        expression = (-1) * _expression
        E0, _expression = sp.linear_eq_to_matrix(expression, list(funcs))
        expression = (-1) * _expression
        G, _expression = sp.linear_eq_to_matrix(expression, list(input_))
        assert _expression == _expression * 0

        # rewrite expression as c' = A c + B * u
        print("\n>>> rewrite as c' = A c + B u")
        if len(E1.atoms(sp.Symbol, sp.Function)) == 0:
            E1_num = np.array(E1).astype(float)
            E1_inv = sp.Matrix(np.linalg.inv(E1_num))

        else:
            warnings.warn("Since the matrix E1 depends on symbol(s) and/or \n"
                          "function(s) the method sympy.Matrix.inv() was \n"
                          "used. Check result! (numpy.linalg.inv() is more \n"
                          "reliable)")
            E1_inv = E1.inv()

        A = -E1_inv * E0
        B = -E1_inv * G

        if interim_results is not None:
            interim_results.update({
                "E1": E1, "E0": E0, "G": G, "A": A, "B": B,
            })

        return A * funcs + B * input_


def implement_as_linear_ode(rhs, funcs, input_):

    A, _rhs = sp.linear_eq_to_matrix(rhs, list(funcs))
    _rhs *= -1
    B, _rhs = sp.linear_eq_to_matrix(_rhs, list(input_))
    assert _rhs == _rhs * 0
    assert len(A.atoms(sp.Symbol, sp.Function)) == 0
    assert len(B.atoms(sp.Symbol, sp.Function)) == 0

    A_num = np.array(A).astype(float)
    B_num = np.array(B).astype(float)

    def __rhs(c, u):
        # since u.dim = 3
        return A_num @ c + B_num @ u[0]

    return new_dummy_variable((funcs, input_), __rhs)


def get_derivative_order(derivative):
    # its really a derivative?
    assert isinstance(derivative, sp.Derivative)

    der_order = derivative.args[1][1]

    # its really a integer value?
    assert der_order == int(der_order)

    return int(der_order)
