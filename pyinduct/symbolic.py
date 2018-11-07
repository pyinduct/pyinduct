import sympy as sp
import numpy as np
import mpmath
import sys
from tqdm import tqdm
import collections
import pyinduct as pi
from pyinduct.core import domain_intersection, integrate_function
from pyinduct.simulation import simulate_state_space
from sympy.utilities.lambdify import implemented_function

__all__ = ["VariablePool"]


class VariablePool:
    variable_pool_registry = dict()

    def __init__(self, description):
        if description in self.variable_pool_registry:
            raise ValueError("Variable pool '{}' already exists.".format(description))

        self.variable_pool_registry.update({description: self})
        self.description = description
        self.variables = dict()
        self.categories = dict()
        self.categories.update({None: list()})

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

    name = "_dummy{}".format(dummy_counter)
    dummy_counter += 1

    return global_variable_pool._new_variable(
        name, dependcy, implementation, "dummies", **kwargs)


def new_dummy_variables(dependcies, implementations, **kwargs):
    dummies = list()

    for dependcy, implementation in zip(dependcies, implementations):
        dummies.append(new_dummy_variable(dependcy, implementation, **kwargs))

    return dummies


def pprint(expr, description=None, n=None, limit=4, num_columns=180):
    """
    Wraps sympy.pprint adds description to the console output
    (if given) and the availability of hiding the output if
    the approximation order exceeds a given approximation order.

    Args:
        expr (sympy.Expr): Sympy expression to pprint.
        description (str): Description of the sympy expression to pprint.
        n (int): Current approximation order, default None, means
            :code:`limit` will be ignored.
        limit (int): Limit approximation order, default 4.
        num_columns (int): Kwarg :code:`num_columns` of sympy.pprint,
            default 180.
    """
    if n is not None and n > limit:
        return

    else:
        if description is not None:
            print("\n {}".format(description))

        sp.pprint(expr, num_columns=num_columns)


class SimulationInputWrapper:
    """
    Wraps a :py:class:`.SimulationInput` into a callable, for further use
    as sympy implemented function (input function) and call during
    the simulation, see :py:class:`.simulate_system`.
    """
    def __init__(self, sim_input):
        from pyinduct.simulation import SimulationInput
        assert isinstance(sim_input, SimulationInput)

        self._sim_input = sim_input

    def __call__(self, kwargs):
        return self._sim_input(**kwargs)


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
    # check if all simulation input symbols have only one
    # depended variable and uniqueness of it
    input_var = input_syms[0].args[0]
    assert all([len(sym.args) == 1 for sym in input_syms])
    assert all([input_var == sym.args[0] for sym in input_syms])

    # check length of args
    n = len(pi.get_base(base_label))
    assert all([n == len(it) for it in [init_conds, funcs]])

    # dictionary / kwargs for the pyinuct simulation input call
    _input_var = dict(time=float(0), weights=np.zeros(n), weight_lbl=base_label)

    # derive callable from the symbolic expression of the right hand side
    print(">>> lambdify right hand side")
    rhs_lam = sp.lambdify((funcs, time_sym, input_var), rhs, modules="numpy")
    print("done!")

    def _rhs(_t, _q):
        _input_var["time"] = _t
        _input_var["weights"] = _q

        return rhs_lam(_q, _t, _input_var)

    return simulate_state_space(_rhs, init_conds, temp_domain, settings)


def evaluate_integrals(expression, presc=mpmath.mp.dps):
    expr_expand = sp.N(expression.expand(), presc)

    replace_dict = dict()
    for integral in tqdm(expr_expand.atoms(sp.Integral), file=sys.stdout,
                         desc=">>> evaluate integrals (dps={})".format(presc)):

        if not len(integral.args[1]) == 3:
            raise ValueError(
                "Only the evaluation of definite integrals is implemented.")

        integrand = integral.args[0]
        dependent_var, limit_a, limit_b = integral.args[1]
        all_funcs = integrand.atoms(sp.Function)
        impl_funcs = {func for func in all_funcs if hasattr(func, "_imp_")}

        if len(impl_funcs) == 0:
            replace_dict.update({integrand: integrand.doit()})

        elif isinstance(integrand, (sp.Mul, sp.Function)):

            constants = list()
            dependents = list()
            if isinstance(integrand, sp.Mul):
                for arg in integrand.args:
                    if dependent_var in arg.free_symbols:
                        dependents.append(arg)

                    else:
                        constants.append(arg)

            elif isinstance(integrand, sp.Function):
                dependents.append(integrand)

            else:
                raise NotImplementedError

            assert len(dependents) != 0
            assert np.prod([sym for sym in constants + dependents]) == integrand

            # collect numeric implementation of all
            # python and pyinduct functions
            py_funcs = list()
            pi_funcs = list()
            prove_integrand = np.prod(constants)
            domain = (float(limit_a), float(limit_b))
            for func in dependents:

                # check: only one sympy function in expression
                _func = func.atoms(sp.Function)
                assert len(_func) == 1

                # check: only one dependent variable
                __func = _func.pop()
                assert len(__func.args) == 1

                # check: correct dependent variable
                assert __func.args[0] == dependent_var

                # determine derivative order
                if isinstance(func, sp.Derivative):
                    der_order = func.args[1][1]

                else:
                    der_order = 0

                # check if we understand things right
                prove_integrand *= sp.diff(__func, dependent_var, der_order)

                # categorize _imp_ in python and pyinduct functions
                implementation = func.atoms(sp.Function).pop()._imp_
                if isinstance(implementation, pi.Function):
                    domain = domain_intersection(domain, implementation.nonzero)
                    pi_funcs.append((implementation, int(der_order)))

                else:
                    if der_order != 0:
                        raise NotImplementedError(
                            "Only derivatives of a pyinduct.Function"
                            "can be aquired.")

                    py_funcs.append(implementation)

            # check if things will be processed correctly
            assert sp.Integral(
                prove_integrand, (dependent_var, limit_a, limit_b)) == integral

            # function to integrate
            def _integrand(z, py_funcs=py_funcs, pi_funcs=pi_funcs):
                mul = ([f(z) for f in py_funcs] +
                       [f.derive(ord)(z) for f, ord in pi_funcs])

                return np.prod(mul)

            _integral = integrate_function(_integrand, domain)[0]
            result = np.prod([sym for sym in constants + [_integral]])

            replace_dict.update({integral: sp.N(result, presc)})

        else:
            raise NotImplementedError

    print("done!")

    return expr_expand.xreplace(replace_dict)


def derive_first_order_representation(expression, funcs, input_,
                                      mode="sympy.solve"):

    # make sure funcs depends on one varialble only
    assert len(funcs.free_symbols) == 1
    depvar = funcs.free_symbols.pop()

    if mode == "sympy.solve":
        # use sympy solve for rewriting
        print(">>> rewrite  as c' = f(c,u)")
        sol = sp.solve(expression, sp.diff(funcs, depvar))
        rhs = sp.Matrix([sol[it] for it in sp.diff(funcs, depvar)])
        print("done!")

        return rhs

    elif mode == "sympy.linear_eq_to_matrix":
        # rewrite expression as E1 * c' + E0 * c + G * u = 0
        print(">>> rewrite as E1 c' + E0 c + G u = 0")
        E1, _expression = sp.linear_eq_to_matrix(expression,
                                                 list(sp.diff(funcs, depvar)))
        expression = (-1) * _expression
        E0, _expression = sp.linear_eq_to_matrix(expression, list(funcs))
        expression = (-1) * _expression
        G, _expression = sp.linear_eq_to_matrix(expression, list(input_))
        assert _expression == _expression * 0
        print("done!")

        # rewrite expression as c' = A c + B * u
        print(">>> rewrite as c' = A c + B u")
        E1_inv = E1.inv()
        A = -E1_inv * E0
        B = -E1_inv * G
        print("done!")

        return A * funcs + B * input_


def implement_as_linear_ode(rhs, funcs, input_):

    # make sure funcs depends on one varialble only
    assert len(funcs.free_symbols) == 1
    depvar = funcs.free_symbols.pop()

    A, _rhs = sp.linear_eq_to_matrix(rhs, list(funcs))
    _rhs *= -1
    B, _rhs = sp.linear_eq_to_matrix(_rhs, list(input_))
    assert _rhs == _rhs * 0
    assert len(A.atoms(sp.Symbol, sp.Function)) == 0
    assert len(B.atoms(sp.Symbol, sp.Function)) == 0

    A_num = np.array(A).astype(float)
    B_num = np.array(B).astype(float)

    def __rhs(c, u):
        return A_num @ c + B_num @ u

    return new_dummy_variable((funcs, input_), __rhs)
