import sympy as sp
import numpy as np
import collections
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
                    time_sym, temp_domain, **settings):
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
    n = len(init_conds)
    assert all([n == len(it) for it in [rhs, funcs]])

    # dictionary / kwargs for the pyinuct simulation input call
    _input_var = dict(time=float(0), weights=np.zeros(n), weight_lbl=base_label)

    # derive callable from the symbolic expression of the right hand side
    rhs_lam = sp.lambdify((funcs, time_sym, input_var), rhs)

    def _rhs(_t, _q):
        _input_var["time"] = _t
        _input_var["weights"] = _q

        return rhs_lam(_q, _t, _input_var)

    from pyinduct.simulation import simulate_state_space

    return simulate_state_space(_rhs, init_conds, temp_domain, **settings)





