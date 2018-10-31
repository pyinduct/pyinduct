import sympy as sp
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
        return self._new_variables(names, dependencies, implementations, category, **kwargs)


global_variable_pool = VariablePool("GLOBAL")


def pprint(expr):
    sp.pprint(expr, num_columns=180)

