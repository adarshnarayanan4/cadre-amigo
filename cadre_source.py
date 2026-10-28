"""Run CADRE's numerical methods without loading the OpenMDAO framework.

This is a validation helper. It reads the methods from the selected checkout,
keeps their equations intact, and replaces framework base classes with a small
container. It does not run OpenMDAO setup, derivatives, integration, or drivers.
"""

import ast
from functools import lru_cache
from math import sqrt

import numpy as np

from cadre_paths import cadre_path


class _KernelComponent:
    def __init__(self, *args, **kwargs):
        self.options = {}


@lru_cache(maxsize=None)
def _namespace(path):
    tree = ast.parse(path.read_text(), filename=str(path))
    body = []

    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            continue

        if isinstance(node, ast.ClassDef):
            node.bases = [ast.Name(id="_KernelComponent", ctx=ast.Load())]

        body.append(node)

    namespace = {"np": np, "sqrt": sqrt, "_KernelComponent": _KernelComponent}

    if path.name != "kinematics.py":
        namespace.update(_namespace(cadre_path("kinematics.py")))

    module = ast.fix_missing_locations(ast.Module(body=body, type_ignores=[]))
    exec(compile(module, str(path), "exec"), namespace)
    return namespace


def source_class(module, name):
    """Read a numerical component class directly from OpenMDAO/CADRE."""
    return _namespace(cadre_path(f"{module}.py"))[name]
