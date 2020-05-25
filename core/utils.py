import importlib


def get_component_constructor(comp):
    path_steps = comp.split(".")
    module_steps, basename = path_steps[:-1], path_steps[-1]
    module_path = ".".join(module_steps)
    return getattr(importlib.import_module(module_path), basename)
