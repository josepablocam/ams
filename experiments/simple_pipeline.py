import importlib

import sklearn
import sklearn.pipeline


def get_component(comp_name):
    module_name = ".".join(comp_name.split(".")[:-1])
    class_name = comp_name.split(".")[-1]
    return getattr(importlib.import_module(module_name), class_name)()


def generate_pipeline(component_names, prefix="step_"):
    components = [get_component(name) for name in component_names]
    named_components = [(prefix + str(ix), comp)
                        for ix, comp in enumerate(components)]
    return sklearn.pipeline.Pipeline(named_components)
