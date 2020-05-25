import sklearn
import sklearn.pipeline
import zss


def get_obj_label(obj):
    module_str = obj.__class__.__module__
    class_str = obj.__class__.__name__
    return module_str + "." + class_str


def convert_obj_to_node(obj):
    obj_label = get_obj_label(obj)
    obj_node = zss.Node(obj_label)

    params = list(obj.get_params().items())
    # sort by param name to guarantee order
    params = sorted(params, key=lambda x: x[0])

    for param_name, param_obj in params:
        if isinstance(param_obj, sklearn.pipeline.Pipeline):
            param_node = to_tree(param_obj)
        elif sklearn.base.is_classifier(param_obj):
            param_node = convert_obj_to_node(param_obj)
        else:
            param_node = zss.Node("{}_{}".format(param_name, param_obj))
        obj_node.addkid(param_node)
    return obj_node


def to_tree(pipeline):
    root = zss.Node("root")
    curr = root
    if isinstance(pipeline, sklearn.pipeline.FeatureUnion):
        steps = pipeline.transformer_list
    else:
        steps = pipeline.steps
    for step_name, step_obj in steps:
        step_node = convert_obj_to_node(step_obj)
        curr.addkid(step_node)
        curr = step_node
    return root


def to_json(node):
    assert isinstance(node, zss.Node)
    return {node.label: [to_json(c) for c in node.children]}


def binary_dist(d1, d2):
    return 1.0 if d1 != d2 else 0.0


def tree_edit_distance(pipeline1, pipeline2):
    if not isinstance(pipeline1, zss.Node):
        pipeline1 = to_tree(pipeline1)
    if not isinstance(pipeline2, zss.Node):
        pipeline2 = to_tree(pipeline2)
    # just binary on labels
    return zss.simple_distance(pipeline1, pipeline2, label_dist=binary_dist)
