from functools import partial
import torch
import torch.nn as nn


def dict_map(fn, dic, leaf_type):
    # traverse a dict

    new_dict = {}
    for k, v in dic.items():
        if type(v) is dict:
            new_dict[k] = dict_map(fn, v, leaf_type)
        else:
            new_dict[k] = tree_map(fn, v, leaf_type)

    return new_dict


def tree_map(fn, tree, leaf_type):
    # traverse a complex tree and apply fn on each leaf_type
    # the tree structure will be remained.

    if isinstance(tree, dict):
        return dict_map(fn, tree, leaf_type)
    elif isinstance(tree, list):
        return [tree_map(fn, x, leaf_type) for x in tree]
    elif isinstance(tree, tuple):
        return tuple([tree_map(fn, x, leaf_type) for x in tree])
    elif isinstance(tree, leaf_type):
        return fn(tree)
    elif isinstance(tree, bool):
        return tree
    elif isinstance(tree, str):
        return tree
    elif isinstance(tree, int):
        return tree
    else:
        print(type(tree))
        raise ValueError("Not supported.")


tensor_tree_map = partial(tree_map, leaf_type=torch.Tensor)
