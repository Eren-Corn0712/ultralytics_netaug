import torch
import torch.nn as nn


def get_parent_class_name(module):
    """
    Get the name of the first parent class of the given object.

    Args:
        module: The object for which to get the parent class name.

    Returns:
        The name of the first parent class of the given object as a string, including the module name and class name.

    Raises:
        IndexError: If the object does not have a parent class.

    """
    try:
        # Get the first parent class of the object.
        parent_class = module.__class__.__bases__[0]
    except IndexError:
        # Raise an error if the object has no parent class.
        raise IndexError("Object has no parent class.")

    # Build the parent class name string using the module name and class name.
    parent_class_name = f"{parent_class.__module__}.{parent_class.__name__}"

    return parent_class_name
