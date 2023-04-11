import torch

from typing import List, Optional, Union, Any


def random_sample(src_list: List[Any], generator: torch.Generator = None, k: int = 1) -> Union[List[Any], Any]:
    """
    Randomly selects k elements from the given list.

    Args:
        src_list (List[Any]): A list of elements to select from.
        generator (torch.Generator, optional): Generator used for random sampling. Defaults to None.
        k (int, optional): The number of elements to select. Defaults to 1.

    Returns:
        Union[List[Any], Any]: A list of k randomly selected elements from the given list.
                              If k=1, returns a single randomly selected element.
    """
    rand_idx = torch.randint(len(src_list), size=(k,), generator=generator)
    out_list = [src_list[i] for i in rand_idx]
    return out_list[0] if k == 1 else out_list


def create_linear_sequence(min_val, max_val, num_points):
    """
    Create a linear sequence of numbers using PyTorch.

    Args:
        min_val (float): The minimum value of the sequence.
        max_val (float): The maximum value of the sequence.
        num_points (int): The number of points to include in the sequence.

    Returns:
        torch.Tensor: A tensor containing the linear sequence of numbers.
    """
    seq = torch.linspace(min_val, max_val, num_points)
    return seq


def count_grad_parameters(model):
    """
    Counts the number of parameters that have a non-zero gradient value.

    Args:
        model (nn.Module): PyTorch model.

    Returns:
        int: The number of parameters with non-zero gradients.
    """
    device = next(model.parameters()).device
    num_grad_params = torch.zeros(1, device=device)
    for param in model.parameters():
        if param.grad is not None:  # Check if the gradient is not None.
            num_grad_params += torch.count_nonzero(param.grad)  # Count the number of non-zero gradient values.

    return num_grad_params.item()
