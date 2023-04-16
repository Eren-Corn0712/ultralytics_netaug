import torch
import torch.nn as nn
from typing import List, Optional, Union, Any, Tuple


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


def sort_param(
        param: nn.Parameter,
        dim: int,
        sorted_idx: torch.Tensor,
) -> None:
    """
    Sorts the values of a tensor parameter along a given dimension based on the
    specified sorted indices tensor. This function modifies the parameter in-place.

    Args:
        param (nn.Parameter): The parameter tensor to be sorted.
        dim (int): The dimension along which to sort the tensor.
        sorted_idx (torch.Tensor): The sorted indices tensor to use for sorting.

    Raises:
        ValueError: If `dim` is out of bounds for the parameter tensor.

    Example usage:
        >>> # Sorting a 2D tensor along its second dimension:
        >>> p = nn.Parameter(torch.tensor([[3, 4, 1], [2, 5, 6]]))
        >>> sort_param(p, 1, torch.tensor([[2, 0, 1], [1, 0, 2]]))
        >>> print(p)
        tensor([[1, 3, 4],
                [2, 5, 6]], requires_grad=True)
    """
    if not isinstance(param, (nn.Parameter, torch.Tensor)):
        raise ValueError("Input tensor must be an instance of `nn.Parameter` `torch.Tensor`.")

    if dim >= param.dim() or dim < -param.dim():
        raise ValueError(f"Invalid dimension index {dim} for tensor with shape {param.shape}")

    # Use `index_select` to gather the sorted values along the specified dimension.
    sorted_param = torch.index_select(param.data, dim, sorted_idx)

    # Replace the data of the original parameter tensor with the sorted data.
    param.data.copy_(sorted_param.clone().detach())


def calc_importance(x: torch.Tensor, dim: Union[int, Tuple[int], List[int]]) -> torch.Tensor:
    """
    Calculates the importance of the input tensor along one or more dimensions by computing the L1 norm
    of the tensor elements along those dimensions, and returns the indices that sort the elements
    in descending order of magnitude.

    Args:
        x (torch.Tensor): The input tensor.
        dim (int, tuple or list): The dimension(s) along which to calculate the importance.

    Returns:
        torch.Tensor: A 1D tensor containing the indices that sort the tensor elements in
        descending order of magnitude along the specified dimension(s).

    Raises:
        ValueError: If `dim` is out of range for the input tensor.

    Example usage:
        >>> # Computing the importance of the feature maps of a convolutional layer along the channel dimension:
        >>> conv = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        >>> x = torch.randn(1, 16, 32, 32)
        >>> sorted_idx = calc_importance(x=conv.weight.data, dim=(1,))
    """
    if isinstance(dim, int):
        dim = (dim,)

    if isinstance(x, nn.Parameter):
        x = x.data

    if max(dim) >= x.ndim or min(dim) < -x.ndim:
        raise ValueError(f"At least one dimension is out of range for input tensor with shape {x.shape}.")

    # Compute the L1 norm of the tensor elements along the specified dimension(s).
    importance = torch.norm(x, p=1, dim=dim)

    # Get the indices that sort the tensor elements in descending order of magnitude.
    sorted_idx = torch.argsort(importance, descending=True)

    return sorted_idx
