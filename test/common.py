import kfunca
import torch
import numpy as np


def assert_allclose(tensor_a, tensor_b, atol=1e-3, rtol=1e-3):
    if not isinstance(tensor_a, np.ndarray):
        tensor_a = tensor_a.contiguous().numpy()
    if not isinstance(tensor_b, np.ndarray):
        tensor_b = tensor_b.contiguous().numpy()
    assert(np.allclose(tensor_a, tensor_b, rtol=atol, atol=rtol) == True)
