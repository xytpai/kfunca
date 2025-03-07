import kfunca
import numpy as np


class TestTensorImpl(object):
    def test_tensor_impl(self):
        arr = np.random.uniform(-10, 10, size=(2, 3))
        print(arr)
        arr_gpu = kfunca.from_numpy(arr, 0)
        print(arr_gpu)
        arr_gpu_cpu = kfunca.to_numpy(arr_gpu)
        print(arr_gpu_cpu)
        assert(np.allclose(arr, arr_gpu_cpu, rtol=1e-05, atol=1e-08, equal_nan=False) == True)
