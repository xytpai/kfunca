import kfunca
import numpy as np

print(kfunca.__file__)


class TestTensorImpl(object):
    def test_tensor_impl(self):
        arr = np.random.uniform(-10, 10, size=(2, 3))
        print(arr)
        arr_gpu = kfunca.from_numpy(arr, 0)
        print(arr_gpu)
        arr_gpu_cpu = kfunca.to_numpy(arr_gpu)
        print(arr_gpu_cpu)
        assert(np.allclose(arr, arr_gpu_cpu) == True)
    
    def test_tensor_add(self):
        for shape in ((2,3), (1000), (12,11,3331)):
            print(shape)
            arr = np.random.uniform(-10, 10, size=shape).astype(np.float32)
            arr_2 = arr + arr
            arr_gpu = kfunca.from_numpy(arr, 0)
            arr_gpu_2 = kfunca.add(arr_gpu, arr_gpu)
            arr_gpu_2_cpu = kfunca.to_numpy(arr_gpu_2)
            assert(np.allclose(arr_2, arr_gpu_2_cpu) == True)
