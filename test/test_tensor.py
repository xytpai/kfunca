import kfunca
import numpy as np

print(kfunca.__file__)


class TestTensorImpl(object):
    def test_tensor_impl(self):
        arr = np.random.uniform(-10, 10, size=(2, 3))
        arr_gpu = kfunca.from_numpy(arr, 0)
        assert(np.allclose(arr, arr_gpu.numpy()) == True)
    
    def test_tensor_add(self):
        for shape in ((2,3), (1000), (12,11,3331)):
            arr = np.random.uniform(-10, 10, size=shape).astype(np.float32)
            arr_2 = arr + arr
            arr_gpu = kfunca.from_numpy(arr, 0)
            arr_gpu_2 = arr_gpu + arr_gpu
            arr_gpu_2_cpu = arr_gpu_2.numpy()
            assert(np.allclose(arr_2, arr_gpu_2_cpu) == True)
            arr1 = np.random.uniform(-10, 10, size=shape).astype(np.int32)
            arr2 = np.random.uniform(-10, 10, size=shape).astype(np.float32)
            out = arr1 + arr2
            out_gpu = kfunca.from_numpy(arr1, 0) + kfunca.from_numpy(arr2, 0)
            assert(np.allclose(out, out_gpu.numpy()) == True)

    def test_data_ptr(self):
        import copy
        arr_ = np.random.uniform(-10, 10, size=(3,4)).astype(np.float32)
        arr_x = kfunca.from_numpy(arr_, 0)
        arr_x_ref = kfunca.from_numpy(arr_, 0)
        arr_x_ref = arr_x
        arr_x_deep = copy.deepcopy(arr_x)
        assert arr_x.data_ptr() == arr_x_ref.data_ptr() == arr_x_deep.data_ptr()
        assert arr_x.storage_ref_count() == arr_x_ref.storage_ref_count() == arr_x_deep.storage_ref_count() == 2
        del arr_x
        assert arr_x_deep.storage_ref_count() == 2
        assert arr_x_ref.storage_ref_count() == 2
        del arr_x_ref
        assert arr_x_deep.storage_ref_count() == 1

    def test_broadcast_basic_binary(self):
        shapes = [
            [[16, 1], [1, 6], 'easy'],
            [[162, 1, 345], [162, 6, 1], 'easy'],
            [[123, 1, 567], [123, 127, 567], 'easy'],
            [[2, 1024, 1024, 512], [2, 1024, 1, 512], 'hard'],
            [[2, 1024, 1024, 512], [2, 1024, 1024, 512], 'hard'],
        ]
        op_ = ['+', '-', '*', '/']
        for shape in shapes:
            for op in op_:
                if not ((shape[2] == 'hard') and (op != '+')):
                    print(op, shape)
                    arr1 = np.random.uniform(-10, 10, size=shape[0]).astype(np.float32)
                    arr2 = np.random.uniform(-10, 10, size=shape[1]).astype(np.float32)
                    out = eval("arr1 {} arr2".format(op))
                    out_gpu = eval("kfunca.from_numpy(arr1, 0) {} kfunca.from_numpy(arr2, 0)".format(op))
                    assert(np.allclose(out, out_gpu.numpy()) == True)
                    arr1 = np.random.uniform(-10, 10, size=shape[0]).astype(np.int32)
                    arr2 = np.random.uniform(-10, 10, size=shape[1]).astype(np.float32)
                    out = eval("arr1 {} arr2".format(op))
                    out_gpu = eval("kfunca.from_numpy(arr1, 0) {} kfunca.from_numpy(arr2, 0)".format(op))
                    assert(np.allclose(out, out_gpu.numpy()) == True)
    
    def test_reduce(self):
        for dim in [0,1,2]:
            arr = np.random.uniform(-10, 10, size=[223,23,3213]).astype(np.float32)
            arr_sum = np.sum(arr, axis=dim, keepdims=True)
            arr_gpu = kfunca.from_numpy(arr, 0)
            arr_gpu_sum = arr_gpu.sum(dim)
            assert(np.allclose(arr_sum, arr_gpu_sum.numpy(), rtol=1e-1, atol=1e-1) == True)


if __name__ == '__main__':
    test_instance = TestTensorImpl()
    for method_name in dir(test_instance):
        if method_name.startswith("test_"):
            method = getattr(test_instance, method_name)
            print(f"Running {method_name}...")
            method()
