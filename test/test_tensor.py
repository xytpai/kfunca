import kfunca
import numpy as np

print(kfunca.__file__)


class TestTensorImpl(object):
    def test_tensor_impl(self):
        arr = np.random.uniform(-10, 10, size=(2, 3))
        arr_gpu = kfunca.from_numpy(arr, 0)
        arr_gpu_cpu = kfunca.to_numpy(arr_gpu)
        assert(np.allclose(arr, arr_gpu_cpu) == True)
    
    def test_tensor_add(self):
        for shape in ((2,3), (1000), (12,11,3331)):
            arr = np.random.uniform(-10, 10, size=shape).astype(np.float32)
            arr_2 = arr + arr
            arr_gpu = kfunca.from_numpy(arr, 0)
            arr_gpu_2 = arr_gpu + arr_gpu
            arr_gpu_2_cpu = kfunca.to_numpy(arr_gpu_2)
            assert(np.allclose(arr_2, arr_gpu_2_cpu) == True)
            arr1 = np.random.uniform(-10, 10, size=shape).astype(np.int32)
            arr2 = np.random.uniform(-10, 10, size=shape).astype(np.float32)
            out = arr1 + arr2
            out_gpu = kfunca.from_numpy(arr1, 0) + kfunca.from_numpy(arr2, 0)
            assert(np.allclose(out, kfunca.to_numpy(out_gpu)) == True)

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
                    assert(np.allclose(out, kfunca.to_numpy(out_gpu)) == True)
                    arr1 = np.random.uniform(-10, 10, size=shape[0]).astype(np.int32)
                    arr2 = np.random.uniform(-10, 10, size=shape[1]).astype(np.float32)
                    out = eval("arr1 {} arr2".format(op))
                    out_gpu = eval("kfunca.from_numpy(arr1, 0) {} kfunca.from_numpy(arr2, 0)".format(op))
                    assert(np.allclose(out, kfunca.to_numpy(out_gpu)) == True)


if __name__ == '__main__':
    test_instance = TestTensorImpl()
    for method_name in dir(test_instance):
        if method_name.startswith("test_"):
            method = getattr(test_instance, method_name)
            print(f"Running {method_name}...")
            method()
