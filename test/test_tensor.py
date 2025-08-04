import kfunca
import torch
import numpy as np
from common import assert_allclose

print(kfunca.__file__)


class TestTensorImpl(object):
    def test_tensor_impl(self):
        arr = np.random.uniform(-10, 10, size=(2, 3))
        arr_gpu = kfunca.from_numpy(arr, 0)
        assert_allclose(arr, arr_gpu)
    
    def test_tensor_add(self):
        for shape in ((2,3), (1000), (12,11,3331)):
            arr = np.random.uniform(-10, 10, size=shape).astype(np.float32)
            arr_2 = arr + arr
            arr_gpu = kfunca.from_numpy(arr, 0)
            arr_gpu_2 = arr_gpu + arr_gpu
            arr_gpu_2_cpu = arr_gpu_2.numpy()
            assert_allclose(arr_2, arr_gpu_2_cpu)
            arr1 = np.random.uniform(-10, 10, size=shape).astype(np.int32)
            arr2 = np.random.uniform(-10, 10, size=shape).astype(np.float32)
            out = arr1 + arr2
            out_gpu = kfunca.from_numpy(arr1, 0) + kfunca.from_numpy(arr2, 0)
            assert_allclose(out, out_gpu)

    def test_inplace_op(self):
        shape1 = (5,7,11)
        shape2 = (5,1,11)
        arr1 = np.random.uniform(-10, 10, size=shape1).astype(np.float32)
        arr2 = np.random.uniform(-10, 10, size=shape2).astype(np.float32)
        arr1_gpu = kfunca.from_numpy(arr1, 0)
        addr1 = arr1_gpu.data_ptr()
        arr2_gpu = kfunca.from_numpy(arr2, 0)
        arr1 += arr2
        arr1_gpu += arr2_gpu
        assert(addr1 == arr1_gpu.data_ptr())
        assert_allclose(arr1, arr1_gpu)
        arr1 -= arr2
        arr1_gpu -= arr2_gpu
        assert(addr1 == arr1_gpu.data_ptr())
        assert_allclose(arr1, arr1_gpu)
        arr1 *= arr2
        arr1_gpu *= arr2_gpu
        assert(addr1 == arr1_gpu.data_ptr())
        assert_allclose(arr1, arr1_gpu)
        arr1 /= arr2
        arr1_gpu /= arr2_gpu
        assert(addr1 == arr1_gpu.data_ptr())
        assert_allclose(arr1, arr1_gpu)
        arr1 += 2
        arr1_gpu += 2
        assert(addr1 == arr1_gpu.data_ptr())
        assert_allclose(arr1, arr1_gpu)
        arr1 -= 3
        arr1_gpu -= 3
        assert(addr1 == arr1_gpu.data_ptr())
        assert_allclose(arr1, arr1_gpu)
        arr1 *= 4
        arr1_gpu *= 4
        assert(addr1 == arr1_gpu.data_ptr())
        assert_allclose(arr1, arr1_gpu)
        arr1 /= 5
        arr1_gpu /= 5
        assert(addr1 == arr1_gpu.data_ptr())
        assert_allclose(arr1, arr1_gpu)

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
                    assert_allclose(out, out_gpu)
                    arr1 = np.random.uniform(-10, 10, size=shape[0]).astype(np.int32)
                    arr2 = np.random.uniform(-10, 10, size=shape[1]).astype(np.float32)
                    out = eval("arr1 {} arr2".format(op))
                    out_gpu = eval("kfunca.from_numpy(arr1, 0) {} kfunca.from_numpy(arr2, 0)".format(op))
                    assert_allclose(out, out_gpu)
    
    def test_reduce(self):
        for op in ['sum', 'mean']:
            print("op:{}".format(op))
            for dim in [0,1,2]:
                arr = np.random.uniform(-10, 10, size=[223,23,3213]).astype(np.float32)
                arr_sum = eval("np.{}(arr, axis=dim, keepdims=True)".format(op))
                arr_gpu = kfunca.from_numpy(arr, 0)
                arr_gpu_sum = eval("arr_gpu.{}(dim)".format(op))
                assert_allclose(arr_sum, arr_gpu_sum, atol=1e-2, rtol=1e-2)
    
    def test_mean_std(self):
        shape = (13, 325, 127)
        dim = 1
        arr = np.random.uniform(-10, 10, size=shape)
        arr_ = kfunca.from_numpy(arr, 0)
        divisor = shape[dim] - 1
        mean = arr_.mean(dim)
        var = ((arr_ - mean) * (arr_ - mean)).sum(dim)
        var = var / divisor
        mean_var = arr_.mean_var(dim, False)
        assert_allclose(mean, mean_var[0], atol=1e-2, rtol=1e-2)
        assert_allclose(var, mean_var[1], atol=1e-2, rtol=1e-2)
        kfunca.memstat()
    
    def test_norm_stat(self):
        for shape in [[64, 64], [1024, 2048], [4096, 4096], [4096*4+3, 4096*4+3]]:
            dim = 0
            arr = np.random.uniform(-10, 10, size=shape).astype(np.float32)
            arr_ = kfunca.from_numpy(arr, 0)
            divisor = shape[dim]
            mean = np.mean(arr, axis=dim, keepdims=True)
            var = ((arr - mean) * (arr - mean))
            var = np.sum(var, axis=dim, keepdims=True)
            invstd = 1.0 / np.sqrt(var / divisor)
            mean_invstd = arr_.norm_stat(dim)
            assert_allclose(mean, mean_invstd[0])
            assert_allclose(invstd, mean_invstd[1])
    
    def test_convert(self):
        arr = np.random.uniform(-10, 10, size=(2, 3))
        arr_gpu = kfunca.from_numpy(arr, 0)
        arr_gpu_half = arr_gpu.half()
        arr_gpu *= arr_gpu
        arr_gpu_half *= arr_gpu_half
        assert_allclose(arr_gpu, arr_gpu_half.float())
        arr = np.random.uniform(-10, 10, size=(2, 3))
        arr_gpu = kfunca.from_numpy(arr, 0)
        arr_gpu_bf = arr_gpu.bfloat16()
        arr_gpu *= arr_gpu
        arr_gpu_bf *= arr_gpu_bf
        assert_allclose(arr_gpu, arr_gpu_bf.float(), atol=1e-1, rtol=1e-1)
    
    def test_permute(self):
        arr = np.random.uniform(-10, 10, size=(16, 8, 64, 11)) # 0,1,2,3
        arr_p = arr.transpose(2,1,0,3)
        arr_gpu = kfunca.from_numpy(arr, 0)
        arr_gpu_p = arr_gpu.permute(2,1,0,3).contiguous()
        assert_allclose(arr_gpu_p, arr_p)
    
    def test_sort_small_slice(self):
        shapes = [
            [2, 3, 4],
            [23, 11, 23],
            [11, 23, 64],
            [13, 65, 1049],
            [5, 11, 22223],
        ]
        dims = [2, 1, 0]
        descendings = [False, True]
        dtypes = [np.float32, np.double, np.int32]
        for dtype in dtypes:
            print(dtype)
            for descending in descendings:
                for dim in dims:
                    for shape in shapes:
                        # print(shape, dim, descending)
                        arr = np.random.uniform(-1000, 1000, size=shape).astype(dtype)
                        arr_t = torch.from_numpy(arr)
                        res, ind = torch.sort(arr_t, dim=dim, descending=descending, stable=True)
                        arr_gpu = kfunca.from_numpy(arr, 0)
                        res_gpu, ind_gpu = arr_gpu.sort(dim, descending)
                        assert_allclose(res_gpu, res)
                        assert_allclose(ind_gpu, ind)
    
    def test_sort_large_slice(self):
        arr = np.random.uniform(-1000, 1000, size=(4, 1024000)).astype(np.float32)
        res = np.sort(arr, axis=1)
        ind = np.argsort(arr, axis=1, kind='stable')
        arr_gpu = kfunca.from_numpy(arr, 0)
        res_gpu, ind_gpu = arr_gpu.sort(1, False)
        assert_allclose(res_gpu, res)
        assert_allclose(ind_gpu, ind)
    
    def test_topk_small(self):
        shapes = [
            [13, 65, 1049],
            [33, 22, 22223],
        ]
        dims = [2, 1, 0]
        descendings = [False, True]
        dtypes = [np.float32, np.double, np.int32]
        k = 8
        for dtype in dtypes:
            print(dtype)
            for descending in descendings:
                for dim in dims:
                    for shape in shapes:
                        arr = np.random.uniform(-100000, 100000, size=shape).astype(dtype)
                        arr_t = torch.from_numpy(arr)
                        res, ind = torch.topk(arr_t, k, dim=dim, largest=descending)
                        arr_gpu = kfunca.from_numpy(arr, 0)
                        res_gpu, ind_gpu = arr_gpu.topk(k, dim, descending)
                        assert_allclose(res_gpu, res)
    
    def test_topk_large(self):
        for k in [2049, 22223]:
            arr = np.random.uniform(-10000, 10000, size=(4, 1024000)).astype(np.float32)
            arr_t = torch.from_numpy(arr)
            res, ind = torch.topk(arr_t, k, dim=1, largest=True)
            arr_gpu = kfunca.from_numpy(arr, 0)
            res_gpu, ind_gpu = arr_gpu.topk(k, 1, True)
            assert_allclose(res_gpu, res)
    
    def test_tensor_slice(self):
        arr = np.random.uniform(-10000, 10000, size=(11, 155, 33, 5)).astype(np.float32)
        arr_t = torch.from_numpy(arr)
        arr_gpu = kfunca.from_numpy(arr, 0)
        arr_t_ = arr_t[3, 3:8, 4:11:2]
        arr_gpu_ = arr_gpu[3, 3:8, 4:11:2]
        assert_allclose(arr_t_, arr_gpu_.contiguous())
    
    def test_view(self):
        arr = np.random.uniform(-10000, 10000, size=(5,2,11,23)).astype(np.float32)
        arr_t = torch.from_numpy(arr)
        arr_gpu = kfunca.from_numpy(arr, 0)
        arr_t = arr_t.view(5,-1,23).contiguous() + 1
        arr_gpu = arr_gpu.view(5,-1,23).contiguous() + 1
        assert_allclose(arr_t, arr_gpu)
    
    def test_cat(self):
        arr1 = np.random.uniform(-10000, 10000, size=(5,11,23)).astype(np.float32)
        arr2 = np.random.uniform(-10000, 10000, size=(5,13,23)).astype(np.float32)
        arr3 = np.random.uniform(-10000, 10000, size=(5,1,23)).astype(np.float32)
        arr1_t = torch.from_numpy(arr1)
        arr2_t = torch.from_numpy(arr2)
        arr3_t = torch.from_numpy(arr3)
        arr1_gpu = kfunca.from_numpy(arr1, 0)
        arr2_gpu = kfunca.from_numpy(arr2, 0)
        arr3_gpu = kfunca.from_numpy(arr3, 0)
        arr_t = torch.cat([arr1_t, arr2_t, arr3_t], 1)
        arr_gpu = kfunca.cat([arr1_gpu, arr2_gpu, arr3_gpu], 1)
        assert_allclose(arr_t, arr_gpu)
    
    def test_split(self):
        arr = np.random.uniform(-10000, 10000, size=(5,25,23)).astype(np.float32)
        arr_t = torch.from_numpy(arr)
        arr_gpu = kfunca.from_numpy(arr, 0)
        arr_t1, arr_t2, arr_t3 = arr_t.split([11,13,1], 1)
        arr_gpu1, arr_gpu2, arr_gpu3 = arr_gpu.split([11,13,1], 1)
        assert_allclose(arr_t1, arr_gpu1)
        assert_allclose(arr_t2, arr_gpu2)
        assert_allclose(arr_t3, arr_gpu3)
    
    def test_index_put(self):
        arr = np.random.uniform(-10000, 10000, size=(13, 15)).astype(np.float32)
        arr_gpu = kfunca.from_numpy(arr, 0)
        indices = [kfunca.from_numpy(np.array([0, 5, 1, 2]).astype('q'), 0), 
                   kfunca.from_numpy(np.array([0, 11, 1, 0]).astype('q'), 0)]
        values = kfunca.from_numpy(np.random.uniform(-10000, 10000, size=(4)).astype(np.float32), 0)
        arr_gpu.index_put_(indices, values)
        arr_gpu_pt = torch.from_numpy(arr)
        indices_t = [torch.from_numpy(indices[0].numpy()), torch.from_numpy(indices[1].numpy())]
        values_pt = torch.from_numpy(values.numpy())
        arr_gpu_pt.index_put_(indices_t, values_pt)
        assert_allclose(arr_gpu, arr_gpu_pt)


if __name__ == '__main__':
    test_instance = TestTensorImpl()
    for method_name in dir(test_instance):
        if method_name.startswith("test_"):
            method = getattr(test_instance, method_name)
            print(f"Running {method_name}...")
            method()
