import kfunca
import numpy as np

print(kfunca.__file__)


class TestGemm(object):
    def test_gemm_base(self):
        a = np.random.uniform(-10, 10, size=(123, 457))
        b = np.random.uniform(-10, 10, size=(457, 234))
        a_gpu = kfunca.from_numpy(a, 0)
        b_gpu = kfunca.from_numpy(b, 0)
        print(a_gpu.sizes(), b_gpu.sizes())
        out_gpu = kfunca.gemm(a_gpu, b_gpu, 1.0, 0.0)
        out = np.matmul(a, b)
        assert(np.allclose(out, out_gpu.numpy()) == True)


if __name__ == '__main__':
    test_instance = TestGemm()
    for method_name in dir(test_instance):
        if method_name.startswith("test_"):
            method = getattr(test_instance, method_name)
            print(f"Running {method_name}...")
            method()
