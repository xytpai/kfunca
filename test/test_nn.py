import kfunca
import numpy as np

print(kfunca.__file__)


class TestNN(object):
    def test_causal_attention(self):
        batch_size = 4
        nheads = 16
        q_seq_length = 1024
        kv_seq_length = 1024
        hidden_size = 64
        q_ = np.random.uniform(-10, 10, size=(batch_size, nheads, q_seq_length, hidden_size))
        k_ = np.random.uniform(-10, 10, size=(batch_size, nheads, kv_seq_length, hidden_size))
        v_ = np.random.uniform(-10, 10, size=(batch_size, nheads, kv_seq_length, hidden_size))
        q = kfunca.from_numpy(q_, 0)
        k = kfunca.from_numpy(k_, 0)
        v = kfunca.from_numpy(v_, 0)
        out = kfunca.causal_attention(q, k, v)
        print(out.sizes())


if __name__ == '__main__':
    test_instance = TestNN()
    for method_name in dir(test_instance):
        if method_name.startswith("test_"):
            method = getattr(test_instance, method_name)
            print(f"Running {method_name}...")
            method()
