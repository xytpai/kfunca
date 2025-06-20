import kfunca
import numpy as np
import torch
import torch.nn.functional as F

print(kfunca.__file__)


class TestNN(object):
    def test_causal_attention(self):
        batch_size = 4
        nheads = 16
        q_seq_length = 64
        kv_seq_length = 1024
        hidden_size = 128
        q_ = np.random.uniform(-10, 10, size=(batch_size, nheads, q_seq_length, hidden_size)).astype(np.float32)
        k_ = np.random.uniform(-10, 10, size=(batch_size, nheads, kv_seq_length, hidden_size)).astype(np.float32)
        v_ = np.random.uniform(-10, 10, size=(batch_size, nheads, kv_seq_length, hidden_size)).astype(np.float32)
        q = kfunca.from_numpy(q_, 0)
        k = kfunca.from_numpy(k_, 0)
        v = kfunca.from_numpy(v_, 0)
        out = kfunca.causal_attention(q, k, v).numpy()
        q_ref = torch.from_numpy(q_)
        k_ref = torch.from_numpy(k_)
        v_ref = torch.from_numpy(v_)
        out_ref = F.scaled_dot_product_attention(q_ref, k_ref, v_ref, is_causal=True).numpy()
        assert(np.allclose(out, out_ref, rtol=1e-3, atol=1e-3) == True)


if __name__ == '__main__':
    test_instance = TestNN()
    for method_name in dir(test_instance):
        if method_name.startswith("test_"):
            method = getattr(test_instance, method_name)
            print(f"Running {method_name}...")
            method()
