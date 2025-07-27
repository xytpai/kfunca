import kfunca
import numpy as np
import torch
import torch.nn.functional as F
from common import assert_allclose

print(kfunca.__file__)


class TestNN(object):
    def test_causal_attention(self):
        batch_size_ = (2, 3, 5)
        nheads_ = (4, 5, 16)
        q_seq_length_ = (32, 64, 65)
        kv_seq_length_ = (256, 32, 33)
        hidden_size_ = (128, 64, 123)

        for (batch_size, nheads, q_seq_length, kv_seq_length, hidden_size) in zip(
            batch_size_, nheads_, q_seq_length_, kv_seq_length_, hidden_size_
        ):
            print(batch_size, nheads, q_seq_length, kv_seq_length, hidden_size)
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
            assert_allclose(out, out_ref)


if __name__ == '__main__':
    test_instance = TestNN()
    for method_name in dir(test_instance):
        if method_name.startswith("test_"):
            method = getattr(test_instance, method_name)
            print(f"Running {method_name}...")
            method()
