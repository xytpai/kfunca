**kfunca** is a minimalist high-performance GPU AI kernel library.
The operator scope is focused solely on multimodal transformers.
Here are the supported features:

#### 1. Basic infrastructure

- [x] GPU Launcher
- [x] Caching Allocator
- [x] Tensor Implementation
- [x] Tensor Iterator

---

#### 2. GPU Operator

> Basic operator:

- [x] from_numpy/to_numpy
- [x] add/sub/mul/div
- [x] permute/contiguous/copy
- [x] sum/mean
- [x] sort/topk
- [x] slice/view
- [ ] concat/split/stack

> Neural network operator:

- [ ] rms_norm
- [x] causal attention
- [ ] embedding
- [x] matmul
- [ ] qkv_linear

---

#### 3. Floating Point Support

- [x] fp32/64
- [x] float16
- [ ] bfloat16
- [ ] float8

---

> Welcome to reach out for collaboration: xytpai@gmail.com
