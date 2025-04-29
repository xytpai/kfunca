# `Core` Components

The directory contains device agnostic code, mainly including the following:

- **Device Allocator**: Storage management component for efficient device memory allocation, ensuring necessary reusability.
- **Tensor Implemention**:  Core data structure like PyTorch::Tensor.
- **Tensor Iterator**: A helper or abstraction that lets you easily loop over (iterate through) elements of one or more tensors, handling all the complicated stuff for you — like broadcasting, type promotion, and memory strides.
