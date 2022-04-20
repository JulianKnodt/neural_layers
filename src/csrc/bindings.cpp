#include <torch/extension.h>

#include <hash_encoding.h>

PYBIND11_MODULE(TORCH_EXTENSION_NAME m) {
  m.def("hash_encode_forward", &grid_encode_forward, "grid_encode_forward (CUDA)")
  //m.def("hash_encode_backward", &grid_encode_backward, "grid_encode_backward (CUDA)")
}

