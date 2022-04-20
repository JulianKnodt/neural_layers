#include <cuda.h>
#include <cuda_runtime.h>

#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>
#include <torch/torch.h>

#include <algorithm>
#include <stdint.h>
#include <cstdio>

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be a contiguous tensor")
#define CHECK_IS_INT(x) TORCH_CHECK(x.scalar_type() == at::ScalarType::Int, #x " must be an int tensor")
#define CHECK_IS_FLOATING(x) TORCH_CHECK(x.scalar_type() == at::ScalarType::Float || x.scalar_type() == at::ScalarType::Half || x.scalar_type() == at::ScalarType::Double, #x " must be a floating tensor")

template <typename scalar_t, uint32_t D> __global__ void fast_hash_forward_cuda(
  const torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> vals,
  torch::PackedTensorAccessor32<scalar_t,1,torch::RestrictPtrTraits> output
) {
    static_assert(D <= 7, "fast_hash can only hash up to 7 dimensions.");

    const auto idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx > vals.size(0)) return;

    // While 1 is technically not a good prime for hashing (or a prime at all), it helps memory coherence
    // and is sufficient for our use case of obtaining a uniformly colliding index from high-dimensional
    // coordinates.
    constexpr uint32_t primes[7] = { 1, 2654435761, 805459861, 3674653429, 2097192037, 1434869437, 2165219737 };

    scalar_t result = 0;

    #pragma unroll
    for (uint32_t i = 0; i < D; ++i) {
      result ^= vals[idx][i] * primes[i];
    }

    output[idx] = result;
}


void hash_encode_forward(
  const at::Tensor inputs,
  at::Tensor outputs
) {
  CHECK_CUDA(inputs);
  CHECK_IS_INT(inputs);
  CHECK_IS_INT(outputs);

  const int threads = 1024;
  const dim3 blocks((inputs.size(0)+threads-1)/threads);

  AT_DISPATCH_INTEGRAL_TYPES(inputs.type(), "fast_hash_forward_cuda", ([&] {
    fast_hash_forward_cuda<scalar_t, 3><<<blocks, threads>>>(
      inputs.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
      outputs.packed_accessor32<scalar_t,1,torch::RestrictPtrTraits>()
    );
  }));
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("hash_encode_forward", &hash_encode_forward, "hash_encode_forward (CUDA)");
  //m.def("hash_encode_backward", &grid_encode_backward, "grid_encode_backward (CUDA)")
}

