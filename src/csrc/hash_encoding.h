#ifndef HASH_ENCODE_H
#define HASH_ENCODE_H

#include <stdint>
#include <torch/torch.h>

void hash_encode_forward(
  const at::Tensor inputs,
  at::Tensor outputs,
);
/*
// This operation is non-differentiable, is it necessary to provide a grad?
void hash_encode_backward(
  const at::Tensor grad,
);
*/

#endif
