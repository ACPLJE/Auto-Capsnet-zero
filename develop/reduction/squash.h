#include <torch/extension.h>
#include <vector>

std::vector<torch::Tensor> squash0_cuda_forward(torch::Tensor input);