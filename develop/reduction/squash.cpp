#include <torch/extension.h>
#include <vector>

#include "squash.h"

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

std::vector<torch::Tensor> squash0_forward(torch::Tensor input) {
    CHECK_INPUT(input);
    return squash0_cuda_forward(input);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){
    m.def("forward", &squash0_forward, "Squash forward (CUDA)");
}