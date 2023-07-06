#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include "squash.h"
#include <torch/nn/functional.h>


template <typename scalar_t>
__global__ void squash0_cuda_forward_kernel(scalar_t* __restrict__ input) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    

    input[index] = torch.nn.functional.softsign(input[index])*torch.nn.functional.softsign(input[index]);

}

std::vector<torch::Tensor> squash0_cuda_forward(torch::Tensor input) {
    const int threads = 1024;
    const int blocks = (input.size(0) + threads - 1) / threads;
    
    AT_DISPATCH_FLOATING_TYPES(input.type(), "squash0_forward_cuda", ([&] {
        squash0_cuda_forward_kernel<scalar_t><<<blocks, threads>>>(input.data<scalar_t>());
    }));

    return {input};
}


