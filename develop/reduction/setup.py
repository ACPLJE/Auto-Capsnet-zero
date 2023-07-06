from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='squash0',
    ext_modules=[
        CUDAExtension('squash0_cuda', [
            'squash.cpp',
            'squash_cuda_kernel.cu',
        ])
    ],
    cmdclass={
        'build_ext': BuildExtension
    })