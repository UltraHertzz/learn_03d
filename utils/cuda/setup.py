from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os

setup(
    name='inverse_projection_cuda',
    ext_modules=[
        CUDAExtension(
            'inverse_projection_cuda', 
            [os.path.join(os.path.dirname(__file__), 'inverse_projection_kernel.cu')],
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    },
    install_requires=[
        'torch>=1.12.1',  # 添加这一行
    ],
)