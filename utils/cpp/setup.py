from setuptools import setup, find_packages
from setuptools import Extension
from pybind11.setup_helpers import Pybind11Extension, build_ext
import os

# 定义你的 C++ 扩展
ext_modules = [
    Pybind11Extension(
        "inverse_projection_cpp",
        ["inverse_projection.cpp"],
    ),
]

setup(
    name="inverse_projection_cpp",
    version="0.1",
    author="Your Name",
    description="A Python wrapper for C++ inverse projection",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    packages=find_packages(),  # 自动找到所有 Python 包
    package_data={
        "cpp": ["*.so"],  # 包含 .so 文件
    },
    zip_safe=False,
)