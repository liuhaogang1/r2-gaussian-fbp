from pathlib import Path

from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension


ROOT = Path(__file__).resolve().parent
# Reuse the existing vendored GLM headers; the original rasterizer remains
# untouched and continues to provide the default backend.
GLM = ROOT.parent / "xray-gaussian-rasterization-voxelization" / "third_party" / "glm"

setup(
    name="self_xray_gaussian_rasterization_voxelization",
    packages=["self_xray_gaussian_rasterization_voxelization"],
    ext_modules=[
        CUDAExtension(
            name="self_xray_gaussian_rasterization_voxelization._C",
            sources=[
                "cuda_rasterizer/rasterizer_impl.cu",
                "cuda_rasterizer/forward.cu",
                "cuda_rasterizer/backward.cu",
                "rasterize_points.cu",
                "cuda_voxelizer/voxelizer_impl.cu",
                "cuda_voxelizer/forward.cu",
                "cuda_voxelizer/backward.cu",
                "voxelize_points.cu",
                "ext.cpp",
            ],
            extra_compile_args={
                "nvcc": [
                    "--allow-unsupported-compiler",
                    "-D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH",
                    "-D_USE_MATH_DEFINES",
                    "-I" + str(GLM),
                ]
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
