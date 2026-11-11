/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */

#include <torch/extension.h>
#include "rasterize_points.h"
#include "voxelize_points.h"

#define R2G_STRINGIZE_IMPL(value) #value
#define R2G_STRINGIZE(value) R2G_STRINGIZE_IMPL(value)

#ifndef R2G_BUILD_SOURCE_HASH
#error "R2G_BUILD_SOURCE_HASH must be provided by setup.py"
#endif


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.attr("build_source_hash") = R2G_STRINGIZE(R2G_BUILD_SOURCE_HASH);
  m.def("rasterize_gaussians", &RasterizeGaussiansCUDA);
  m.def("rasterize_gaussians_backward", &RasterizeGaussiansBackwardCUDA);
  m.def("voxelize_gaussians", &VoxelizeGaussiansCUDA);
  m.def("voxelize_gaussians_backward", &VoxelizeGaussiansBackwardCUDA);
  m.def("mark_visible", &markVisible);
}
