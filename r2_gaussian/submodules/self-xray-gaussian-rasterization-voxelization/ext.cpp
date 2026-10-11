/*
 * Original authors:
 *   Copyright (C) 2023 Inria GRAPHDECO research group (george.drettakis@inria.fr)
 *   Copyright (C) 2024 The Australian National University Prof. Hongdong Li's Lab
 */

#include <torch/extension.h>
#include "rasterize_points.h"
#include "voxelize_points.h"


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("rasterize_gaussians", &RasterizeGaussiansCUDA);
  m.def("rasterize_gaussians_backward", &RasterizeGaussiansBackwardCUDA);
  m.def("voxelize_gaussians", &VoxelizeGaussiansCUDA);
  m.def("voxelize_gaussians_backward", &VoxelizeGaussiansBackwardCUDA);
  // m.def("mark_visible", &markVisible);
}