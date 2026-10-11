/*
 * Original authors:
 *   Copyright (C) 2023 Inria GRAPHDECO research group (george.drettakis@inria.fr)
 *   Copyright (C) 2024 The Australian National University Prof. Hongdong Li's Lab
 *
 * Additional modifications:
 *   Copyright (C) 2026 KAIST Visual Computing Lab
 *
 * Contact:
 *   Kiseok Choi <cks0605@gmail.com>
 */

#pragma once
#include <torch/extension.h>
#include <cstdio>
#include <tuple>
#include <string>

std::tuple<int, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
RasterizeGaussiansCUDA(
	const torch::Tensor& means3D,
    const torch::Tensor& opacity,
	const torch::Tensor& scales,
	const torch::Tensor& rotations,
	const float scale_modifier,
	const torch::Tensor& cov3D_precomp,
	const torch::Tensor& viewmatrix,
	const torch::Tensor& projmatrix,
	const torch::Tensor& tan_fovx, 
	const torch::Tensor& tan_fovy,
    const int image_height,
    const int image_width,
	const torch::Tensor& campos,
	const bool prefiltered,
	const int mode,
	const bool debug);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, 
	torch::Tensor, torch::Tensor, torch::Tensor,
	torch::Tensor, torch::Tensor, torch::Tensor,
	torch::Tensor, torch::Tensor>
RasterizeGaussiansBackwardCUDA(
	const torch::Tensor& means3D,
	const torch::Tensor& radii,
	const torch::Tensor& scales,
	const torch::Tensor& rotations,
	const float scale_modifier,
	const torch::Tensor& cov3D_precomp,
	const torch::Tensor& viewmatrix,
    const torch::Tensor& projmatrix,
	const torch::Tensor& tan_fovx, 
	const torch::Tensor& tan_fovy,
    const torch::Tensor& dL_dout_color,
	const torch::Tensor& campos,
	const torch::Tensor& geomBuffer,
	const int R,
	const torch::Tensor& binningBuffer,
	const torch::Tensor& imageBuffer,
	const int mode,
	const bool debug);
		


