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


#ifndef CUDA_RASTERIZER_H_INCLUDED
#define CUDA_RASTERIZER_H_INCLUDED

#include <vector>
#include <functional>

namespace CudaRasterizer
{
    class Rasterizer
	{
	public:
	
		static int forward(
			std::function<char* (size_t)> geometryBuffer,
			std::function<char* (size_t)> binningBuffer,
			std::function<char* (size_t)> imageBuffer,
			const int P,
			const int width, int height,
			const float* means3D,
			const float* opacities,
			const float* scales,
			const float scale_modifier,
			const float* rotations,
			const float* cov3D_precomp,
			const float* viewmatrix,
			const float* projmatrix,
			const float* cam_pos,
			const float* tan_fovx, 
			const float* tan_fovy,
			const bool prefiltered,
			const int mode,
			float* out_color,
			int* radii = nullptr,
			bool debug = false);

		static void backward(
			const int P, int R,
			const int width, int height,
			const float* means3D,
			const float* scales,
			const float scale_modifier,
			const float* rotations,
			const float* cov3D_precomp,
			const float* viewmatrix,
			const float* projmatrix,
			const float* campos,
			const float* tan_fovx,
			const float* tan_fovy,
			const int* radii,
			char* geom_buffer,
			char* binning_buffer,
			char* image_buffer,
			const float* dL_dpix,
			float* dL_dmean2D,
			float* dL_dconic,
			float* dL_dopacity,
			float* dL_dmu,
			float* dL_dmean3D,
			float* dL_dcov3D,
			float* dL_dscale,
			float* dL_drot,
			float* dL_dtanfovx,	// kschoi
			float* dL_dtanfovy, 	// kschoi
			float* dL_dviewmatrix, 	// kschoi
			float* dL_dprojmatrix, 	// kschoi
			float* dL_dcampos,		// kschoi					
			const int mode,
			bool debug);
	};

};

#endif