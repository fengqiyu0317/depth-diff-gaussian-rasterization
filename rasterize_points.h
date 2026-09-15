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

#pragma once
#include <torch/extension.h>
#include <cstdint>
#include <cstdio>
#include <tuple>
#include <string>
#include <vector>
	
std::tuple<int, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
RasterizeGaussiansCUDA(
	const torch::Tensor& background,
	const torch::Tensor& means3D,
    const torch::Tensor& colors,
    const torch::Tensor& opacity,
	const torch::Tensor& scales,
	const torch::Tensor& rotations,
	const float scale_modifier,
	const torch::Tensor& cov3D_precomp,
	const torch::Tensor& viewmatrix,
	const torch::Tensor& projmatrix,
	const float tan_fovx, 
	const float tan_fovy,
    const int image_height,
    const int image_width,
	const torch::Tensor& sh,
	const int degree,
	const torch::Tensor& campos,
	const bool prefiltered,
	const bool debug);

// Inference-only Raster+deformation-head mixed path.  The first seven return
// values have exactly the same order and meaning as RasterizeGaussiansCUDA;
// the eighth is the FP32 [N, 128] head output.
std::tuple<int, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
RasterizeGaussiansWithHeadCUDA(
	const torch::Tensor& background,
	const torch::Tensor& means3D,
    const torch::Tensor& colors,
    const torch::Tensor& opacity,
	const torch::Tensor& scales,
	const torch::Tensor& rotations,
	const float scale_modifier,
	const torch::Tensor& cov3D_precomp,
	const torch::Tensor& viewmatrix,
	const torch::Tensor& projmatrix,
	const float tan_fovx,
	const float tan_fovy,
    const int image_height,
    const int image_width,
	const torch::Tensor& sh,
	const int degree,
	const torch::Tensor& campos,
	const bool prefiltered,
	const bool debug,
	const torch::Tensor& head_input,
	const torch::Tensor& head_weight,
	const torch::Tensor& head_bias,
	const int64_t persistent_blocks);

// Mixed ABI v2.  Raster return values retain their legacy order; the eighth
// value is a task-ordered vector of distinct FP32 [N_i, 128] outputs.
std::tuple<int, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
	torch::Tensor, torch::Tensor, std::vector<torch::Tensor>>
RasterizeGaussiansWithHeadsCUDA(
	const torch::Tensor& background,
	const torch::Tensor& means3D,
	const torch::Tensor& colors,
	const torch::Tensor& opacity,
	const torch::Tensor& scales,
	const torch::Tensor& rotations,
	const float scale_modifier,
	const torch::Tensor& cov3D_precomp,
	const torch::Tensor& viewmatrix,
	const torch::Tensor& projmatrix,
	const float tan_fovx,
	const float tan_fovy,
	const int image_height,
	const int image_width,
	const torch::Tensor& sh,
	const int degree,
	const torch::Tensor& campos,
	const bool prefiltered,
	const bool debug,
	const std::vector<torch::Tensor>& head_inputs,
	const std::vector<torch::Tensor>& head_weights,
	const std::vector<torch::Tensor>& head_biases,
	const int64_t worker_groups,
	const int64_t persistent_blocks);

// C3 packed shared-input first-linear backend.  The eighth value is one
// contiguous FP32 [H, N, 128] tensor.
std::tuple<int, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
	torch::Tensor, torch::Tensor, torch::Tensor>
RasterizeGaussiansWithPackedHeadsCUDA(
	const torch::Tensor& background,
	const torch::Tensor& means3D,
	const torch::Tensor& colors,
	const torch::Tensor& opacity,
	const torch::Tensor& scales,
	const torch::Tensor& rotations,
	const float scale_modifier,
	const torch::Tensor& cov3D_precomp,
	const torch::Tensor& viewmatrix,
	const torch::Tensor& projmatrix,
	const float tan_fovx,
	const float tan_fovy,
	const int image_height,
	const int image_width,
	const torch::Tensor& sh,
	const int degree,
	const torch::Tensor& campos,
	const bool prefiltered,
	const bool debug,
	const torch::Tensor& head_input,
	const torch::Tensor& head_weights,
	const torch::Tensor& head_biases,
	const int64_t worker_groups,
	const int64_t persistent_blocks);

// C4 whole-head backend.  The plural form supports 1--5 tasks and returns
// task-ordered FP32 [N_i, O_i] outputs; the singular form is a strict
// convenience binding for the one-task physical configuration.
std::tuple<int, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
	torch::Tensor, torch::Tensor, std::vector<torch::Tensor>>
RasterizeGaussiansWithWholeHeadsCUDA(
	const torch::Tensor& background,
	const torch::Tensor& means3D,
	const torch::Tensor& colors,
	const torch::Tensor& opacity,
	const torch::Tensor& scales,
	const torch::Tensor& rotations,
	const float scale_modifier,
	const torch::Tensor& cov3D_precomp,
	const torch::Tensor& viewmatrix,
	const torch::Tensor& projmatrix,
	const float tan_fovx,
	const float tan_fovy,
	const int image_height,
	const int image_width,
	const torch::Tensor& sh,
	const int degree,
	const torch::Tensor& campos,
	const bool prefiltered,
	const bool debug,
	const std::vector<torch::Tensor>& head_inputs,
	const std::vector<torch::Tensor>& first_weights,
	const std::vector<torch::Tensor>& first_biases,
	const std::vector<torch::Tensor>& tail_weights,
	const std::vector<torch::Tensor>& tail_biases,
	const int64_t worker_groups,
	const int64_t persistent_blocks);

std::tuple<int, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
	torch::Tensor, torch::Tensor, torch::Tensor>
RasterizeGaussiansWithWholeHeadCUDA(
	const torch::Tensor& background,
	const torch::Tensor& means3D,
	const torch::Tensor& colors,
	const torch::Tensor& opacity,
	const torch::Tensor& scales,
	const torch::Tensor& rotations,
	const float scale_modifier,
	const torch::Tensor& cov3D_precomp,
	const torch::Tensor& viewmatrix,
	const torch::Tensor& projmatrix,
	const float tan_fovx,
	const float tan_fovy,
	const int image_height,
	const int image_width,
	const torch::Tensor& sh,
	const int degree,
	const torch::Tensor& campos,
	const bool prefiltered,
	const bool debug,
	const torch::Tensor& head_input,
	const torch::Tensor& first_weight,
	const torch::Tensor& first_bias,
	const torch::Tensor& tail_weight,
	const torch::Tensor& tail_bias,
	const int64_t persistent_blocks);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
 RasterizeGaussiansBackwardCUDA(
 	const torch::Tensor& background,
	const torch::Tensor& means3D,
	const torch::Tensor& radii,
    const torch::Tensor& colors,
	const torch::Tensor& scales,
	const torch::Tensor& rotations,
	const float scale_modifier,
	const torch::Tensor& cov3D_precomp,
	const torch::Tensor& viewmatrix,
    const torch::Tensor& projmatrix,
	const float tan_fovx, 
	const float tan_fovy,
    const torch::Tensor& dL_dout_color,
	const torch::Tensor& dL_dout_depth,
	const torch::Tensor& sh,
	const int degree,
	const torch::Tensor& campos,
	const torch::Tensor& geomBuffer,
	const int R,
	const torch::Tensor& binningBuffer,
	const torch::Tensor& imageBuffer,
	const bool debug);
		
torch::Tensor markVisible(
		torch::Tensor& means3D,
		torch::Tensor& viewmatrix,
		torch::Tensor& projmatrix);
