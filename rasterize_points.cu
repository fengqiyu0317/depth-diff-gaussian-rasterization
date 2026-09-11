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

#include <math.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/core/grad_mode.h>
#include <c10/cuda/CUDAGuard.h>
#include <climits>
#include <cstdint>
#include <cstdio>
#include <sstream>
#include <iostream>
#include <tuple>
#include <stdio.h>
#include <cuda_runtime_api.h>
#include <memory>
#include <vector>
#include "cuda_rasterizer/config.h"
#include "cuda_rasterizer/rasterizer.h"
#include "cuda_rasterizer/tacker_mixed.h"
#include <fstream>
#include <string>
#include <functional>

std::function<char*(size_t N)> resizeFunctional(torch::Tensor& t) {
    auto lambda = [&t](size_t N) {
        t.resize_({(long long)N});
		return reinterpret_cast<char*>(t.contiguous().data_ptr());
    };
    return lambda;
}

namespace
{

constexpr uintptr_t kWmmaAlignment = 32;

bool isPlaceholder(const torch::Tensor& tensor)
{
	return tensor.dim() == 1 && tensor.size(0) == 0;
}

bool isNativeWmmaAligned(const torch::Tensor& tensor)
{
	return reinterpret_cast<uintptr_t>(tensor.data_ptr()) % kWmmaAlignment == 0;
}

void checkInt32RowCount(const char* name, const torch::Tensor& tensor)
{
	TORCH_CHECK(
		tensor.size(0) <= static_cast<int64_t>(INT_MAX),
		name,
		" row count exceeds the int32 Raster ABI");
}

void checkRasterDimensions(int image_height, int image_width)
{
	TORCH_CHECK(image_height > 0, "image_height must be > 0");
	TORCH_CHECK(image_width > 0, "image_width must be > 0");
	const int64_t pixels = static_cast<int64_t>(image_height) * image_width;
	TORCH_CHECK(
		pixels <= static_cast<int64_t>(INT_MAX) / NUM_CHANNELS,
		"image dimensions exceed the int32 Raster color-indexing ABI");
}

int legacyShCoefficientCount(const torch::Tensor& sh)
{
	if (sh.size(0) == 0)
		return 0;
	TORCH_CHECK(sh.dim() >= 2, "non-empty sh tensor must have a coefficient axis");
	TORCH_CHECK(
		sh.size(1) <= static_cast<int64_t>(INT_MAX),
		"sh coefficient count exceeds the int32 Raster ABI");
	return static_cast<int>(sh.size(1));
}

void checkCudaFloatOnRasterDevice(
	const char* name,
	const torch::Tensor& tensor,
	const torch::Tensor& means3D)
{
	TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
	TORCH_CHECK(tensor.scalar_type() == at::kFloat, name, " must be float32");
	TORCH_CHECK(
		tensor.device() == means3D.device(),
		name,
		" must be on the same CUDA device as means3D");
}

struct MixedRasterChoices
{
	bool use_sh;
	bool use_cov3d;
};

MixedRasterChoices checkMixedRasterArguments(
	const torch::Tensor& background,
	const torch::Tensor& means3D,
	const torch::Tensor& colors,
	const torch::Tensor& opacity,
	const torch::Tensor& scales,
	const torch::Tensor& rotations,
	const torch::Tensor& cov3D_precomp,
	const torch::Tensor& viewmatrix,
	const torch::Tensor& projmatrix,
	const torch::Tensor& sh,
	const torch::Tensor& campos,
	int degree)
{
	TORCH_CHECK(means3D.is_cuda(), "means3D must be a CUDA tensor");
	TORCH_CHECK(means3D.scalar_type() == at::kFloat, "means3D must be float32");
	TORCH_CHECK(
		means3D.dim() == 2 && means3D.size(1) == 3,
		"means3D must have shape [P, 3]");

	checkCudaFloatOnRasterDevice("background", background, means3D);
	checkCudaFloatOnRasterDevice("colors", colors, means3D);
	checkCudaFloatOnRasterDevice("opacity", opacity, means3D);
	checkCudaFloatOnRasterDevice("scales", scales, means3D);
	checkCudaFloatOnRasterDevice("rotations", rotations, means3D);
	checkCudaFloatOnRasterDevice("cov3D_precomp", cov3D_precomp, means3D);
	checkCudaFloatOnRasterDevice("viewmatrix", viewmatrix, means3D);
	checkCudaFloatOnRasterDevice("projmatrix", projmatrix, means3D);
	checkCudaFloatOnRasterDevice("sh", sh, means3D);
	checkCudaFloatOnRasterDevice("campos", campos, means3D);

	const int64_t rows = means3D.size(0);
	TORCH_CHECK(
		background.dim() == 1 && background.size(0) == NUM_CHANNELS,
		"background must have shape [3]");
	TORCH_CHECK(
		opacity.dim() == 2 && opacity.size(0) == rows && opacity.size(1) == 1,
		"opacity must have shape [P, 1]");
	TORCH_CHECK(
		viewmatrix.dim() == 2 && viewmatrix.size(0) == 4 &&
			viewmatrix.size(1) == 4,
		"viewmatrix must have shape [4, 4]");
	TORCH_CHECK(
		projmatrix.dim() == 2 && projmatrix.size(0) == 4 &&
			projmatrix.size(1) == 4,
		"projmatrix must have shape [4, 4]");
	TORCH_CHECK(
		campos.dim() == 1 && campos.size(0) == 3,
		"campos must have shape [3]");

	// Rank/shape, not numel(), identifies an omitted alternative. This is
	// important at P==0: [0, M, 3] is a real SH tensor while [0] is a
	// placeholder, even though both have zero elements.
	const bool has_sh = sh.dim() == 3 && sh.size(0) == rows &&
		sh.size(1) > 0 && sh.size(2) == 3;
	const bool has_colors = colors.dim() == 2 && colors.size(0) == rows &&
		colors.size(1) == NUM_CHANNELS;
	TORCH_CHECK(degree >= 0 && degree <= 3, "degree must be in [0, 3]");
	if (has_sh)
	{
		const int64_t required_coefficients =
			static_cast<int64_t>(degree + 1) * (degree + 1);
		TORCH_CHECK(
			sh.size(1) >= required_coefficients,
			"sh coefficient count is too small for degree");
		TORCH_CHECK(
			sh.size(1) <= static_cast<int64_t>(INT_MAX),
			"sh coefficient count exceeds the int32 Raster ABI");
	}
	TORCH_CHECK(
		(has_sh && isPlaceholder(colors)) ||
			(has_colors && isPlaceholder(sh)),
		"provide exactly one of sh [P, M, 3] or colors [P, 3]; "
		"the omitted tensor must be a [0] placeholder");

	const bool has_scales = scales.dim() == 2 && scales.size(0) == rows &&
		scales.size(1) == 3;
	const bool has_rotations = rotations.dim() == 2 &&
		rotations.size(0) == rows && rotations.size(1) == 4;
	const bool has_cov3d = cov3D_precomp.dim() == 2 &&
		cov3D_precomp.size(0) == rows && cov3D_precomp.size(1) == 6;
	TORCH_CHECK(
		(has_cov3d && isPlaceholder(scales) && isPlaceholder(rotations)) ||
			(has_scales && has_rotations && isPlaceholder(cov3D_precomp)),
		"provide exactly one of cov3D_precomp [P, 6] or the pair "
		"scales [P, 3] and rotations [P, 4]; omitted tensors must be "
		"[0] placeholders");

	return MixedRasterChoices{has_sh, has_cov3d};
}

void checkMixedHeadArguments(
	const torch::Tensor& means3D,
	const torch::Tensor& input,
	const torch::Tensor& weight,
	const torch::Tensor& bias,
	int64_t persistent_blocks)
{
	TORCH_CHECK(
		!at::GradMode::is_enabled(),
		"rasterize_gaussians_with_head is inference-only; call it under torch.no_grad()");
	TORCH_CHECK(means3D.is_cuda(), "means3D must be a CUDA tensor");
	TORCH_CHECK(input.is_cuda(), "head_input must be a CUDA tensor");
	TORCH_CHECK(weight.is_cuda(), "head_weight must be a CUDA tensor");
	TORCH_CHECK(bias.is_cuda(), "head_bias must be a CUDA tensor");
	TORCH_CHECK(input.is_contiguous(), "head_input must be contiguous");
	TORCH_CHECK(weight.is_contiguous(), "head_weight must be contiguous");
	TORCH_CHECK(bias.is_contiguous(), "head_bias must be contiguous");
	TORCH_CHECK(input.scalar_type() == at::kHalf, "head_input must be float16");
	TORCH_CHECK(weight.scalar_type() == at::kHalf, "head_weight must be float16");
	TORCH_CHECK(bias.scalar_type() == at::kFloat, "head_bias must be float32");
	TORCH_CHECK(
		input.dim() == 2 && input.size(1) == 128,
		"head_input must have shape [N, 128]");
	TORCH_CHECK(
		weight.dim() == 2 && weight.size(0) == 128 && weight.size(1) == 128,
		"head_weight must have shape [128, 128]");
	TORCH_CHECK(
		bias.dim() == 1 && bias.size(0) == 128,
		"head_bias must have shape [128]");
	TORCH_CHECK(
		means3D.device() == input.device() && input.device() == weight.device() &&
			input.device() == bias.device(),
		"raster and head tensors must be on the same CUDA device");
	TORCH_CHECK(
		means3D.size(0) <= static_cast<int64_t>(INT_MAX),
		"Gaussian row count exceeds the int32 raster ABI");
	TORCH_CHECK(
		input.size(0) <= static_cast<int64_t>(INT_MAX),
		"head_input row count exceeds the int32 mixed ABI");
	TORCH_CHECK(
		isNativeWmmaAligned(input),
		"head_input data pointer must be natively 32-byte aligned for WMMA; "
		"misaligned contiguous views are not supported");
	TORCH_CHECK(
		isNativeWmmaAligned(weight),
		"head_weight data pointer must be natively 32-byte aligned for WMMA; "
		"misaligned contiguous views are not supported");
	TORCH_CHECK(persistent_blocks >= 0, "persistent_blocks must be >= 0");
	TORCH_CHECK(
		persistent_blocks <= static_cast<int64_t>(INT_MAX),
		"persistent_blocks exceeds the int32 mixed ABI");
}

void checkMixedHeadsArguments(
	const torch::Tensor& means3D,
	const std::vector<torch::Tensor>& inputs,
	const std::vector<torch::Tensor>& weights,
	const std::vector<torch::Tensor>& biases,
	int64_t worker_groups,
	int64_t persistent_blocks)
{
	TORCH_CHECK(
		!at::GradMode::is_enabled(),
		"rasterize_gaussians_with_heads is inference-only; "
		"call it under torch.no_grad()");
	TORCH_CHECK(
		inputs.size() == weights.size() && inputs.size() == biases.size(),
		"head_inputs, head_weights, and head_biases must have equal lengths");
	TORCH_CHECK(
		!inputs.empty() &&
			inputs.size() <=
				static_cast<std::size_t>(CudaRasterizer::Tacker::kMaxHeadTasksV2),
		"mixed ABI v2 requires between 1 and 5 head tasks");
	TORCH_CHECK(
		worker_groups >= CudaRasterizer::Tacker::kMinWorkerGroupsV2 &&
			worker_groups <= static_cast<int64_t>(inputs.size()),
		"worker_groups must be in [1, task_count]");
	TORCH_CHECK(persistent_blocks >= 0, "persistent_blocks must be >= 0");
	TORCH_CHECK(
		persistent_blocks <= static_cast<int64_t>(INT_MAX),
		"persistent_blocks exceeds the int32 mixed ABI v2");
	TORCH_CHECK(
		means3D.size(0) <= static_cast<int64_t>(INT_MAX),
		"Gaussian row count exceeds the int32 raster ABI");

	for (std::size_t task_index = 0; task_index < inputs.size(); ++task_index)
	{
		const torch::Tensor& input = inputs[task_index];
		const torch::Tensor& weight = weights[task_index];
		const torch::Tensor& bias = biases[task_index];
		TORCH_CHECK(
			input.is_cuda(), "head_inputs[", task_index, "] must be a CUDA tensor");
		TORCH_CHECK(
			weight.is_cuda(), "head_weights[", task_index, "] must be a CUDA tensor");
		TORCH_CHECK(
			bias.is_cuda(), "head_biases[", task_index, "] must be a CUDA tensor");
		TORCH_CHECK(
			input.is_contiguous(), "head_inputs[", task_index, "] must be contiguous");
		TORCH_CHECK(
			weight.is_contiguous(), "head_weights[", task_index, "] must be contiguous");
		TORCH_CHECK(
			bias.is_contiguous(), "head_biases[", task_index, "] must be contiguous");
		TORCH_CHECK(
			input.scalar_type() == at::kHalf,
			"head_inputs[", task_index, "] must be float16");
		TORCH_CHECK(
			weight.scalar_type() == at::kHalf,
			"head_weights[", task_index, "] must be float16");
		TORCH_CHECK(
			bias.scalar_type() == at::kFloat,
			"head_biases[", task_index, "] must be float32");
		TORCH_CHECK(
			input.dim() == 2 && input.size(1) == 128,
			"head_inputs[", task_index, "] must have shape [N, 128]");
		TORCH_CHECK(
			weight.dim() == 2 && weight.size(0) == 128 &&
				weight.size(1) == 128,
			"head_weights[", task_index, "] must have shape [128, 128]");
		TORCH_CHECK(
			bias.dim() == 1 && bias.size(0) == 128,
			"head_biases[", task_index, "] must have shape [128]");
		TORCH_CHECK(
			means3D.device() == input.device() &&
				input.device() == weight.device() &&
				input.device() == bias.device(),
			"raster and head task ", task_index,
			" tensors must be on the same CUDA device");
		TORCH_CHECK(
			input.size(0) <= static_cast<int64_t>(INT_MAX),
			"head_inputs[", task_index,
			"] row count exceeds the int32 mixed ABI v2");
		TORCH_CHECK(
			isNativeWmmaAligned(input),
			"head_inputs[", task_index,
			"] data pointer must be natively 32-byte aligned for WMMA; "
			"misaligned contiguous views are not supported");
		TORCH_CHECK(
			isNativeWmmaAligned(weight),
			"head_weights[", task_index,
			"] data pointer must be natively 32-byte aligned for WMMA; "
			"misaligned contiguous views are not supported");
	}
}

}  // namespace

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
	const bool debug)
{
  if (means3D.ndimension() != 2 || means3D.size(1) != 3) {
    AT_ERROR("means3D must have dimensions (num_points, 3)");
  }

	TORCH_CHECK(means3D.is_cuda(), "means3D must be a CUDA tensor");
	const c10::cuda::CUDAGuard device_guard(means3D.device());
	checkInt32RowCount("means3D", means3D);
	checkRasterDimensions(image_height, image_width);
  
	const int P = static_cast<int>(means3D.size(0));
  const int H = image_height;
  const int W = image_width;

  auto int_opts = means3D.options().dtype(torch::kInt32);
  auto float_opts = means3D.options().dtype(torch::kFloat32);

  torch::Tensor out_color = torch::full({NUM_CHANNELS, H, W}, 0.0, float_opts);
  torch::Tensor out_depth = torch::full({1, H, W}, 0.0, float_opts);
  torch::Tensor radii = torch::full({P}, 0, means3D.options().dtype(torch::kInt32));
  
  torch::Device device(torch::kCUDA);
  torch::TensorOptions options(torch::kByte);
  torch::Tensor geomBuffer = torch::empty({0}, options.device(device));
  torch::Tensor binningBuffer = torch::empty({0}, options.device(device));
  torch::Tensor imgBuffer = torch::empty({0}, options.device(device));
  std::function<char*(size_t)> geomFunc = resizeFunctional(geomBuffer);
  std::function<char*(size_t)> binningFunc = resizeFunctional(binningBuffer);
  std::function<char*(size_t)> imgFunc = resizeFunctional(imgBuffer);
  
	int rendered = 0;
	if(P != 0)
	{
	  const int M = legacyShCoefficientCount(sh);
	  const cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

	  rendered = CudaRasterizer::Rasterizer::forward(
	    geomFunc,
		binningFunc,
		imgFunc,
	    P, degree, M,
		background.contiguous().data<float>(),
		W, H,
		means3D.contiguous().data<float>(),
		sh.contiguous().data_ptr<float>(),
		colors.contiguous().data<float>(), 
		opacity.contiguous().data<float>(), 
		scales.contiguous().data_ptr<float>(),
		scale_modifier,
		rotations.contiguous().data_ptr<float>(),
		cov3D_precomp.contiguous().data<float>(), 
		viewmatrix.contiguous().data<float>(), 
		projmatrix.contiguous().data<float>(),
		campos.contiguous().data<float>(),
		tan_fovx,
		tan_fovy,
		prefiltered,
		out_color.contiguous().data<float>(),
		out_depth.contiguous().data<float>(),
		radii.contiguous().data<int>(),
			debug,
			stream);
  }
  return std::make_tuple(rendered, out_color, out_depth, radii, geomBuffer, binningBuffer, imgBuffer);
}

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
	const int64_t persistent_blocks)
{
	const MixedRasterChoices raster_choices = checkMixedRasterArguments(
		background,
		means3D,
		colors,
		opacity,
		scales,
		rotations,
		cov3D_precomp,
		viewmatrix,
		projmatrix,
		sh,
		campos,
		degree);
	checkMixedHeadArguments(
		means3D, head_input, head_weight, head_bias, persistent_blocks);
	checkRasterDimensions(image_height, image_width);

	const c10::cuda::CUDAGuard device_guard(means3D.device());
	const int P = static_cast<int>(means3D.size(0));
	const int H = image_height;
	const int W = image_width;
	const int head_rows = static_cast<int>(head_input.size(0));

	auto float_opts = means3D.options().dtype(torch::kFloat32);
	torch::Tensor out_color = torch::full({NUM_CHANNELS, H, W}, 0.0, float_opts);
	torch::Tensor out_depth = torch::full({1, H, W}, 0.0, float_opts);
	torch::Tensor radii = torch::full(
		{P}, 0, means3D.options().dtype(torch::kInt32));
	torch::Tensor head_output = torch::empty(
		{head_input.size(0), 128}, head_input.options().dtype(torch::kFloat32));
	TORCH_CHECK(
		isNativeWmmaAligned(head_output),
		"head_output data pointer must be natively 32-byte aligned for WMMA");

	const torch::TensorOptions byte_options =
		means3D.options().dtype(torch::kByte);
	torch::Tensor geomBuffer = torch::empty({0}, byte_options);
	torch::Tensor binningBuffer = torch::empty({0}, byte_options);
	torch::Tensor imgBuffer = torch::empty({0}, byte_options);
	std::function<char*(size_t)> geomFunc = resizeFunctional(geomBuffer);
	std::function<char*(size_t)> binningFunc = resizeFunctional(binningBuffer);
	std::function<char*(size_t)> imgFunc = resizeFunctional(imgBuffer);

	CudaRasterizer::MixedHeadTask head_task = {
		reinterpret_cast<const void*>(head_input.data_ptr<at::Half>()),
		reinterpret_cast<const void*>(head_weight.data_ptr<at::Half>()),
		head_bias.data_ptr<float>(),
		head_output.data_ptr<float>(),
		head_rows,
		static_cast<int>(persistent_blocks)};

	const cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
	const torch::Tensor background_contiguous = background.contiguous();
	int rendered = 0;
	if (P != 0)
	{
		const int M = raster_choices.use_sh ? static_cast<int>(sh.size(1)) : 0;

		// Keep any materialized contiguous views alive through launch and pass
		// literal null pointers for omitted alternatives.  The rasterizer uses
		// pointer nullness to choose SH/colors and covariance/scale-rotation.
		const torch::Tensor means3D_contiguous = means3D.contiguous();
		const torch::Tensor sh_contiguous = sh.contiguous();
		const torch::Tensor colors_contiguous = colors.contiguous();
		const torch::Tensor opacity_contiguous = opacity.contiguous();
		const torch::Tensor scales_contiguous = scales.contiguous();
		const torch::Tensor rotations_contiguous = rotations.contiguous();
		const torch::Tensor cov3D_contiguous = cov3D_precomp.contiguous();
		const torch::Tensor viewmatrix_contiguous = viewmatrix.contiguous();
		const torch::Tensor projmatrix_contiguous = projmatrix.contiguous();
		const torch::Tensor campos_contiguous = campos.contiguous();
		const float* sh_ptr = raster_choices.use_sh
			? sh_contiguous.data_ptr<float>()
			: nullptr;
		const float* colors_ptr = raster_choices.use_sh
			? nullptr
			: colors_contiguous.data_ptr<float>();
		const float* scales_ptr = raster_choices.use_cov3d
			? nullptr
			: scales_contiguous.data_ptr<float>();
		const float* rotations_ptr = raster_choices.use_cov3d
			? nullptr
			: rotations_contiguous.data_ptr<float>();
		const float* cov3D_ptr = raster_choices.use_cov3d
			? cov3D_contiguous.data_ptr<float>()
			: nullptr;

		rendered = CudaRasterizer::Rasterizer::forward(
			geomFunc,
			binningFunc,
			imgFunc,
			P, degree, M,
			background_contiguous.data_ptr<float>(),
			W, H,
			means3D_contiguous.data_ptr<float>(),
			sh_ptr,
			colors_ptr,
			opacity_contiguous.data_ptr<float>(),
			scales_ptr,
			scale_modifier,
			rotations_ptr,
			cov3D_ptr,
			viewmatrix_contiguous.data_ptr<float>(),
			projmatrix_contiguous.data_ptr<float>(),
			campos_contiguous.data_ptr<float>(),
			tan_fovx,
			tan_fovy,
			prefiltered,
			out_color.contiguous().data<float>(),
			out_depth.contiguous().data<float>(),
			radii.contiguous().data<int>(),
			debug,
			stream,
			&head_task);
	}
	else
	{
		// Preserve the legacy P==0 raster outputs/buffers while still running
		// the independent head task exactly once when N>0.
		CudaRasterizer::Tacker::launchMixedRenderHead(
			dim3(0, 0, 1),
			nullptr,
			nullptr,
			W, H,
			nullptr,
			nullptr,
			nullptr,
			nullptr,
			nullptr,
			nullptr,
			background_contiguous.data_ptr<float>(),
			out_color.data_ptr<float>(),
			out_depth.data_ptr<float>(),
			head_task,
			stream);
	}

	if (P != 0 || head_rows != 0)
	{
		const cudaError_t status = cudaGetLastError();
		TORCH_CHECK(
			status == cudaSuccess,
			"rasterize_gaussians_with_head launch failed: ",
			cudaGetErrorString(status));
	}

	return std::make_tuple(
		rendered,
		out_color,
		out_depth,
		radii,
		geomBuffer,
		binningBuffer,
		imgBuffer,
		head_output);
}

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
	const int64_t persistent_blocks)
{
	const MixedRasterChoices raster_choices = checkMixedRasterArguments(
		background,
		means3D,
		colors,
		opacity,
		scales,
		rotations,
		cov3D_precomp,
		viewmatrix,
		projmatrix,
		sh,
		campos,
		degree);
	checkMixedHeadsArguments(
		means3D,
		head_inputs,
		head_weights,
		head_biases,
		worker_groups,
		persistent_blocks);
	checkRasterDimensions(image_height, image_width);

	const c10::cuda::CUDAGuard device_guard(means3D.device());
	const int P = static_cast<int>(means3D.size(0));
	const int H = image_height;
	const int W = image_width;

	auto float_opts = means3D.options().dtype(torch::kFloat32);
	torch::Tensor out_color = torch::full({NUM_CHANNELS, H, W}, 0.0, float_opts);
	torch::Tensor out_depth = torch::full({1, H, W}, 0.0, float_opts);
	torch::Tensor radii = torch::full(
		{P}, 0, means3D.options().dtype(torch::kInt32));
	std::vector<torch::Tensor> head_outputs;
	head_outputs.reserve(head_inputs.size());
	for (std::size_t task_index = 0; task_index < head_inputs.size(); ++task_index)
	{
		head_outputs.push_back(torch::empty(
			{head_inputs[task_index].size(0), 128},
			head_inputs[task_index].options().dtype(torch::kFloat32)));
		TORCH_CHECK(
			isNativeWmmaAligned(head_outputs.back()),
			"head_outputs[", task_index,
			"] data pointer must be natively 32-byte aligned for WMMA");
		if (head_outputs.back().numel() != 0)
		{
			for (std::size_t other = 0; other < task_index; ++other)
			{
				TORCH_CHECK(
					head_outputs[other].numel() == 0 ||
						head_outputs[other].data_ptr() !=
							head_outputs.back().data_ptr(),
					"mixed ABI v2 head outputs must not alias each other");
			}
		}
	}

	const torch::TensorOptions byte_options =
		means3D.options().dtype(torch::kByte);
	torch::Tensor geomBuffer = torch::empty({0}, byte_options);
	torch::Tensor binningBuffer = torch::empty({0}, byte_options);
	torch::Tensor imgBuffer = torch::empty({0}, byte_options);
	std::function<char*(size_t)> geomFunc = resizeFunctional(geomBuffer);
	std::function<char*(size_t)> binningFunc = resizeFunctional(binningBuffer);
	std::function<char*(size_t)> imgFunc = resizeFunctional(imgBuffer);

	CudaRasterizer::MixedHeadBundleV2 head_bundle = {};
	head_bundle.task_count = static_cast<int>(head_inputs.size());
	head_bundle.worker_groups = static_cast<int>(worker_groups);
	head_bundle.persistent_blocks = static_cast<int>(persistent_blocks);
	bool any_head_rows = false;
	for (std::size_t task_index = 0; task_index < head_inputs.size(); ++task_index)
	{
		const int rows = static_cast<int>(head_inputs[task_index].size(0));
		head_bundle.tasks[task_index] = {
			reinterpret_cast<const void*>(
				head_inputs[task_index].data_ptr<at::Half>()),
			reinterpret_cast<const void*>(
				head_weights[task_index].data_ptr<at::Half>()),
			head_biases[task_index].data_ptr<float>(),
			head_outputs[task_index].data_ptr<float>(),
			rows};
		any_head_rows = any_head_rows || rows != 0;
	}

	const cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
	const torch::Tensor background_contiguous = background.contiguous();
	int rendered = 0;
	if (P != 0)
	{
		const int M = raster_choices.use_sh ? static_cast<int>(sh.size(1)) : 0;

		// Materialized Raster views and the three task vectors stay alive until
		// after the single v2 mixed kernel has been enqueued on this stream.
		const torch::Tensor means3D_contiguous = means3D.contiguous();
		const torch::Tensor sh_contiguous = sh.contiguous();
		const torch::Tensor colors_contiguous = colors.contiguous();
		const torch::Tensor opacity_contiguous = opacity.contiguous();
		const torch::Tensor scales_contiguous = scales.contiguous();
		const torch::Tensor rotations_contiguous = rotations.contiguous();
		const torch::Tensor cov3D_contiguous = cov3D_precomp.contiguous();
		const torch::Tensor viewmatrix_contiguous = viewmatrix.contiguous();
		const torch::Tensor projmatrix_contiguous = projmatrix.contiguous();
		const torch::Tensor campos_contiguous = campos.contiguous();
		const float* sh_ptr = raster_choices.use_sh
			? sh_contiguous.data_ptr<float>()
			: nullptr;
		const float* colors_ptr = raster_choices.use_sh
			? nullptr
			: colors_contiguous.data_ptr<float>();
		const float* scales_ptr = raster_choices.use_cov3d
			? nullptr
			: scales_contiguous.data_ptr<float>();
		const float* rotations_ptr = raster_choices.use_cov3d
			? nullptr
			: rotations_contiguous.data_ptr<float>();
		const float* cov3D_ptr = raster_choices.use_cov3d
			? cov3D_contiguous.data_ptr<float>()
			: nullptr;

		rendered = CudaRasterizer::Rasterizer::forward(
			geomFunc,
			binningFunc,
			imgFunc,
			P, degree, M,
			background_contiguous.data_ptr<float>(),
			W, H,
			means3D_contiguous.data_ptr<float>(),
			sh_ptr,
			colors_ptr,
			opacity_contiguous.data_ptr<float>(),
			scales_ptr,
			scale_modifier,
			rotations_ptr,
			cov3D_ptr,
			viewmatrix_contiguous.data_ptr<float>(),
			projmatrix_contiguous.data_ptr<float>(),
			campos_contiguous.data_ptr<float>(),
			tan_fovx,
			tan_fovy,
			prefiltered,
			out_color.contiguous().data<float>(),
			out_depth.contiguous().data<float>(),
			radii.contiguous().data<int>(),
			debug,
			stream,
			nullptr,
			&head_bundle);
	}
	else
	{
		// Empty Raster input does not suppress independent head work.  Conversely,
		// all-zero-row heads plus an empty Raster result in no physical launch.
		CudaRasterizer::Tacker::launchMixedRenderHeads(
			dim3(0, 0, 1),
			nullptr,
			nullptr,
			W, H,
			nullptr,
			nullptr,
			nullptr,
			nullptr,
			nullptr,
			nullptr,
			background_contiguous.data_ptr<float>(),
			out_color.data_ptr<float>(),
			out_depth.data_ptr<float>(),
			head_bundle,
			stream);
	}

	if (P != 0 || any_head_rows)
	{
		const cudaError_t status = cudaGetLastError();
		TORCH_CHECK(
			status == cudaSuccess,
			"rasterize_gaussians_with_heads launch failed: ",
			cudaGetErrorString(status));
	}

	return std::make_tuple(
		rendered,
		out_color,
		out_depth,
		radii,
		geomBuffer,
		binningBuffer,
		imgBuffer,
		head_outputs);
}

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
	const bool debug) 
{
	TORCH_CHECK(means3D.is_cuda(), "means3D must be a CUDA tensor");
	const c10::cuda::CUDAGuard device_guard(means3D.device());
	const cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
	TORCH_CHECK(
		dL_dout_color.dim() == 3,
		"dL_dout_color must have shape [channels, H, W]");
	TORCH_CHECK(
		dL_dout_color.size(1) <= static_cast<int64_t>(INT_MAX) &&
			dL_dout_color.size(2) <= static_cast<int64_t>(INT_MAX),
		"gradient image dimensions exceed the int32 Raster ABI");
	checkInt32RowCount("means3D", means3D);
	const int P = static_cast<int>(means3D.size(0));
	const int H = static_cast<int>(dL_dout_color.size(1));
	const int W = static_cast<int>(dL_dout_color.size(2));
	checkRasterDimensions(H, W);
  
	const int M = legacyShCoefficientCount(sh);

  torch::Tensor dL_dmeans3D = torch::zeros({P, 3}, means3D.options());
  torch::Tensor dL_dmeans2D = torch::zeros({P, 3}, means3D.options());
  torch::Tensor dL_dcolors = torch::zeros({P, NUM_CHANNELS}, means3D.options());
  torch::Tensor dL_ddepths = torch::zeros({P, 1}, means3D.options());
  torch::Tensor dL_dconic = torch::zeros({P, 2, 2}, means3D.options());
  torch::Tensor dL_dopacity = torch::zeros({P, 1}, means3D.options());
  torch::Tensor dL_dcov3D = torch::zeros({P, 6}, means3D.options());
  torch::Tensor dL_dsh = torch::zeros({P, M, 3}, means3D.options());
  torch::Tensor dL_dscales = torch::zeros({P, 3}, means3D.options());
  torch::Tensor dL_drotations = torch::zeros({P, 4}, means3D.options());
  
  if(P != 0)
  {  
	  CudaRasterizer::Rasterizer::backward(P, degree, M, R,
	  background.contiguous().data<float>(),
	  W, H, 
	  means3D.contiguous().data<float>(),
	  sh.contiguous().data<float>(),
	  colors.contiguous().data<float>(),
	  scales.data_ptr<float>(),
	  scale_modifier,
	  rotations.data_ptr<float>(),
	  cov3D_precomp.contiguous().data<float>(),
	  viewmatrix.contiguous().data<float>(),
	  projmatrix.contiguous().data<float>(),
	  campos.contiguous().data<float>(),
	  tan_fovx,
	  tan_fovy,
	  radii.contiguous().data<int>(),
	  reinterpret_cast<char*>(geomBuffer.contiguous().data_ptr()),
	  reinterpret_cast<char*>(binningBuffer.contiguous().data_ptr()),
	  reinterpret_cast<char*>(imageBuffer.contiguous().data_ptr()),
	  dL_dout_color.contiguous().data<float>(),
	  dL_dout_depth.contiguous().data<float>(),
	  dL_dmeans2D.contiguous().data<float>(),
	  dL_dconic.contiguous().data<float>(),  
	  dL_dopacity.contiguous().data<float>(),
	  dL_dcolors.contiguous().data<float>(),
	  dL_ddepths.contiguous().data<float>(),
	  dL_dmeans3D.contiguous().data<float>(),
	  dL_dcov3D.contiguous().data<float>(),
	  dL_dsh.contiguous().data<float>(),
	  dL_dscales.contiguous().data<float>(),
	  dL_drotations.contiguous().data<float>(),
	  debug,
	  stream);
  }

  return std::make_tuple(dL_dmeans2D, dL_dcolors, dL_dopacity, dL_dmeans3D, dL_dcov3D, dL_dsh, dL_dscales, dL_drotations);
}

torch::Tensor markVisible(
		torch::Tensor& means3D,
		torch::Tensor& viewmatrix,
		torch::Tensor& projmatrix)
{ 
	TORCH_CHECK(means3D.is_cuda(), "means3D must be a CUDA tensor");
	const c10::cuda::CUDAGuard device_guard(means3D.device());
	checkInt32RowCount("means3D", means3D);
	const int P = static_cast<int>(means3D.size(0));
  
  torch::Tensor present = torch::full({P}, false, means3D.options().dtype(at::kBool));
 
	if(P != 0)
	{
		const cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
		CudaRasterizer::Rasterizer::markVisible(P,
		means3D.contiguous().data<float>(),
		viewmatrix.contiguous().data<float>(),
		projmatrix.contiguous().data<float>(),
			present.contiguous().data<bool>(),
			stream);
  }
  
  return present;
}
