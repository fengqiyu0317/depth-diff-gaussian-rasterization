#include "tacker_mixed.h"

#include <algorithm>
#include <climits>
#include <cstdint>
#include <stdexcept>

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "config.h"
#include "head_linear_device.cuh"
#include "tacker_forward.cuh"

static_assert(
	CudaRasterizer::Tacker::kRasterThreads == BLOCK_X * BLOCK_Y,
	"mixed Raster subgroup must match the exact 16x16 raster tile");
static_assert(
	CudaRasterizer::Tacker::kHeadThreads == tacker_4dgs::kHeadThreads,
	"mixed head subgroup must match head_linear_gptb_device");
static_assert(
	CudaRasterizer::Tacker::kRasterThreads + CudaRasterizer::Tacker::kHeadThreads ==
		CudaRasterizer::Tacker::kMixedThreads,
	"mixed subgroup ranges must exactly cover the physical CTA");

extern "C" __global__ void __launch_bounds__(CudaRasterizer::Tacker::kMixedThreads)
tacker_mix_render_head_v1(
	const uint2* ranges,
	const uint32_t* point_list,
	int width,
	int height,
	const float2* points_xy_image,
	const float* features,
	const float* depths,
	const float4* conic_opacity,
	float* final_T,
	uint32_t* n_contrib,
	const float* background,
	float* out_color,
	float* out_depth,
	int raster_grid_x,
	int raster_grid_y,
	const half* head_input,
	const half* head_weight,
	const float* head_bias,
	float* head_output,
	int head_rows,
	int head_grid_x,
	int head_grid_y,
	int physical_blocks)
{
	// Raster owns [0, 256).  Its adapter uses named barrier 1 with an explicit
	// participant count of 256, so the head warps never participate.
	CudaRasterizer::Tacker::general_ptb_render<NUM_CHANNELS>(
		ranges,
		point_list,
		width,
		height,
		points_xy_image,
		features,
		depths,
		conic_opacity,
		final_T,
		n_contrib,
		background,
		out_color,
		out_depth,
		raster_grid_x,
		raster_grid_y,
		0,
		physical_blocks,
		raster_grid_x * raster_grid_y,
		0,
		CudaRasterizer::Tacker::kRasterBarrierId);

	// Head owns [256, 384).  The device adapter synchronizes only within each
	// of its four warps and therefore cannot deadlock with the Raster subgroup.
	tacker_4dgs::head_linear_gptb_device(
		head_input,
		head_weight,
		head_bias,
		head_output,
		head_rows,
		head_grid_x,
		head_grid_y,
		0,
		physical_blocks,
		head_grid_x * head_grid_y,
		CudaRasterizer::Tacker::kHeadThreadBase);
}

namespace
{

int activeSmCount()
{
	int device = 0;
	cudaError_t status = cudaGetDevice(&device);
	if (status != cudaSuccess)
		throw std::runtime_error(cudaGetErrorString(status));
	static thread_local int cached_device = -1;
	static thread_local int cached_sm_count = 0;
	if (cached_device == device && cached_sm_count > 0)
		return cached_sm_count;

	cudaDeviceProp properties;
	status = cudaGetDeviceProperties(&properties, device);
	if (status != cudaSuccess)
		throw std::runtime_error(cudaGetErrorString(status));
	cached_device = device;
	cached_sm_count = properties.multiProcessorCount;
	return cached_sm_count;
}

}  // namespace

void CudaRasterizer::Tacker::launchMixedRenderHead(
	const dim3 raster_grid,
	const uint2* ranges,
	const uint32_t* point_list,
	int width,
	int height,
	const float2* points_xy_image,
	const float* features,
	const float* depths,
	const float4* conic_opacity,
	float* final_T,
	uint32_t* n_contrib,
	const float* background,
	float* out_color,
	float* out_depth,
	const MixedHeadTask& head,
	cudaStream_t stream)
{
	if (head.rows < 0)
		throw std::invalid_argument("mixed head rows must be >= 0");
	if (head.persistent_blocks < 0)
		throw std::invalid_argument("persistent_blocks must be >= 0");
	if (head.rows > 0 &&
		(head.input == nullptr || head.weight == nullptr || head.bias == nullptr ||
		 head.output == nullptr))
		throw std::invalid_argument("mixed head pointers must be non-null when rows > 0");

	const int64_t raster_blocks_64 =
		static_cast<int64_t>(raster_grid.x) * static_cast<int64_t>(raster_grid.y);
	if (raster_blocks_64 > INT_MAX)
		throw std::overflow_error("raster logical grid exceeds int32 mixed ABI");

	const int head_grid_x = static_cast<int>(
		(static_cast<int64_t>(head.rows) + 15) / 16);
	const int head_grid_y = head.rows == 0 ? 0 : 2;
	const int64_t head_blocks_64 =
		static_cast<int64_t>(head_grid_x) * static_cast<int64_t>(head_grid_y);
	if (head_blocks_64 > INT_MAX)
		throw std::overflow_error("head logical grid exceeds int32 mixed ABI");

	const int logical_blocks = std::max(
		static_cast<int>(raster_blocks_64), static_cast<int>(head_blocks_64));
	if (logical_blocks == 0)
		return;

	int physical_blocks = head.persistent_blocks;
	if (physical_blocks == 0)
		physical_blocks = activeSmCount();
	physical_blocks = std::max(1, std::min(physical_blocks, logical_blocks));

	tacker_mix_render_head_v1<<<
		physical_blocks, kMixedThreads, 0, stream>>>(
		ranges,
		point_list,
		width,
		height,
		points_xy_image,
		features,
		depths,
		conic_opacity,
		final_T,
		n_contrib,
		background,
		out_color,
		out_depth,
		static_cast<int>(raster_grid.x),
		static_cast<int>(raster_grid.y),
		reinterpret_cast<const half*>(head.input),
		reinterpret_cast<const half*>(head.weight),
		head.bias,
		head.output,
		head.rows,
		head_grid_x,
		head_grid_y,
		physical_blocks);

	// Detect configuration/resource errors without synchronizing either task.
	const cudaError_t launch_status = cudaPeekAtLastError();
	if (launch_status != cudaSuccess)
		throw std::runtime_error(cudaGetErrorString(launch_status));
}
