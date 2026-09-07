#pragma once

#include <cstdint>
#include <cuda_runtime_api.h>

#include "rasterizer.h"

namespace CudaRasterizer
{
namespace Tacker
{

// Stable source-level ABI for the first physical Raster+deformation-head
// mixed leaf.  A physical CTA is split into two independent contiguous
// subgroups; neither subgroup is allowed to execute a CTA-wide barrier.
static const int kMixedAbiVersion = 1;
static const int kMixedThreads = 384;
static const int kRasterThreads = 256;
static const int kHeadThreads = 128;
static const int kHeadThreadBase = 256;
static const int kRasterBarrierId = 1;

// Launch a real 384-thread mixed kernel.  raster_grid may be empty, in which
// case the Raster subgroup performs no memory access and the head subgroup is
// still executed.  persistent_blocks==0 is resolved from the active device's
// multiProcessorCount; no GPU model-specific SM count is embedded here.
void launchMixedRenderHead(
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
	cudaStream_t stream);

}  // namespace Tacker
}  // namespace CudaRasterizer
