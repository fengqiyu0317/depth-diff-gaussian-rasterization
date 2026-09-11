#pragma once

#include <cstddef>
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

// Mixed ABI v2 keeps the exact Raster subgroup and adds 1--5 independent
// 128-thread worker groups.  The worker groups execute 1--5 first-linear head
// tasks using tacker_ext's stable multi-task device adapter.
static const int kMixedMultiAbiVersion = 2;
static const int kMaxHeadTasksV2 = 5;
static const int kMinWorkerGroupsV2 = 1;
static const int kMaxWorkerGroupsV2 = 5;
static const int kWorkerGroupThreadsV2 = 128;
static const int kHeadDescriptorBarrierIdV2 = 2;
static const int kMaxMixedThreadsV2 =
	kRasterThreads + kMaxWorkerGroupsV2 * kWorkerGroupThreadsV2;

struct MixedKernelResources
{
	int abi_version;
	int worker_groups;
	int physical_threads;
	int device_ordinal;
	int compute_capability_major;
	int compute_capability_minor;
	int multiprocessor_count;
	int device_max_threads_per_block;
	int device_max_threads_per_multiprocessor;
	int warp_size;
	int kernel_max_threads_per_block;
	int registers_per_thread;
	std::size_t static_shared_bytes;
	std::size_t local_bytes_per_thread;
	std::size_t max_dynamic_shared_bytes;
	int active_blocks_per_multiprocessor;
	int active_warps_per_multiprocessor;
	int max_warps_per_multiprocessor;
	double occupancy;
	bool launch_supported;
};

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

// Launch mixed ABI v2.  task_count is in [1, 5] and worker_groups is in
// [1, task_count].  For group g, the tacker_ext adapter evaluates tasks
// g, g + worker_groups, ... exactly once.  Empty-row tasks are legal.
void launchMixedRenderHeads(
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
	const MixedHeadBundleV2& heads,
	cudaStream_t stream);

// Runtime resource/capability query used by offline candidate filtering.
// ABI v1 requires worker_groups==1; ABI v2 accepts [1, 5].  This call queries
// the active device and does not launch or synchronize a kernel.
MixedKernelResources queryMixedKernelResources(
	int abi_version,
	int worker_groups);

}  // namespace Tacker
}  // namespace CudaRasterizer
