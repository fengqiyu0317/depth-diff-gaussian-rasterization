#include "tacker_mixed.h"

#include <algorithm>
#include <cstddef>
#include <climits>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <type_traits>

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "config.h"
#include "head_linear_device.cuh"
#include "head_linear_v2_device.cuh"
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
static_assert(
	CudaRasterizer::Tacker::kMaxHeadTasksV2 == tacker_4dgs::kMaxHeadTasksV2,
	"Raster mixed ABI v2 and tacker_ext must agree on maximum task count");
static_assert(
	CudaRasterizer::Tacker::kWorkerGroupThreadsV2 == tacker_4dgs::kHeadThreads,
	"mixed ABI v2 worker group must match tacker_ext head threads");
static_assert(
	CudaRasterizer::Tacker::kMaxMixedThreadsV2 <= 1024,
	"mixed ABI v2 exceeds the CUDA architectural CTA limit");
static_assert(
	CudaRasterizer::Tacker::kRasterBarrierId !=
		CudaRasterizer::Tacker::kHeadDescriptorBarrierIdV2,
	"Raster and v2 head descriptor barriers must be disjoint");
static_assert(
	CudaRasterizer::Tacker::kRasterBarrierId <
		CudaRasterizer::Tacker::kWholeHeadBarrierBaseIdV2 ||
	CudaRasterizer::Tacker::kRasterBarrierId >=
		CudaRasterizer::Tacker::kWholeHeadBarrierBaseIdV2 +
			CudaRasterizer::Tacker::kMaxWorkerGroupsV2,
	"Raster and whole-head worker barriers must be disjoint");
static_assert(
	CudaRasterizer::Tacker::kWholeHeadDescriptorBarrierIdV2 !=
		CudaRasterizer::Tacker::kRasterBarrierId &&
	CudaRasterizer::Tacker::kWholeHeadDescriptorBarrierIdV2 >=
		CudaRasterizer::Tacker::kWholeHeadBarrierBaseIdV2 +
			CudaRasterizer::Tacker::kMaxWorkerGroupsV2 &&
	CudaRasterizer::Tacker::kWholeHeadDescriptorBarrierIdV2 < 16,
	"whole-head descriptor barrier must be disjoint and architecturally valid");
static_assert(
	CudaRasterizer::Tacker::kMaxTailFeaturesV2 ==
		tacker_4dgs::kMaxTailFeaturesV2,
	"Raster and tacker_ext whole-head tail limits must agree");

// Keep the core Rasterizer independent of CUDA half headers while proving
// that its opaque host descriptor may be copied into the tacker_ext device
// descriptor without changing the source ABI.
static_assert(
	std::is_standard_layout<CudaRasterizer::MixedHeadTaskV2>::value &&
		std::is_standard_layout<tacker_4dgs::HeadLinearTaskV2>::value,
	"mixed v2 task descriptors must be standard-layout PODs");
static_assert(
	sizeof(CudaRasterizer::MixedHeadTaskV2) ==
		sizeof(tacker_4dgs::HeadLinearTaskV2),
	"mixed v2 task descriptor size mismatch");
static_assert(
	alignof(CudaRasterizer::MixedHeadTaskV2) ==
		alignof(tacker_4dgs::HeadLinearTaskV2),
	"mixed v2 task descriptor alignment mismatch");
#define TACKER_ASSERT_TASK_OFFSET(field) \
	static_assert(offsetof(CudaRasterizer::MixedHeadTaskV2, field) == \
		offsetof(tacker_4dgs::HeadLinearTaskV2, field), \
		"mixed v2 task descriptor field offset mismatch: " #field)
TACKER_ASSERT_TASK_OFFSET(input);
TACKER_ASSERT_TASK_OFFSET(weight);
TACKER_ASSERT_TASK_OFFSET(bias);
TACKER_ASSERT_TASK_OFFSET(output);
TACKER_ASSERT_TASK_OFFSET(rows);
#undef TACKER_ASSERT_TASK_OFFSET

static_assert(
	std::is_standard_layout<CudaRasterizer::MixedWholeHeadTaskV2>::value &&
		std::is_standard_layout<tacker_4dgs::WholeHeadTaskV2>::value,
	"mixed whole-head task descriptors must be standard-layout PODs");
static_assert(
	sizeof(CudaRasterizer::MixedWholeHeadTaskV2) ==
		sizeof(tacker_4dgs::WholeHeadTaskV2),
	"mixed whole-head task descriptor size mismatch");
static_assert(
	alignof(CudaRasterizer::MixedWholeHeadTaskV2) ==
		alignof(tacker_4dgs::WholeHeadTaskV2),
	"mixed whole-head task descriptor alignment mismatch");
#define TACKER_ASSERT_WHOLE_TASK_OFFSET(field) \
	static_assert(offsetof(CudaRasterizer::MixedWholeHeadTaskV2, field) == \
		offsetof(tacker_4dgs::WholeHeadTaskV2, field), \
		"mixed whole-head descriptor field offset mismatch: " #field)
TACKER_ASSERT_WHOLE_TASK_OFFSET(input);
TACKER_ASSERT_WHOLE_TASK_OFFSET(first_weight);
TACKER_ASSERT_WHOLE_TASK_OFFSET(first_bias);
TACKER_ASSERT_WHOLE_TASK_OFFSET(tail_weight);
TACKER_ASSERT_WHOLE_TASK_OFFSET(tail_bias);
TACKER_ASSERT_WHOLE_TASK_OFFSET(output);
TACKER_ASSERT_WHOLE_TASK_OFFSET(rows);
TACKER_ASSERT_WHOLE_TASK_OFFSET(tail_features);
#undef TACKER_ASSERT_WHOLE_TASK_OFFSET

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

extern "C" __global__ void __launch_bounds__(CudaRasterizer::Tacker::kMaxMixedThreadsV2)
tacker_mix_render_heads_v2(
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
	CudaRasterizer::MixedHeadTaskV2 head_task_0,
	CudaRasterizer::MixedHeadTaskV2 head_task_1,
	CudaRasterizer::MixedHeadTaskV2 head_task_2,
	CudaRasterizer::MixedHeadTaskV2 head_task_3,
	CudaRasterizer::MixedHeadTaskV2 head_task_4,
	int head_task_count,
	int worker_groups,
	int max_head_logical_blocks,
	int physical_blocks)
{
	// Raster owns [0, 256) exactly as in ABI v1.  Named barrier 1 still has
	// exactly 256 participants and never includes a head worker.
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

	// CUDA 11 cannot mark an address-taken kernel parameter as grid-constant.
	// Copy the five small descriptors once per CTA instead of making every
	// backend thread materialize a private local-memory copy.  Barrier 2 is
	// restricted to the contiguous backend range and is disjoint from Raster.
	__shared__ CudaRasterizer::MixedHeadTaskV2 shared_head_tasks[
		CudaRasterizer::Tacker::kMaxHeadTasksV2];
	const int backend_thread = static_cast<int>(threadIdx.x) -
		CudaRasterizer::Tacker::kHeadThreadBase;
	const int backend_threads =
		worker_groups * CudaRasterizer::Tacker::kWorkerGroupThreadsV2;
	if (backend_thread >= 0 && backend_thread < backend_threads)
	{
		if (backend_thread < head_task_count)
		{
			switch (backend_thread)
			{
			case 0: shared_head_tasks[0] = head_task_0; break;
			case 1: shared_head_tasks[1] = head_task_1; break;
			case 2: shared_head_tasks[2] = head_task_2; break;
			case 3: shared_head_tasks[3] = head_task_3; break;
			case 4: shared_head_tasks[4] = head_task_4; break;
			}
		}
		CudaRasterizer::Tacker::subgroup_sync(
			CudaRasterizer::Tacker::kHeadDescriptorBarrierIdV2,
			backend_threads);

		tacker_4dgs::head_linear_multi_gptb_device(
			reinterpret_cast<const tacker_4dgs::HeadLinearTaskV2*>(
				shared_head_tasks),
			head_task_count,
			worker_groups,
			0,
			physical_blocks,
			max_head_logical_blocks,
			CudaRasterizer::Tacker::kHeadThreadBase);
	}
}

extern "C" __global__ void __launch_bounds__(CudaRasterizer::Tacker::kMaxMixedThreadsV2)
tacker_mix_render_packed_heads_v3(
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
	const half* packed_weights,
	const float* packed_biases,
	float* packed_outputs,
	int head_rows,
	int head_count,
	int worker_groups,
	int max_head_logical_blocks,
	int physical_blocks)
{
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

	// C3 needs no named or CTA-wide backend barrier.  The adapter rejects the
	// Raster lanes by thread_base and schedules every packed head exactly once.
	tacker_4dgs::head_linear_packed_gptb_device(
		head_input,
		packed_weights,
		packed_biases,
		packed_outputs,
		head_rows,
		head_count,
		worker_groups,
		0,
		physical_blocks,
		max_head_logical_blocks,
		CudaRasterizer::Tacker::kHeadThreadBase);
}

extern "C" __global__ void __launch_bounds__(CudaRasterizer::Tacker::kMaxMixedThreadsV2)
tacker_mix_render_whole_heads_v4(
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
	CudaRasterizer::MixedWholeHeadTaskV2 head_task_0,
	CudaRasterizer::MixedWholeHeadTaskV2 head_task_1,
	CudaRasterizer::MixedWholeHeadTaskV2 head_task_2,
	CudaRasterizer::MixedWholeHeadTaskV2 head_task_3,
	CudaRasterizer::MixedWholeHeadTaskV2 head_task_4,
	int head_task_count,
	int worker_groups,
	int max_head_rows,
	int physical_blocks)
{
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

	__shared__ CudaRasterizer::MixedWholeHeadTaskV2 shared_head_tasks[
		CudaRasterizer::Tacker::kMaxHeadTasksV2];
	__shared__ float shared_hidden[
		CudaRasterizer::Tacker::kMaxWorkerGroupsV2 *
		tacker_4dgs::kWholeHeadScratchFloatsPerGroupV2];
	const int backend_thread = static_cast<int>(threadIdx.x) -
		CudaRasterizer::Tacker::kHeadThreadBase;
	const int backend_threads =
		worker_groups * CudaRasterizer::Tacker::kWorkerGroupThreadsV2;
	if (backend_thread >= 0 && backend_thread < backend_threads)
	{
		if (backend_thread < head_task_count)
		{
			switch (backend_thread)
			{
			case 0: shared_head_tasks[0] = head_task_0; break;
			case 1: shared_head_tasks[1] = head_task_1; break;
			case 2: shared_head_tasks[2] = head_task_2; break;
			case 3: shared_head_tasks[3] = head_task_3; break;
			case 4: shared_head_tasks[4] = head_task_4; break;
			}
		}
		// Barrier 7 has exactly all backend lanes.  Whole-head worker groups then
		// use barriers 2..6 with exactly 128 participants apiece; Raster owns 1.
		CudaRasterizer::Tacker::subgroup_sync(
			CudaRasterizer::Tacker::kWholeHeadDescriptorBarrierIdV2,
			backend_threads);

		tacker_4dgs::whole_head_multi_gptb_device(
			reinterpret_cast<const tacker_4dgs::WholeHeadTaskV2*>(
				shared_head_tasks),
			head_task_count,
			worker_groups,
			0,
			physical_blocks,
			max_head_rows,
			CudaRasterizer::Tacker::kHeadThreadBase,
			shared_hidden,
			CudaRasterizer::Tacker::kWholeHeadBarrierBaseIdV2);
	}
}

namespace
{

void checkCudaStatus(cudaError_t status, const char* operation)
{
	if (status != cudaSuccess)
		throw std::runtime_error(
			std::string(operation) + ": " + cudaGetErrorString(status));
}

const cudaDeviceProp& activeDeviceProperties(int* active_device = nullptr)
{
	int device = 0;
	checkCudaStatus(cudaGetDevice(&device), "cudaGetDevice failed");
	static thread_local int cached_device = -1;
	static thread_local cudaDeviceProp cached_properties;
	if (cached_device != device)
	{
		checkCudaStatus(
			cudaGetDeviceProperties(&cached_properties, device),
			"cudaGetDeviceProperties failed");
		cached_device = device;
	}
	if (active_device != nullptr)
		*active_device = device;
	return cached_properties;
}

int activeSmCount()
{
	return activeDeviceProperties().multiProcessorCount;
}

void throwOnLaunchError()
{
	const cudaError_t status = cudaPeekAtLastError();
	if (status == cudaSuccess)
		return;
	const std::string message = cudaGetErrorString(status);
	// Peek is intentionally non-synchronizing, but it leaves the host thread's
	// last-error slot set.  Clear that slot before Python replays the sequence
	// through the safe fallback; otherwise its first valid launch can observe
	// this stale configuration error and fail a second time.
	(void)cudaGetLastError();
	throw std::runtime_error(message);
}

int headLogicalBlocks(int rows)
{
	const int64_t grid_x = (static_cast<int64_t>(rows) + 15) / 16;
	const int64_t logical_blocks = rows == 0 ? 0 : grid_x * 2;
	if (logical_blocks > INT_MAX)
		throw std::overflow_error("head logical grid exceeds int32 mixed ABI v2");
	return static_cast<int>(logical_blocks);
}

void validateMixedHeadBundleV2(const CudaRasterizer::MixedHeadBundleV2& heads)
{
	if (heads.task_count < 1 ||
		heads.task_count > CudaRasterizer::Tacker::kMaxHeadTasksV2)
		throw std::invalid_argument("mixed v2 task_count must be in [1, 5]");
	if (heads.worker_groups < CudaRasterizer::Tacker::kMinWorkerGroupsV2 ||
		heads.worker_groups > heads.task_count)
		throw std::invalid_argument(
			"mixed v2 worker_groups must be in [1, task_count]");
	if (heads.persistent_blocks < 0)
		throw std::invalid_argument("persistent_blocks must be >= 0");

	for (int task_index = 0; task_index < heads.task_count; ++task_index)
	{
		const CudaRasterizer::MixedHeadTaskV2& task = heads.tasks[task_index];
		if (task.rows < 0)
			throw std::invalid_argument("mixed v2 head rows must be >= 0");
		if (task.rows > 0 &&
			(task.input == nullptr || task.weight == nullptr ||
			 task.bias == nullptr || task.output == nullptr))
			throw std::invalid_argument(
				"mixed v2 head pointers must be non-null when rows > 0");
		if (task.rows == 0)
			continue;
		for (int other = 0; other < task_index; ++other)
		{
			if (heads.tasks[other].rows > 0 &&
				heads.tasks[other].output == task.output)
				throw std::invalid_argument(
					"mixed v2 head outputs must not alias each other");
		}
	}
}

void validateMixedPackedHeadBundleV2(
	const CudaRasterizer::MixedPackedHeadBundleV2& heads)
{
	if (heads.head_count < 1 ||
		heads.head_count > CudaRasterizer::Tacker::kMaxHeadTasksV2)
		throw std::invalid_argument("mixed packed head_count must be in [1, 5]");
	if (heads.worker_groups < CudaRasterizer::Tacker::kMinWorkerGroupsV2 ||
		heads.worker_groups > heads.head_count)
		throw std::invalid_argument(
			"mixed packed worker_groups must be in [1, head_count]");
	if (heads.persistent_blocks < 0)
		throw std::invalid_argument("persistent_blocks must be >= 0");
	if (heads.rows < 0)
		throw std::invalid_argument("mixed packed rows must be >= 0");
	if (heads.rows > 0 &&
		(heads.input == nullptr || heads.weights == nullptr ||
		 heads.biases == nullptr || heads.output == nullptr))
		throw std::invalid_argument(
			"mixed packed pointers must be non-null when rows > 0");
}

void validateMixedWholeHeadBundleV2(
	const CudaRasterizer::MixedWholeHeadBundleV2& heads)
{
	if (heads.task_count < 1 ||
		heads.task_count > CudaRasterizer::Tacker::kMaxHeadTasksV2)
		throw std::invalid_argument(
			"mixed whole-head task_count must be in [1, 5]");
	if (heads.worker_groups < CudaRasterizer::Tacker::kMinWorkerGroupsV2 ||
		heads.worker_groups > heads.task_count)
		throw std::invalid_argument(
			"mixed whole-head worker_groups must be in [1, task_count]");
	if (heads.persistent_blocks < 0)
		throw std::invalid_argument("persistent_blocks must be >= 0");

	for (int task_index = 0; task_index < heads.task_count; ++task_index)
	{
		const CudaRasterizer::MixedWholeHeadTaskV2& task =
			heads.tasks[task_index];
		if (task.rows < 0)
			throw std::invalid_argument(
				"mixed whole-head rows must be >= 0");
		if (task.tail_features < 1 ||
			task.tail_features > CudaRasterizer::Tacker::kMaxTailFeaturesV2)
			throw std::invalid_argument(
				"mixed whole-head tail_features must be in [1, 128]");
		if (task.rows > 0 &&
			(task.input == nullptr || task.first_weight == nullptr ||
			 task.first_bias == nullptr || task.tail_weight == nullptr ||
			 task.tail_bias == nullptr || task.output == nullptr))
			throw std::invalid_argument(
				"mixed whole-head pointers must be non-null when rows > 0");
		if (task.rows == 0)
			continue;
		for (int other = 0; other < task_index; ++other)
		{
			if (heads.tasks[other].rows > 0 &&
				heads.tasks[other].output == task.output)
				throw std::invalid_argument(
					"mixed whole-head outputs must not alias each other");
		}
	}
}

CudaRasterizer::Tacker::MixedKernelResources queryKernelResources(
	int abi_version,
	CudaRasterizer::Tacker::MixedBackendFamily family,
	int worker_groups,
	int physical_threads,
	const void* kernel)
{
	int device = 0;
	const cudaDeviceProp& properties = activeDeviceProperties(&device);
	cudaFuncAttributes attributes;
	checkCudaStatus(
		cudaFuncGetAttributes(&attributes, kernel),
		"cudaFuncGetAttributes failed");

	CudaRasterizer::Tacker::MixedKernelResources result = {};
	result.abi_version = abi_version;
	result.backend_family = family;
	result.worker_groups = worker_groups;
	result.physical_threads = physical_threads;
	result.device_ordinal = device;
	result.compute_capability_major = properties.major;
	result.compute_capability_minor = properties.minor;
	result.multiprocessor_count = properties.multiProcessorCount;
	result.device_max_threads_per_block = properties.maxThreadsPerBlock;
	result.device_max_threads_per_multiprocessor =
		properties.maxThreadsPerMultiProcessor;
	result.warp_size = properties.warpSize;
	result.kernel_max_threads_per_block = attributes.maxThreadsPerBlock;
	result.registers_per_thread = attributes.numRegs;
	result.static_shared_bytes = attributes.sharedSizeBytes;
	result.local_bytes_per_thread = attributes.localSizeBytes;
#if CUDART_VERSION >= 9000
	result.max_dynamic_shared_bytes = attributes.maxDynamicSharedSizeBytes;
#else
	result.max_dynamic_shared_bytes = 0;
#endif
	result.max_warps_per_multiprocessor = properties.warpSize > 0
		? properties.maxThreadsPerMultiProcessor / properties.warpSize
		: 0;

	if (physical_threads <= 0 ||
		physical_threads > properties.maxThreadsPerBlock ||
		physical_threads > attributes.maxThreadsPerBlock)
	{
		result.launch_supported = false;
		return result;
	}

	int active_blocks = 0;
	checkCudaStatus(
		cudaOccupancyMaxActiveBlocksPerMultiprocessor(
			&active_blocks, kernel, physical_threads, 0),
		"cudaOccupancyMaxActiveBlocksPerMultiprocessor failed");
	result.active_blocks_per_multiprocessor = active_blocks;
	const int warps_per_block =
		(physical_threads + properties.warpSize - 1) / properties.warpSize;
	result.active_warps_per_multiprocessor = active_blocks * warps_per_block;
	result.occupancy = result.max_warps_per_multiprocessor > 0
		? static_cast<double>(result.active_warps_per_multiprocessor) /
			static_cast<double>(result.max_warps_per_multiprocessor)
		: 0.0;
	result.launch_supported = active_blocks > 0;
	return result;
}

void ensureLaunchSupported(
	int abi_version,
	CudaRasterizer::Tacker::MixedBackendFamily family,
	int worker_groups,
	const void* kernel)
{
	int device = 0;
	activeDeviceProperties(&device);
	static thread_local int cached_device = -1;
	static thread_local unsigned int checked_mask = 0;
	static thread_local unsigned int supported_mask = 0;
	if (cached_device != device)
	{
		cached_device = device;
		checked_mask = 0;
		supported_mask = 0;
	}
	const unsigned int family_index =
		static_cast<unsigned int>(family);
	const unsigned int bit = 1u << (family_index * 8u + worker_groups);
	if ((checked_mask & bit) == 0)
	{
		const CudaRasterizer::Tacker::MixedKernelResources resources =
			queryKernelResources(
				abi_version,
				family,
				worker_groups,
				CudaRasterizer::Tacker::kRasterThreads +
					worker_groups *
						CudaRasterizer::Tacker::kWorkerGroupThreadsV2,
				kernel);
		checked_mask |= bit;
		if (resources.launch_supported)
			supported_mask |= bit;
	}
	if ((supported_mask & bit) == 0)
		throw std::runtime_error(
			"mixed backend kernel has zero active blocks or exceeds the active "
			"device/kernel thread limit");
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
	throwOnLaunchError();
}

void CudaRasterizer::Tacker::launchMixedRenderHeads(
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
	cudaStream_t stream)
{
	validateMixedHeadBundleV2(heads);

	const int64_t raster_blocks_64 =
		static_cast<int64_t>(raster_grid.x) * static_cast<int64_t>(raster_grid.y);
	if (raster_blocks_64 > INT_MAX)
		throw std::overflow_error("raster logical grid exceeds int32 mixed ABI v2");

	int max_head_logical_blocks = 0;
	for (int task_index = 0; task_index < heads.task_count; ++task_index)
	{
		max_head_logical_blocks = std::max(
			max_head_logical_blocks,
			headLogicalBlocks(heads.tasks[task_index].rows));
	}
	const int logical_blocks = std::max(
		static_cast<int>(raster_blocks_64), max_head_logical_blocks);
	if (logical_blocks == 0)
		return;
	ensureLaunchSupported(
		kMixedMultiAbiVersion,
		MixedBackendFamily::FirstLinear,
		heads.worker_groups,
		reinterpret_cast<const void*>(tacker_mix_render_heads_v2));

	int physical_blocks = heads.persistent_blocks;
	if (physical_blocks == 0)
		physical_blocks = activeSmCount();
	physical_blocks = std::max(1, std::min(physical_blocks, logical_blocks));
	const int physical_threads = kRasterThreads +
		heads.worker_groups * kWorkerGroupThreadsV2;

	// Unused descriptors are zero-initialized by the binding/host contract and
	// are still passed explicitly so the global kernel argument ABI is fixed.
	tacker_mix_render_heads_v2<<<
		physical_blocks, physical_threads, 0, stream>>>(
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
		heads.tasks[0],
		heads.tasks[1],
		heads.tasks[2],
		heads.tasks[3],
		heads.tasks[4],
		heads.task_count,
		heads.worker_groups,
		max_head_logical_blocks,
		physical_blocks);

	// Configuration/resource/argument failures are surfaced synchronously to
	// the caller without waiting for either independent task to complete.
	throwOnLaunchError();
}

void CudaRasterizer::Tacker::launchMixedRenderPackedHeads(
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
	const MixedPackedHeadBundleV2& heads,
	cudaStream_t stream)
{
	validateMixedPackedHeadBundleV2(heads);
	const int64_t raster_blocks_64 =
		static_cast<int64_t>(raster_grid.x) * static_cast<int64_t>(raster_grid.y);
	if (raster_blocks_64 > INT_MAX)
		throw std::overflow_error(
			"raster logical grid exceeds int32 mixed packed ABI v3");
	const int max_head_logical_blocks = headLogicalBlocks(heads.rows);
	const int logical_blocks = std::max(
		static_cast<int>(raster_blocks_64), max_head_logical_blocks);
	if (logical_blocks == 0)
		return;
	ensureLaunchSupported(
		kMixedPackedAbiVersion,
		MixedBackendFamily::PackedFirstLinear,
		heads.worker_groups,
		reinterpret_cast<const void*>(tacker_mix_render_packed_heads_v3));

	int physical_blocks = heads.persistent_blocks;
	if (physical_blocks == 0)
		physical_blocks = activeSmCount();
	physical_blocks = std::max(1, std::min(physical_blocks, logical_blocks));
	const int physical_threads = kRasterThreads +
		heads.worker_groups * kWorkerGroupThreadsV2;

	tacker_mix_render_packed_heads_v3<<<
		physical_blocks, physical_threads, 0, stream>>>(
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
		reinterpret_cast<const half*>(heads.input),
		reinterpret_cast<const half*>(heads.weights),
		heads.biases,
		heads.output,
		heads.rows,
		heads.head_count,
		heads.worker_groups,
		max_head_logical_blocks,
		physical_blocks);
	throwOnLaunchError();
}

void CudaRasterizer::Tacker::launchMixedRenderWholeHeads(
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
	const MixedWholeHeadBundleV2& heads,
	cudaStream_t stream)
{
	validateMixedWholeHeadBundleV2(heads);
	const int64_t raster_blocks_64 =
		static_cast<int64_t>(raster_grid.x) * static_cast<int64_t>(raster_grid.y);
	if (raster_blocks_64 > INT_MAX)
		throw std::overflow_error(
			"raster logical grid exceeds int32 mixed whole-head ABI v4");
	int max_head_rows = 0;
	for (int task_index = 0; task_index < heads.task_count; ++task_index)
		max_head_rows = std::max(max_head_rows, heads.tasks[task_index].rows);
	const int logical_blocks = std::max(
		static_cast<int>(raster_blocks_64), max_head_rows);
	if (logical_blocks == 0)
		return;
	ensureLaunchSupported(
		kMixedWholeHeadAbiVersion,
		MixedBackendFamily::WholeHead,
		heads.worker_groups,
		reinterpret_cast<const void*>(tacker_mix_render_whole_heads_v4));

	int physical_blocks = heads.persistent_blocks;
	if (physical_blocks == 0)
		physical_blocks = activeSmCount();
	physical_blocks = std::max(1, std::min(physical_blocks, logical_blocks));
	const int physical_threads = kRasterThreads +
		heads.worker_groups * kWorkerGroupThreadsV2;

	tacker_mix_render_whole_heads_v4<<<
		physical_blocks, physical_threads, 0, stream>>>(
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
		heads.tasks[0],
		heads.tasks[1],
		heads.tasks[2],
		heads.tasks[3],
		heads.tasks[4],
		heads.task_count,
		heads.worker_groups,
		max_head_rows,
		physical_blocks);
	throwOnLaunchError();
}

const char* CudaRasterizer::Tacker::mixedBackendFamilyName(
	MixedBackendFamily family)
{
	switch (family)
	{
	case MixedBackendFamily::FirstLinear:
		return "first_linear_heads_v2";
	case MixedBackendFamily::PackedFirstLinear:
		return "packed_first_linear_v3";
	case MixedBackendFamily::WholeHead:
		return "whole_heads_v4";
	}
	throw std::invalid_argument("unsupported mixed backend family");
}

CudaRasterizer::Tacker::MixedKernelResources
CudaRasterizer::Tacker::queryMixedKernelResources(
	int abi_version,
	int worker_groups)
{
	return queryMixedKernelResources(
		abi_version, worker_groups, MixedBackendFamily::FirstLinear);
}

CudaRasterizer::Tacker::MixedKernelResources
CudaRasterizer::Tacker::queryMixedKernelResources(
	int abi_version,
	int worker_groups,
	MixedBackendFamily family)
{
	if (abi_version == kMixedAbiVersion)
	{
		if (family != MixedBackendFamily::FirstLinear)
			throw std::invalid_argument(
				"mixed ABI v1 only supports the first-linear family");
		if (worker_groups != 1)
			throw std::invalid_argument(
				"mixed ABI v1 requires worker_groups == 1");
		return queryKernelResources(
			abi_version,
			family,
			worker_groups,
			kMixedThreads,
			reinterpret_cast<const void*>(tacker_mix_render_head_v1));
	}
	if (abi_version == kMixedMultiAbiVersion)
	{
		if (family != MixedBackendFamily::FirstLinear)
			throw std::invalid_argument(
				"mixed ABI v2 requires family first_linear_heads_v2");
		if (worker_groups < kMinWorkerGroupsV2 ||
			worker_groups > kMaxWorkerGroupsV2)
			throw std::invalid_argument(
				"mixed ABI v2 worker_groups must be in [1, 5]");
		return queryKernelResources(
			abi_version,
			family,
			worker_groups,
			kRasterThreads + worker_groups * kWorkerGroupThreadsV2,
			reinterpret_cast<const void*>(tacker_mix_render_heads_v2));
	}
	if (abi_version == kMixedPackedAbiVersion)
	{
		if (family != MixedBackendFamily::PackedFirstLinear)
			throw std::invalid_argument(
				"mixed ABI v3 requires family packed_first_linear_v3");
		if (worker_groups < kMinWorkerGroupsV2 ||
			worker_groups > kMaxWorkerGroupsV2)
			throw std::invalid_argument(
				"mixed ABI v3 worker_groups must be in [1, 5]");
		return queryKernelResources(
			abi_version,
			family,
			worker_groups,
			kRasterThreads + worker_groups * kWorkerGroupThreadsV2,
			reinterpret_cast<const void*>(tacker_mix_render_packed_heads_v3));
	}
	if (abi_version == kMixedWholeHeadAbiVersion)
	{
		if (family != MixedBackendFamily::WholeHead)
			throw std::invalid_argument(
				"mixed ABI v4 requires family whole_heads_v4");
		if (worker_groups < kMinWorkerGroupsV2 ||
			worker_groups > kMaxWorkerGroupsV2)
			throw std::invalid_argument(
				"mixed ABI v4 worker_groups must be in [1, 5]");
		return queryKernelResources(
			abi_version,
			family,
			worker_groups,
			kRasterThreads + worker_groups * kWorkerGroupThreadsV2,
			reinterpret_cast<const void*>(tacker_mix_render_whole_heads_v4));
	}
	throw std::invalid_argument("unsupported mixed ABI version");
}
