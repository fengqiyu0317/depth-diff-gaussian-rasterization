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
#include "cuda_rasterizer/tacker_mixed.h"

namespace py = pybind11;

py::dict TackerCapabilities() {
  py::dict capabilities;
  capabilities["stream_aware"] = true;
  capabilities["mixed_render_head_abi"] = CudaRasterizer::Tacker::kMixedAbiVersion;
  capabilities["mixed_render_head"] = true;
  capabilities["mixed_symbol"] = "tacker_mix_render_head_v1";
  capabilities["mixed_threads"] = CudaRasterizer::Tacker::kMixedThreads;
  capabilities["raster_threads"] = CudaRasterizer::Tacker::kRasterThreads;
  capabilities["head_threads"] = CudaRasterizer::Tacker::kHeadThreads;
  capabilities["head_thread_base"] = CudaRasterizer::Tacker::kHeadThreadBase;
  capabilities["raster_named_barrier_id"] = CudaRasterizer::Tacker::kRasterBarrierId;
  capabilities["head_features"] = 128;
  capabilities["rasterizer_commit"] = "e49506654e8e11ed8a62d22bcb693e943fdecacf";
  capabilities["sm_target"] = "sm_86";
  capabilities["mixed_manifest_sha256"] =
      "231c90c429321b2673b88ecd09efb40b6aedda7a23f3e061a2bcedec06d44426";
  capabilities["head_manifest_sha256"] =
      "24570aa6e67e8b9b10fa94524fec4dc03a4eb3fdc3bf822af34c2c52ce4937ac";

  // ABI v1 fields above are intentionally unchanged.  ABI v2 describes a
  // generic 1--5 first-linear bundle; semantic head identities live in the
  // caller's candidate descriptor, not in this physical wrapper.
  capabilities["mixed_render_heads_abi"] =
      CudaRasterizer::Tacker::kMixedMultiAbiVersion;
  capabilities["mixed_render_heads"] = true;
  capabilities["mixed_multi_symbol"] = "tacker_mix_render_heads_v2";
  capabilities["mixed_multi_manifest"] =
      "abi/tacker_mixed_render_heads_v2.json";
  capabilities["mixed_multi_manifest_sha256"] =
      "310b15957c5920773bb03a61a37c5771f6d4570393061ece4e1805fd20989056";
  capabilities["head_multi_manifest_sha256"] =
      "9d6a1558acd6b642b975bcabe22abcbe3fd7242e4c9e0d635636ef4d2eb5da7f";
  capabilities["max_head_tasks"] = CudaRasterizer::Tacker::kMaxHeadTasksV2;
  capabilities["max_mixed_heads"] = CudaRasterizer::Tacker::kMaxHeadTasksV2;
  capabilities["min_worker_groups"] =
      CudaRasterizer::Tacker::kMinWorkerGroupsV2;
  capabilities["max_worker_groups"] =
      CudaRasterizer::Tacker::kMaxWorkerGroupsV2;
  capabilities["worker_group_threads"] =
      CudaRasterizer::Tacker::kWorkerGroupThreadsV2;
  capabilities["head_descriptor_named_barrier_id"] =
      CudaRasterizer::Tacker::kHeadDescriptorBarrierIdV2;
  py::list supported_worker_groups;
  py::dict threads_by_worker_groups;
  for (int worker_groups = CudaRasterizer::Tacker::kMinWorkerGroupsV2;
       worker_groups <= CudaRasterizer::Tacker::kMaxWorkerGroupsV2;
       ++worker_groups) {
    supported_worker_groups.append(worker_groups);
    threads_by_worker_groups[py::int_(worker_groups)] =
        CudaRasterizer::Tacker::kRasterThreads +
        worker_groups * CudaRasterizer::Tacker::kWorkerGroupThreadsV2;
  }
  capabilities["supported_worker_groups"] = supported_worker_groups;
  capabilities["mixed_threads_by_worker_groups"] = threads_by_worker_groups;
  capabilities["resource_query"] = "tacker_resource_requirements";
  return capabilities;
}

py::dict TackerResourceRequirements(int abi_version, int worker_groups) {
  const CudaRasterizer::Tacker::MixedKernelResources resources =
      CudaRasterizer::Tacker::queryMixedKernelResources(
          abi_version, worker_groups);
  py::dict result;
  result["abi_version"] = resources.abi_version;
  result["worker_groups"] = resources.worker_groups;
  result["physical_threads"] = resources.physical_threads;
  result["device_ordinal"] = resources.device_ordinal;
  result["compute_capability_major"] = resources.compute_capability_major;
  result["compute_capability_minor"] = resources.compute_capability_minor;
  result["multiprocessor_count"] = resources.multiprocessor_count;
  result["device_max_threads_per_block"] =
      resources.device_max_threads_per_block;
  result["device_max_threads_per_multiprocessor"] =
      resources.device_max_threads_per_multiprocessor;
  result["warp_size"] = resources.warp_size;
  result["kernel_max_threads_per_block"] =
      resources.kernel_max_threads_per_block;
  result["registers_per_thread"] = resources.registers_per_thread;
  result["static_shared_bytes"] = resources.static_shared_bytes;
  result["local_bytes_per_thread"] = resources.local_bytes_per_thread;
  result["max_dynamic_shared_bytes"] = resources.max_dynamic_shared_bytes;
  result["active_blocks_per_multiprocessor"] =
      resources.active_blocks_per_multiprocessor;
  result["active_warps_per_multiprocessor"] =
      resources.active_warps_per_multiprocessor;
  result["max_warps_per_multiprocessor"] =
      resources.max_warps_per_multiprocessor;
  result["occupancy"] = resources.occupancy;
  result["launch_supported"] = resources.launch_supported;
  return result;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("rasterize_gaussians", &RasterizeGaussiansCUDA);
  m.def(
      "rasterize_gaussians_with_head",
      &RasterizeGaussiansWithHeadCUDA,
      "Inference-only physical Raster+head mixed kernel");
  m.def(
      "rasterize_gaussians_with_heads",
      &RasterizeGaussiansWithHeadsCUDA,
      "Inference-only physical Raster+1--5 first-linear heads mixed kernel");
  m.def("rasterize_gaussians_backward", &RasterizeGaussiansBackwardCUDA);
  m.def("mark_visible", &markVisible);
  m.def("tacker_capabilities", &TackerCapabilities);
  m.def(
      "tacker_resource_requirements",
      &TackerResourceRequirements,
      py::arg("abi_version") = CudaRasterizer::Tacker::kMixedMultiAbiVersion,
      py::arg("worker_groups") = 1,
      "Query active-device registers, shared memory, and occupancy");
}
