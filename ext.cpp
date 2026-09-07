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
  return capabilities;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("rasterize_gaussians", &RasterizeGaussiansCUDA);
  m.def(
      "rasterize_gaussians_with_head",
      &RasterizeGaussiansWithHeadCUDA,
      "Inference-only physical Raster+head mixed kernel");
  m.def("rasterize_gaussians_backward", &RasterizeGaussiansBackwardCUDA);
  m.def("mark_visible", &markVisible);
  m.def("tacker_capabilities", &TackerCapabilities);
}
