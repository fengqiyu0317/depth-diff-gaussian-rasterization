"""Torch/CUDA-free source contract tests for the mixed Raster+head ABI."""

import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


def source(relative_path):
    return (ROOT / relative_path).read_text(encoding="utf-8")


class MixedKernelContractTest(unittest.TestCase):
    def test_wrapper_is_one_real_384_thread_kernel(self):
        header = source("cuda_rasterizer/tacker_mixed.h")
        cuda = source("cuda_rasterizer/tacker_mixed.cu")
        self.assertIn("kMixedThreads = 384", header)
        self.assertIn("tacker_mix_render_head_v1", cuda)
        self.assertIn("<<<\n\t\tphysical_blocks, kMixedThreads", cuda)

    def test_wrapper_calls_both_real_device_adapters(self):
        cuda = source("cuda_rasterizer/tacker_mixed.cu")
        adapter = source("cuda_rasterizer/tacker_forward.cuh")
        self.assertIn("general_ptb_render<NUM_CHANNELS>", cuda)
        self.assertIn("tacker_4dgs::head_linear_gptb_device", cuda)
        self.assertIn("#include \"head_linear_device.cuh\"", cuda)
        self.assertNotIn("#define GLM_FORCE_CUDA", adapter)

    def test_thread_ranges_and_barrier_are_disjoint(self):
        header = source("cuda_rasterizer/tacker_mixed.h")
        cuda = source("cuda_rasterizer/tacker_mixed.cu")
        self.assertIn("kRasterThreads = 256", header)
        self.assertIn("kHeadThreads = 128", header)
        self.assertIn("kHeadThreadBase = 256", header)
        self.assertIn("kRasterBarrierId = 1", header)
        self.assertIn("static_assert(", cuda)
        self.assertNotIn("__syncthreads(", cuda)

    def test_zero_policy_queries_device_and_launch_check_is_async(self):
        cuda = source("cuda_rasterizer/tacker_mixed.cu")
        self.assertIn("cudaGetDeviceProperties", cuda)
        self.assertIn("cached_sm_count", cuda)
        self.assertIn("cudaPeekAtLastError", cuda)
        self.assertNotIn("cudaDeviceSynchronize", cuda)
        self.assertNotIn("physical_blocks = 68", cuda)
        self.assertNotIn("physical_blocks = 142", cuda)

    def test_manifest_matches_source_contract(self):
        manifest = json.loads(
            source("abi/tacker_mixed_render_head_v1.json")
        )
        self.assertEqual(manifest["abi_version"], 1)
        self.assertEqual(manifest["cuda_arch"], "sm_86")
        self.assertEqual(manifest["global_kernel_symbol"], "tacker_mix_render_head_v1")
        self.assertEqual(
            manifest["subgroups"]["raster"]["thread_range"], [0, 255]
        )
        self.assertEqual(
            manifest["subgroups"]["head"]["thread_range"], [256, 383]
        )
        self.assertEqual(manifest["subgroups"]["raster"]["named_barrier_id"], 1)


class ExtensionApiContractTest(unittest.TestCase):
    def test_old_binding_and_forward_path_remain(self):
        extension = source("ext.cpp")
        implementation = source("cuda_rasterizer/rasterizer_impl.cu")
        self.assertIn('m.def("rasterize_gaussians", &RasterizeGaussiansCUDA)', extension)
        self.assertIn("if (mixed_head != nullptr)", implementation)
        self.assertIn("FORWARD::render(", implementation)
        self.assertIn("nullptr);", implementation)

    def test_new_binding_and_capability_query_are_exposed(self):
        extension = source("ext.cpp")
        python_api = source("diff_gaussian_rasterization/__init__.py")
        self.assertIn("rasterize_gaussians_with_head", extension)
        self.assertIn("tacker_capabilities", extension)
        self.assertIn('capabilities["stream_aware"] = true', extension)
        self.assertIn("def tacker_capabilities():", python_api)
        self.assertIn("def forward_with_head(", python_api)

    def test_inference_guard_and_no_backward_registration(self):
        python_api = source("diff_gaussian_rasterization/__init__.py")
        extension = source("ext.cpp")
        self.assertIn("torch.is_grad_enabled()", python_api)
        self.assertIn("call it under torch.no_grad()", python_api)
        self.assertNotIn(
            'm.def("rasterize_gaussians_with_head_backward"', extension
        )

    def test_cpp_checks_head_contract_and_int32_rows(self):
        binding = source("rasterize_points.cu")
        for required in (
            "head_input must be contiguous",
            "head_weight must be contiguous",
            "head_bias must be contiguous",
            "head_input must be float16",
            "head_weight must be float16",
            "head_bias must be float32",
            "head_input must have shape [N, 128]",
            "head_weight must have shape [128, 128]",
            "head_input row count exceeds the int32 mixed ABI",
            "head_input data pointer must be natively 32-byte aligned",
            "head_weight data pointer must be natively 32-byte aligned",
            "head_output data pointer must be natively 32-byte aligned",
        ):
            self.assertIn(required, binding)

    def test_cpp_validates_complete_raster_contract_and_empty_shape_tags(self):
        binding = source("rasterize_points.cu")
        for required in (
            "background must have shape [3]",
            "opacity must have shape [P, 1]",
            "viewmatrix must have shape [4, 4]",
            "projmatrix must have shape [4, 4]",
            "campos must have shape [3]",
            "provide exactly one of sh [P, M, 3] or colors [P, 3]",
            "provide exactly one of cov3D_precomp [P, 6]",
            "image_height must be > 0",
            "image_width must be > 0",
            "image dimensions exceed the int32 Raster color-indexing ABI",
            "degree must be in [0, 3]",
            "sh coefficient count is too small for degree",
        ):
            self.assertIn(required, binding)
        self.assertIn("bool isPlaceholder", binding)
        self.assertIn("sh.dim() == 3", binding)
        self.assertIn("colors.dim() == 2", binding)
        self.assertNotIn("sh_contiguous.numel() == 0", binding)

    def test_legacy_forward_backward_and_visibility_use_current_stream(self):
        binding = source("rasterize_points.cu")
        rasterizer_header = source("cuda_rasterizer/rasterizer.h")
        rasterizer_impl = source("cuda_rasterizer/rasterizer_impl.cu")
        backward_header = source("cuda_rasterizer/backward.h")
        backward_cuda = source("cuda_rasterizer/backward.cu")
        self.assertGreaterEqual(binding.count("c10::cuda::CUDAGuard"), 4)
        self.assertGreaterEqual(
            binding.count("at::cuda::getCurrentCUDAStream().stream()"), 4
        )
        self.assertIn("cudaStream_t stream = nullptr", rasterizer_header)
        self.assertIn("dL_ddepth,\n\t\tstream), debug)", rasterizer_impl)
        self.assertIn("cudaStream_t stream = nullptr", backward_header)
        preprocess_kernel_section = backward_cuda.split(
            "void BACKWARD::preprocess", 1
        )[0]
        self.assertNotIn("cudaStream_t stream", preprocess_kernel_section)
        self.assertIn("glm::vec4* dL_drot,\n\tcudaStream_t stream)", backward_cuda)
        self.assertIn("256, 0, stream", backward_cuda)
        self.assertIn("grid, block, 0, stream", backward_cuda)

    def test_rendered_count_cuda_api_errors_are_checked_without_debug(self):
        implementation = source("cuda_rasterizer/rasterizer_impl.cu")
        self.assertIn("rendered_copy_status = cudaMemcpyAsync", implementation)
        self.assertIn("rendered_copy_status != cudaSuccess", implementation)
        self.assertIn("rendered_sync_status = cudaStreamSynchronize", implementation)
        self.assertIn("rendered_sync_status != cudaSuccess", implementation)
        self.assertIn("uint32_t* value", implementation)
        self.assertIn("num_rendered_u32", implementation)
        self.assertIn(
            "rendered Gaussian instance count exceeds the int32 Raster ABI",
            implementation,
        )

    def test_zero_gaussian_path_still_launches_head_wrapper(self):
        binding = source("rasterize_points.cu")
        self.assertIn("if (P != 0)", binding)
        self.assertIn("dim3(0, 0, 1)", binding)
        self.assertIn("Preserve the legacy P==0 raster outputs", binding)

    def test_setup_uses_absolute_paths_and_compiles_mixed_source(self):
        setup = source("setup.py")
        self.assertIn("Path(__file__).resolve().parent", setup)
        self.assertIn('"tacker_mixed.cu"', setup)
        self.assertIn("TACKER_4DGS_HEAD_INCLUDE", setup)
        self.assertIn('os.environ["TORCH_CUDA_ARCH_LIST"] = "8.6"', setup)
        self.assertIn('"-Xptxas=-v"', setup)


if __name__ == "__main__":
    unittest.main()
