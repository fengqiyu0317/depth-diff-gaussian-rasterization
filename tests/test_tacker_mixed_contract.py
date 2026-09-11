"""Torch/CUDA-free source contract tests for the mixed Raster+head ABI."""

import json
import hashlib
from pathlib import Path
import re
import unittest


ROOT = Path(__file__).resolve().parents[1]


def source(relative_path):
    return (ROOT / relative_path).read_text(encoding="utf-8")


def manifest_sha256(relative_path):
    return hashlib.sha256((ROOT / relative_path).read_bytes()).hexdigest()


def capability_sha256(extension, key):
    match = re.search(
        rf'capabilities\["{re.escape(key)}"\]\s*=\s*"([0-9a-f]{{64}})";',
        extension,
    )
    if match is None:
        raise AssertionError(f"missing SHA-256 capability {key!r}")
    return match.group(1)


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
        self.assertIn("cached_properties", cuda)
        self.assertIn("cudaPeekAtLastError", cuda)
        self.assertIn("(void)cudaGetLastError()", cuda)
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

    def test_v2_wrapper_supports_one_to_five_heads_and_dynamic_ctas(self):
        header = source("cuda_rasterizer/tacker_mixed.h")
        cuda = source("cuda_rasterizer/tacker_mixed.cu")
        adapter = source("../../tacker_ext/include/head_linear_v2_device.cuh")
        self.assertIn("kMixedMultiAbiVersion = 2", header)
        self.assertIn("kMaxHeadTasksV2 = 5", header)
        self.assertIn("kMaxMixedThreadsV2", header)
        self.assertIn("tacker_mix_render_heads_v2", cuda)
        self.assertIn("head_linear_multi_gptb_device", cuda)
        self.assertIn("head_linear_multi_gptb_device", adapter)
        self.assertIn(
            "heads.worker_groups * kWorkerGroupThreadsV2", cuda
        )
        self.assertIn("heads.tasks[4]", cuda)

    def test_v2_named_barriers_are_disjoint_and_never_cta_wide(self):
        header = source("cuda_rasterizer/tacker_mixed.h")
        cuda = source("cuda_rasterizer/tacker_mixed.cu")
        self.assertIn("kRasterBarrierId = 1", header)
        self.assertIn("kHeadDescriptorBarrierIdV2 = 2", header)
        self.assertIn("backend_threads", cuda)
        self.assertIn("shared_head_tasks", cuda)
        self.assertNotIn("__syncthreads(", cuda)

    def test_v2_manifest_matches_source_and_covers_c1_c2(self):
        manifest_text = source("abi/tacker_mixed_render_heads_v2.json")
        manifest = json.loads(manifest_text)
        self.assertEqual(manifest["abi_version"], 2)
        self.assertEqual(
            manifest["global_kernel_symbol"], "tacker_mix_render_heads_v2"
        )
        self.assertEqual(manifest["limits"]["task_count"], [1, 5])
        self.assertEqual(
            manifest["physical_launch"]["thread_counts_by_worker_groups"],
            {"1": 384, "2": 512, "3": 640, "4": 768, "5": 896},
        )
        self.assertEqual(
            manifest["candidate_coverage"]["C1_head_names"],
            ["pos", "scales", "rotations", "opacity", "shs"],
        )
        self.assertEqual(
            manifest["candidate_coverage"]["required_dual_head_variant"][
                "supported_worker_groups"
            ],
            [1, 2],
        )
        self.assertEqual(
            manifest["subgroups"]["head_workers"][
                "descriptor_named_barrier_id"
            ],
            2,
        )
        # Make the byte-level artifact consumed by the parent profile easy to
        # reproduce without importing torch or normalizing JSON.
        self.assertEqual(
            len(hashlib.sha256(manifest_text.encode("utf-8")).hexdigest()), 64
        )


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

    def test_v2_binding_capabilities_and_resource_query_are_exposed(self):
        extension = source("ext.cpp")
        python_api = source("diff_gaussian_rasterization/__init__.py")
        header = source("rasterize_points.h")
        for required in (
            "rasterize_gaussians_with_heads",
            "mixed_render_heads_abi",
            "max_mixed_heads",
            "supported_worker_groups",
            "tacker_resource_requirements",
        ):
            self.assertIn(required, extension)
        self.assertIn("RasterizeGaussiansWithHeadsCUDA", header)
        self.assertIn("def forward_with_heads(", python_api)
        self.assertIn("def tacker_variant_resources(", python_api)
        for resource_key in (
            "block_threads",
            "registers_per_thread",
            "static_shared_memory_bytes",
            "max_threads_per_block",
            "active_blocks_per_sm",
        ):
            self.assertIn(resource_key, python_api)

    def test_capability_manifest_hashes_match_exact_artifact_bytes(self):
        extension = source("ext.cpp")
        manifest_by_capability = {
            "mixed_manifest_sha256": "abi/tacker_mixed_render_head_v1.json",
            "head_manifest_sha256": "../../tacker_ext/abi/head_linear_v1.json",
            "mixed_multi_manifest_sha256": (
                "abi/tacker_mixed_render_heads_v2.json"
            ),
            "head_multi_manifest_sha256": (
                "../../tacker_ext/abi/head_linear_v2.json"
            ),
        }
        for capability, manifest_path in manifest_by_capability.items():
            with self.subTest(capability=capability):
                self.assertEqual(
                    capability_sha256(extension, capability),
                    manifest_sha256(manifest_path),
                )

    def test_v2_cpp_validation_is_fail_closed(self):
        binding = source("rasterize_points.cu")
        for required in (
            "head_inputs, head_weights, and head_biases must have equal lengths",
            "mixed ABI v2 requires between 1 and 5 head tasks",
            "worker_groups must be in [1, task_count]",
            "must have shape [N, 128]",
            "must have shape [128, 128]",
            "must have shape [128]",
            "row count exceeds the int32 mixed ABI v2",
            "must be natively 32-byte aligned for WMMA",
            "mixed ABI v2 head outputs must not alias each other",
        ):
            self.assertIn(required, binding)
        self.assertIn("MixedHeadBundleV2 head_bundle = {}", binding)
        self.assertIn("any_head_rows", binding)

    def test_v2_resources_include_register_smem_and_occupancy(self):
        cuda = source("cuda_rasterizer/tacker_mixed.cu")
        header = source("cuda_rasterizer/tacker_mixed.h")
        for required in (
            "cudaFuncGetAttributes",
            "attributes.numRegs",
            "attributes.sharedSizeBytes",
            "cudaOccupancyMaxActiveBlocksPerMultiprocessor",
            "launch_supported",
        ):
            self.assertIn(required, cuda + header)
        self.assertIn("cudaPeekAtLastError", cuda)
        self.assertIn("(void)cudaGetLastError()", cuda)
        self.assertNotIn("cudaDeviceSynchronize", cuda)

    def test_v1_symbol_binding_and_manifest_are_preserved_with_v2(self):
        cuda = source("cuda_rasterizer/tacker_mixed.cu")
        extension = source("ext.cpp")
        manifest = json.loads(source("abi/tacker_mixed_render_head_v1.json"))
        self.assertIn("tacker_mix_render_head_v1", cuda)
        self.assertIn('"rasterize_gaussians_with_head"', extension)
        self.assertEqual(manifest["abi_version"], 1)
        self.assertEqual(
            manifest["global_kernel_symbol"], "tacker_mix_render_head_v1"
        )

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
        self.assertIn("head_linear_v2_device.cuh", setup)
        self.assertIn('os.environ["TORCH_CUDA_ARCH_LIST"] = "8.6"', setup)
        self.assertIn('"-Xptxas=-v"', setup)


if __name__ == "__main__":
    unittest.main()
