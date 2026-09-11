"""GPU boundary tests for the inference-only mixed entry point."""

import unittest

try:
    import torch
    from diff_gaussian_rasterization import _C
except (ImportError, OSError):
    torch = None
    _C = None


def _mixed_available():
    return (
        torch is not None
        and torch.cuda.is_available()
        and _C is not None
        and hasattr(_C, "rasterize_gaussians_with_head")
    )


def _mixed_v2_available():
    return (
        torch is not None
        and torch.cuda.is_available()
        and _C is not None
        and hasattr(_C, "rasterize_gaussians_with_heads")
        and hasattr(_C, "tacker_resource_requirements")
    )


@unittest.skipUnless(_mixed_available(), "mixed CUDA extension required")
class MixedCudaBoundaryTest(unittest.TestCase):
    @staticmethod
    def _args(head_input, use_sh=False):
        device = head_input.device
        float_options = {"device": device, "dtype": torch.float32}
        means = torch.empty((0, 3), **float_options)
        colors = torch.empty((0,), **float_options)
        sh = torch.empty((0, 1, 3), **float_options)
        if not use_sh:
            colors = torch.empty((0, 3), **float_options)
            sh = torch.empty((0,), **float_options)
        return (
            torch.zeros((3,), **float_options),
            means,
            colors,
            torch.empty((0, 1), **float_options),
            torch.empty((0,), **float_options),
            torch.empty((0,), **float_options),
            1.0,
            torch.empty((0, 6), **float_options),
            torch.eye(4, **float_options),
            torch.eye(4, **float_options),
            1.0,
            1.0,
            16,
            16,
            sh,
            0,
            torch.zeros((3,), **float_options),
            False,
            False,
            head_input,
            torch.empty((128, 128), device=device, dtype=torch.float16),
            torch.empty((128,), **float_options),
            1,
        )

    def test_zero_row_ranked_alternatives_are_not_placeholders(self):
        head_input = torch.empty((0, 128), device="cuda", dtype=torch.float16)
        with torch.no_grad():
            colors_result = _C.rasterize_gaussians_with_head(
                *self._args(head_input, use_sh=False)
            )
            sh_result = _C.rasterize_gaussians_with_head(
                *self._args(head_input, use_sh=True)
            )
        self.assertEqual(tuple(colors_result[-1].shape), (0, 128))
        self.assertEqual(tuple(sh_result[-1].shape), (0, 128))

    def test_misaligned_mixed_head_input_is_rejected(self):
        storage = torch.empty(128 + 1, device="cuda", dtype=torch.float16)
        head_input = storage[1:].view(1, 128)
        self.assertTrue(head_input.is_contiguous())
        with torch.no_grad():
            with self.assertRaisesRegex(RuntimeError, "head_input.*32-byte"):
                _C.rasterize_gaussians_with_head(*self._args(head_input))

    def test_misaligned_mixed_head_weight_is_rejected(self):
        head_input = torch.empty((1, 128), device="cuda", dtype=torch.float16)
        storage = torch.empty(
            128 * 128 + 1, device="cuda", dtype=torch.float16
        )
        weight = storage[1:].view(128, 128)
        args = list(self._args(head_input))
        args[20] = weight
        self.assertTrue(weight.is_contiguous())
        with torch.no_grad():
            with self.assertRaisesRegex(RuntimeError, "head_weight.*32-byte"):
                _C.rasterize_gaussians_with_head(*args)

    def test_cpu_placeholder_is_rejected(self):
        head_input = torch.empty((0, 128), device="cuda", dtype=torch.float16)
        args = list(self._args(head_input))
        args[4] = torch.empty((0,), dtype=torch.float32)
        with torch.no_grad():
            with self.assertRaisesRegex(RuntimeError, "scales must be a CUDA"):
                _C.rasterize_gaussians_with_head(*args)

    def test_zero_row_alternatives_remain_strictly_exclusive(self):
        head_input = torch.empty((0, 128), device="cuda", dtype=torch.float16)
        args = list(self._args(head_input))
        args[2] = torch.empty((0,), device="cuda", dtype=torch.float32)
        with torch.no_grad():
            with self.assertRaisesRegex(RuntimeError, "exactly one of sh"):
                _C.rasterize_gaussians_with_head(*args)

        args = list(self._args(head_input))
        args[4] = torch.empty((0, 3), device="cuda", dtype=torch.float32)
        args[5] = torch.empty((0, 4), device="cuda", dtype=torch.float32)
        with torch.no_grad():
            with self.assertRaisesRegex(RuntimeError, "exactly one of cov3D"):
                _C.rasterize_gaussians_with_head(*args)


@unittest.skipUnless(_mixed_v2_available(), "mixed CUDA ABI v2 required")
class MixedV2CudaBoundaryTest(unittest.TestCase):
    @staticmethod
    def _args(head_inputs, head_weights=None, head_biases=None, worker_groups=1):
        device = head_inputs[0].device
        float_options = {"device": device, "dtype": torch.float32}
        if head_weights is None:
            head_weights = [
                torch.randn((128, 128), device=device, dtype=torch.float16)
                * 0.05
                for _ in head_inputs
            ]
        if head_biases is None:
            head_biases = [
                torch.randn((128,), **float_options) * 0.05
                for _ in head_inputs
            ]
        return (
            torch.zeros((3,), **float_options),
            torch.empty((0, 3), **float_options),
            torch.empty((0, 3), **float_options),
            torch.empty((0, 1), **float_options),
            torch.empty((0,), **float_options),
            torch.empty((0,), **float_options),
            1.0,
            torch.empty((0, 6), **float_options),
            torch.eye(4, **float_options),
            torch.eye(4, **float_options),
            1.0,
            1.0,
            16,
            16,
            torch.empty((0,), **float_options),
            0,
            torch.zeros((3,), **float_options),
            False,
            False,
            list(head_inputs),
            list(head_weights),
            list(head_biases),
            worker_groups,
            1,
        )

    def test_one_to_five_heads_all_worker_groups_match_fp32_reference(self):
        torch.manual_seed(20260910)
        device = torch.device("cuda")
        rows = [0, 1, 17, 31, 33]
        inputs = [
            torch.randn((row_count, 128), device=device, dtype=torch.float16)
            * 0.05
            for row_count in rows
        ]
        weights = [
            torch.randn((128, 128), device=device, dtype=torch.float16) * 0.05
            for _ in rows
        ]
        biases = [
            torch.randn((128,), device=device, dtype=torch.float32) * 0.05
            for _ in rows
        ]

        with torch.no_grad():
            for task_count in range(1, 6):
                for worker_groups in range(1, task_count + 1):
                    result = _C.rasterize_gaussians_with_heads(
                        *self._args(
                            inputs[:task_count],
                            weights[:task_count],
                            biases[:task_count],
                            worker_groups,
                        )
                    )
                    outputs = result[-1]
                    self.assertEqual(len(outputs), task_count)
                    for task_index, output in enumerate(outputs):
                        reference = torch.nn.functional.linear(
                            inputs[task_index].float(),
                            weights[task_index].float(),
                            biases[task_index],
                        )
                        torch.testing.assert_close(
                            output, reference, atol=2e-3, rtol=2e-3
                        )

    def test_dual_head_c2_serial_and_parallel_groups_agree(self):
        device = torch.device("cuda")
        inputs = [
            torch.randn((19, 128), device=device, dtype=torch.float16) * 0.05,
            torch.randn((7, 128), device=device, dtype=torch.float16) * 0.05,
        ]
        weights = [
            torch.randn((128, 128), device=device, dtype=torch.float16) * 0.05,
            torch.randn((128, 128), device=device, dtype=torch.float16) * 0.05,
        ]
        biases = [
            torch.randn((128,), device=device, dtype=torch.float32) * 0.05,
            torch.randn((128,), device=device, dtype=torch.float32) * 0.05,
        ]
        with torch.no_grad():
            serial = _C.rasterize_gaussians_with_heads(
                *self._args(inputs, weights, biases, 1)
            )[-1]
            parallel = _C.rasterize_gaussians_with_heads(
                *self._args(inputs, weights, biases, 2)
            )[-1]
        for serial_output, parallel_output in zip(serial, parallel):
            torch.testing.assert_close(
                serial_output, parallel_output, atol=2e-3, rtol=2e-3
            )

    def test_shared_read_only_input_alias_is_allowed_outputs_are_distinct(self):
        device = torch.device("cuda")
        shared_input = torch.randn(
            (17, 128), device=device, dtype=torch.float16
        ) * 0.05
        inputs = [shared_input] * 5
        with torch.no_grad():
            outputs = _C.rasterize_gaussians_with_heads(
                *self._args(inputs, worker_groups=5)
            )[-1]
        self.assertEqual(len({output.data_ptr() for output in outputs}), 5)

    def test_invalid_bundle_is_rejected_and_next_valid_launch_succeeds(self):
        device = torch.device("cuda")
        head_input = torch.empty((1, 128), device=device, dtype=torch.float16)
        args = list(self._args([head_input]))
        args[21] = []
        with torch.no_grad():
            with self.assertRaisesRegex(RuntimeError, "equal lengths"):
                _C.rasterize_gaussians_with_heads(*args)

        args = list(self._args([head_input]))
        args[22] = 2
        with torch.no_grad():
            with self.assertRaisesRegex(RuntimeError, "worker_groups"):
                _C.rasterize_gaussians_with_heads(*args)
            valid = _C.rasterize_gaussians_with_heads(
                *self._args([head_input])
            )
        self.assertEqual(tuple(valid[-1][0].shape), (1, 128))

    def test_misaligned_task_and_wrong_dtype_are_rejected(self):
        device = torch.device("cuda")
        storage = torch.empty(128 + 1, device=device, dtype=torch.float16)
        misaligned = storage[1:].view(1, 128)
        with torch.no_grad():
            with self.assertRaisesRegex(RuntimeError, r"head_inputs\[0\].*32-byte"):
                _C.rasterize_gaussians_with_heads(*self._args([misaligned]))

        wrong_dtype = torch.empty((1, 128), device=device, dtype=torch.float32)
        with torch.no_grad():
            with self.assertRaisesRegex(RuntimeError, r"head_inputs\[0\].*float16"):
                _C.rasterize_gaussians_with_heads(*self._args([wrong_dtype]))

    def test_resource_query_covers_all_dynamic_cta_sizes(self):
        expected_threads = {1: 384, 2: 512, 3: 640, 4: 768, 5: 896}
        for worker_groups, threads in expected_threads.items():
            resources = dict(_C.tacker_resource_requirements(2, worker_groups))
            self.assertEqual(resources["physical_threads"], threads)
            self.assertGreater(resources["registers_per_thread"], 0)
            self.assertGreaterEqual(resources["static_shared_bytes"], 0)
            self.assertGreater(resources["kernel_max_threads_per_block"], 0)
            self.assertGreater(resources["active_blocks_per_multiprocessor"], 0)
            self.assertTrue(resources["launch_supported"])

        legacy = dict(_C.tacker_resource_requirements(1, 1))
        self.assertEqual(legacy["physical_threads"], 384)
        # pybind11 maps the launcher's std::invalid_argument to ValueError.
        with self.assertRaisesRegex(ValueError, "worker_groups"):
            _C.tacker_resource_requirements(2, 6)

if __name__ == "__main__":
    unittest.main()
