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


if __name__ == "__main__":
    unittest.main()
