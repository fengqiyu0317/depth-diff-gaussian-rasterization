#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from typing import NamedTuple
import torch.nn as nn
import torch
from . import _C

def cpu_deep_copy_tuple(input_tuple):
    copied_tensors = [item.cpu().clone() if isinstance(item, torch.Tensor) else item for item in input_tuple]
    return tuple(copied_tensors)


def tacker_capabilities():
    """Return the compiled Raster+head ABI, or an explicit legacy marker."""

    if not hasattr(_C, "tacker_capabilities"):
        return {
            "stream_aware": False,
            "mixed_render_head_abi": 0,
            "reason": "installed extension predates the Tacker mixed ABI",
        }
    return dict(_C.tacker_capabilities())


def tacker_resource_requirements(
    abi_version=2, worker_groups=1, family=None
):
    """Query launch resources/occupancy for one compiled mixed variant."""

    for name, value in (
        ("abi_version", abi_version),
        ("worker_groups", worker_groups),
    ):
        if not isinstance(value, int) or isinstance(value, bool):
            raise TypeError(f"{name} must be an int")
    if family is not None and not isinstance(family, str):
        raise TypeError("family must be a str or None")
    if not hasattr(_C, "tacker_resource_requirements"):
        raise RuntimeError(
            "installed diff_gaussian_rasterization extension does not "
            "provide the Tacker resource query; rebuild the extension"
        )
    if family is None:
        return dict(
            _C.tacker_resource_requirements(abi_version, worker_groups)
        )
    return dict(
        _C.tacker_resource_requirements(
            abi_version, worker_groups, family
        )
    )


def tacker_variant_resources(
    worker_groups, family=None
):
    """Return the schema-v2 runtime's normalized resource vocabulary."""

    family_to_abi = {
        "first_linear_heads_v2": 2,
        "packed_first_linear_v3": 3,
        "whole_heads_v4": 4,
    }
    if family is None:
        raw = tacker_resource_requirements(
            abi_version=2, worker_groups=worker_groups
        )
        family = "first_linear_heads_v2"
    elif not isinstance(family, str):
        raise TypeError("family must be a str or None")
    else:
        try:
            abi_version = family_to_abi[family]
        except KeyError:
            raise ValueError("unsupported Tacker mixed backend family")
        raw = tacker_resource_requirements(
            abi_version=abi_version,
            worker_groups=worker_groups,
            family=family,
        )
    normalized = dict(raw)
    normalized.update(
        {
            "block_threads": raw["physical_threads"],
            "registers_per_thread": raw["registers_per_thread"],
            "static_shared_memory_bytes": raw["static_shared_bytes"],
            "max_threads_per_block": raw["kernel_max_threads_per_block"],
            "active_blocks_per_sm": raw[
                "active_blocks_per_multiprocessor"
            ],
        }
    )
    return normalized

def rasterize_gaussians(
    means3D,
    means2D,
    sh,
    colors_precomp,
    opacities,
    scales,
    rotations,
    cov3Ds_precomp,
    raster_settings,
):
    return _RasterizeGaussians.apply(
        means3D,
        means2D,
        sh,
        colors_precomp,
        opacities,
        scales,
        rotations,
        cov3Ds_precomp,
        raster_settings,
    )

class _RasterizeGaussians(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        means3D,
        means2D,
        sh,
        colors_precomp,
        opacities,
        scales,
        rotations,
        cov3Ds_precomp,
        raster_settings,
    ):

        # Restructure arguments the way that the C++ lib expects them
        args = (
            raster_settings.bg, 
            means3D,
            colors_precomp,
            opacities,
            scales,
            rotations,
            raster_settings.scale_modifier,
            cov3Ds_precomp,
            raster_settings.viewmatrix,
            raster_settings.projmatrix,
            raster_settings.tanfovx,
            raster_settings.tanfovy,
            raster_settings.image_height,
            raster_settings.image_width,
            sh,
            raster_settings.sh_degree,
            raster_settings.campos,
            raster_settings.prefiltered,
            raster_settings.debug
        )

        # Invoke C++/CUDA rasterizer
        if raster_settings.debug:
            cpu_args = cpu_deep_copy_tuple(args) # Copy them before they can be corrupted
            try:
                num_rendered, color, depth, radii, geomBuffer, binningBuffer, imgBuffer = _C.rasterize_gaussians(*args)
            except Exception as ex:
                torch.save(cpu_args, "snapshot_fw.dump")
                print("\nAn error occured in forward. Please forward snapshot_fw.dump for debugging.")
                raise ex
        else:
            num_rendered, color, depth, radii, geomBuffer, binningBuffer, imgBuffer = _C.rasterize_gaussians(*args)

        # Keep relevant tensors for backward
        ctx.raster_settings = raster_settings
        ctx.num_rendered = num_rendered
        ctx.save_for_backward(colors_precomp, means3D, scales, rotations, cov3Ds_precomp, radii, sh, geomBuffer, binningBuffer, imgBuffer)
        return color, radii, depth

    @staticmethod
    def backward(ctx, grad_out_color, grad_radii, grad_depth):

        # Restore necessary values from context
        num_rendered = ctx.num_rendered
        raster_settings = ctx.raster_settings
        colors_precomp, means3D, scales, rotations, cov3Ds_precomp, radii, sh, geomBuffer, binningBuffer, imgBuffer = ctx.saved_tensors

        # Restructure args as C++ method expects them
        args = (raster_settings.bg,
                means3D, 
                radii, 
                colors_precomp, 
                scales, 
                rotations, 
                raster_settings.scale_modifier, 
                cov3Ds_precomp, 
                raster_settings.viewmatrix, 
                raster_settings.projmatrix, 
                raster_settings.tanfovx, 
                raster_settings.tanfovy, 
                grad_out_color,
                grad_depth,
                sh, 
                raster_settings.sh_degree, 
                raster_settings.campos,
                geomBuffer,
                num_rendered,
                binningBuffer,
                imgBuffer,
                raster_settings.debug)

        # Compute gradients for relevant tensors by invoking backward method
        if raster_settings.debug:
            cpu_args = cpu_deep_copy_tuple(args) # Copy them before they can be corrupted
            try:
                grad_means2D, grad_colors_precomp, grad_opacities, grad_means3D, grad_cov3Ds_precomp, grad_sh, grad_scales, grad_rotations = _C.rasterize_gaussians_backward(*args)
            except Exception as ex:
                torch.save(cpu_args, "snapshot_bw.dump")
                print("\nAn error occured in backward. Writing snapshot_bw.dump for debugging.\n")
                raise ex
        else:
             grad_means2D, grad_colors_precomp, grad_opacities, grad_means3D, grad_cov3Ds_precomp, grad_sh, grad_scales, grad_rotations = _C.rasterize_gaussians_backward(*args)

        grads = (
            grad_means3D,
            grad_means2D,
            grad_sh,
            grad_colors_precomp,
            grad_opacities,
            grad_scales,
            grad_rotations,
            grad_cov3Ds_precomp,
            None,
        )

        return grads

class GaussianRasterizationSettings(NamedTuple):
    image_height: int
    image_width: int 
    tanfovx : float
    tanfovy : float
    bg : torch.Tensor
    scale_modifier : float
    viewmatrix : torch.Tensor
    projmatrix : torch.Tensor
    sh_degree : int
    campos : torch.Tensor
    prefiltered : bool
    debug : bool

class GaussianRasterizer(nn.Module):
    def __init__(self, raster_settings):
        super().__init__()
        self.raster_settings = raster_settings

    def markVisible(self, positions):
        # Mark visible points (based on frustum culling for camera) with a boolean 
        with torch.no_grad():
            raster_settings = self.raster_settings
            visible = _C.mark_visible(
                positions,
                raster_settings.viewmatrix,
                raster_settings.projmatrix)
            
        return visible

    def forward(self, means3D, means2D, opacities, shs = None, colors_precomp = None, scales = None, rotations = None, cov3D_precomp = None):
        
        raster_settings = self.raster_settings

        if (shs is None and colors_precomp is None) or (shs is not None and colors_precomp is not None):
            raise Exception('Please provide excatly one of either SHs or precomputed colors!')
        
        if ((scales is None or rotations is None) and cov3D_precomp is None) or ((scales is not None or rotations is not None) and cov3D_precomp is not None):
            raise Exception('Please provide exactly one of either scale/rotation pair or precomputed 3D covariance!')
        
        if shs is None:
            shs = torch.Tensor([])
        if colors_precomp is None:
            colors_precomp = torch.Tensor([])

        if scales is None:
            scales = torch.Tensor([])
        if rotations is None:
            rotations = torch.Tensor([])
        if cov3D_precomp is None:
            cov3D_precomp = torch.Tensor([])

        # Invoke C++/CUDA rasterization routine
        return rasterize_gaussians(
            means3D,
            means2D,
            shs,
            colors_precomp,
            opacities,
            scales, 
            rotations,
            cov3D_precomp,
            raster_settings, 
        )

    def forward_with_head(
        self,
        means3D,
        means2D,
        opacities,
        head_input,
        head_weight,
        head_bias,
        shs=None,
        colors_precomp=None,
        scales=None,
        rotations=None,
        cov3D_precomp=None,
        persistent_blocks=0,
    ):
        """Inference-only physical fusion of Raster and one 128x128 head.

        ``head_input`` and ``head_weight`` are contiguous CUDA FP16 tensors;
        ``head_bias`` is contiguous CUDA FP32.  The returned head tensor is
        FP32 and implements ``head_input @ head_weight.T + head_bias``.
        ``means2D`` is retained for call-site compatibility but is not part of
        the inference-only C++ ABI.
        """

        if torch.is_grad_enabled():
            raise RuntimeError(
                "GaussianRasterizer.forward_with_head is inference-only; "
                "call it under torch.no_grad()"
            )
        if not hasattr(_C, "rasterize_gaussians_with_head"):
            raise RuntimeError(
                "installed diff_gaussian_rasterization extension does not "
                "provide the Tacker mixed ABI; rebuild the extension"
            )
        if not isinstance(persistent_blocks, int) or isinstance(
            persistent_blocks, bool
        ):
            raise TypeError("persistent_blocks must be an int")
        if persistent_blocks < 0:
            raise ValueError("persistent_blocks must be >= 0")

        if (shs is None and colors_precomp is None) or (
            shs is not None and colors_precomp is not None
        ):
            raise Exception(
                "Please provide exactly one of either SHs or precomputed colors!"
            )
        if (
            (scales is None or rotations is None) and cov3D_precomp is None
        ) or (
            (scales is not None or rotations is not None)
            and cov3D_precomp is not None
        ):
            raise Exception(
                "Please provide exactly one of either scale/rotation pair or "
                "precomputed 3D covariance!"
            )

        # Unlike the legacy autograd path, keep placeholders on the raster
        # tensor's device.  They remain empty and are never dereferenced.
        empty = means3D.new_empty((0,))
        if shs is None:
            shs = empty
        if colors_precomp is None:
            colors_precomp = empty
        if scales is None:
            scales = empty
        if rotations is None:
            rotations = empty
        if cov3D_precomp is None:
            cov3D_precomp = empty

        raster_settings = self.raster_settings
        args = (
            raster_settings.bg,
            means3D,
            colors_precomp,
            opacities,
            scales,
            rotations,
            raster_settings.scale_modifier,
            cov3D_precomp,
            raster_settings.viewmatrix,
            raster_settings.projmatrix,
            raster_settings.tanfovx,
            raster_settings.tanfovy,
            raster_settings.image_height,
            raster_settings.image_width,
            shs,
            raster_settings.sh_degree,
            raster_settings.campos,
            raster_settings.prefiltered,
            raster_settings.debug,
            head_input,
            head_weight,
            head_bias,
            persistent_blocks,
        )

        (
            _num_rendered,
            color,
            depth,
            radii,
            _geom_buffer,
            _binning_buffer,
            _image_buffer,
            head_output,
        ) = _C.rasterize_gaussians_with_head(*args)
        return color, radii, depth, head_output

    def forward_with_heads(
        self,
        means3D,
        means2D,
        opacities,
        head_inputs,
        head_weights,
        head_biases,
        worker_groups=1,
        persistent_blocks=0,
        shs=None,
        colors_precomp=None,
        scales=None,
        rotations=None,
        cov3D_precomp=None,
    ):
        """Fuse Raster with 1--5 first-linear deformation-head tasks.

        Each task computes ``input @ weight.T + bias`` and returns a distinct
        FP32 ``[N, 128]`` tensor.  Read-only operands may alias across tasks,
        which permits all five heads to share one hidden activation.  Worker
        group ``g`` evaluates task indices ``g, g + worker_groups, ...``.
        """

        if torch.is_grad_enabled():
            raise RuntimeError(
                "GaussianRasterizer.forward_with_heads is inference-only; "
                "call it under torch.no_grad()"
            )
        if not hasattr(_C, "rasterize_gaussians_with_heads"):
            raise RuntimeError(
                "installed diff_gaussian_rasterization extension does not "
                "provide the Tacker mixed ABI v2; rebuild the extension"
            )
        for name, values in (
            ("head_inputs", head_inputs),
            ("head_weights", head_weights),
            ("head_biases", head_biases),
        ):
            if not isinstance(values, (list, tuple)):
                raise TypeError(f"{name} must be a list or tuple of tensors")
        head_inputs = tuple(head_inputs)
        head_weights = tuple(head_weights)
        head_biases = tuple(head_biases)
        if not isinstance(worker_groups, int) or isinstance(worker_groups, bool):
            raise TypeError("worker_groups must be an int")
        if not isinstance(persistent_blocks, int) or isinstance(
            persistent_blocks, bool
        ):
            raise TypeError("persistent_blocks must be an int")
        if worker_groups < 1:
            raise ValueError("worker_groups must be >= 1")
        if persistent_blocks < 0:
            raise ValueError("persistent_blocks must be >= 0")

        if (shs is None and colors_precomp is None) or (
            shs is not None and colors_precomp is not None
        ):
            raise Exception(
                "Please provide exactly one of either SHs or precomputed colors!"
            )
        if (
            (scales is None or rotations is None) and cov3D_precomp is None
        ) or (
            (scales is not None or rotations is not None)
            and cov3D_precomp is not None
        ):
            raise Exception(
                "Please provide exactly one of either scale/rotation pair or "
                "precomputed 3D covariance!"
            )

        empty = means3D.new_empty((0,))
        if shs is None:
            shs = empty
        if colors_precomp is None:
            colors_precomp = empty
        if scales is None:
            scales = empty
        if rotations is None:
            rotations = empty
        if cov3D_precomp is None:
            cov3D_precomp = empty

        raster_settings = self.raster_settings
        args = (
            raster_settings.bg,
            means3D,
            colors_precomp,
            opacities,
            scales,
            rotations,
            raster_settings.scale_modifier,
            cov3D_precomp,
            raster_settings.viewmatrix,
            raster_settings.projmatrix,
            raster_settings.tanfovx,
            raster_settings.tanfovy,
            raster_settings.image_height,
            raster_settings.image_width,
            shs,
            raster_settings.sh_degree,
            raster_settings.campos,
            raster_settings.prefiltered,
            raster_settings.debug,
            head_inputs,
            head_weights,
            head_biases,
            worker_groups,
            persistent_blocks,
        )

        (
            _num_rendered,
            color,
            depth,
            radii,
            _geom_buffer,
            _binning_buffer,
            _image_buffer,
            head_outputs,
        ) = _C.rasterize_gaussians_with_heads(*args)
        return color, radii, depth, tuple(head_outputs)

    def forward_with_packed_heads(
        self,
        means3D,
        means2D,
        opacities,
        head_input,
        packed_head_weights,
        packed_head_biases,
        worker_groups=1,
        persistent_blocks=0,
        shs=None,
        colors_precomp=None,
        scales=None,
        rotations=None,
        cov3D_precomp=None,
    ):
        """Fuse Raster with C3 packed shared-input first-linear heads."""

        if torch.is_grad_enabled():
            raise RuntimeError(
                "GaussianRasterizer.forward_with_packed_heads is "
                "inference-only; call it under torch.no_grad()"
            )
        if not hasattr(_C, "rasterize_gaussians_with_packed_heads"):
            raise RuntimeError(
                "installed diff_gaussian_rasterization extension does not "
                "provide the Tacker packed mixed ABI v3; rebuild the extension"
            )
        for name, value in (
            ("worker_groups", worker_groups),
            ("persistent_blocks", persistent_blocks),
        ):
            if not isinstance(value, int) or isinstance(value, bool):
                raise TypeError(f"{name} must be an int")
        if worker_groups < 1:
            raise ValueError("worker_groups must be >= 1")
        if persistent_blocks < 0:
            raise ValueError("persistent_blocks must be >= 0")

        if (shs is None and colors_precomp is None) or (
            shs is not None and colors_precomp is not None
        ):
            raise Exception(
                "Please provide exactly one of either SHs or precomputed colors!"
            )
        if (
            (scales is None or rotations is None) and cov3D_precomp is None
        ) or (
            (scales is not None or rotations is not None)
            and cov3D_precomp is not None
        ):
            raise Exception(
                "Please provide exactly one of either scale/rotation pair or "
                "precomputed 3D covariance!"
            )

        empty = means3D.new_empty((0,))
        if shs is None:
            shs = empty
        if colors_precomp is None:
            colors_precomp = empty
        if scales is None:
            scales = empty
        if rotations is None:
            rotations = empty
        if cov3D_precomp is None:
            cov3D_precomp = empty

        raster_settings = self.raster_settings
        args = (
            raster_settings.bg,
            means3D,
            colors_precomp,
            opacities,
            scales,
            rotations,
            raster_settings.scale_modifier,
            cov3D_precomp,
            raster_settings.viewmatrix,
            raster_settings.projmatrix,
            raster_settings.tanfovx,
            raster_settings.tanfovy,
            raster_settings.image_height,
            raster_settings.image_width,
            shs,
            raster_settings.sh_degree,
            raster_settings.campos,
            raster_settings.prefiltered,
            raster_settings.debug,
            head_input,
            packed_head_weights,
            packed_head_biases,
            worker_groups,
            persistent_blocks,
        )
        (
            _num_rendered,
            color,
            depth,
            radii,
            _geom_buffer,
            _binning_buffer,
            _image_buffer,
            packed_head_outputs,
        ) = _C.rasterize_gaussians_with_packed_heads(*args)
        return color, radii, depth, packed_head_outputs

    def forward_with_whole_heads(
        self,
        means3D,
        means2D,
        opacities,
        head_inputs,
        first_weights,
        first_biases,
        tail_weights,
        tail_biases,
        output_widths,
        worker_groups=1,
        persistent_blocks=0,
        shs=None,
        colors_precomp=None,
        scales=None,
        rotations=None,
        cov3D_precomp=None,
    ):
        """Fuse Raster with one to five complete C4 deformation heads."""

        if torch.is_grad_enabled():
            raise RuntimeError(
                "GaussianRasterizer.forward_with_whole_heads is "
                "inference-only; call it under torch.no_grad()"
            )
        if not hasattr(_C, "rasterize_gaussians_with_whole_heads"):
            raise RuntimeError(
                "installed diff_gaussian_rasterization extension does not "
                "provide the Tacker whole-head mixed ABI v4; rebuild the extension"
            )
        sequences = (
            ("head_inputs", head_inputs),
            ("first_weights", first_weights),
            ("first_biases", first_biases),
            ("tail_weights", tail_weights),
            ("tail_biases", tail_biases),
            ("output_widths", output_widths),
        )
        for name, values in sequences:
            if not isinstance(values, (list, tuple)):
                raise TypeError(f"{name} must be a list or tuple")
        (
            head_inputs,
            first_weights,
            first_biases,
            tail_weights,
            tail_biases,
            output_widths,
        ) = tuple(tuple(values) for _name, values in sequences)
        task_count = len(head_inputs)
        if task_count < 1 or task_count > 5:
            raise ValueError("whole-head task count must be in [1, 5]")
        if any(
            len(values) != task_count
            for values in (
                first_weights,
                first_biases,
                tail_weights,
                tail_biases,
                output_widths,
            )
        ):
            raise ValueError(
                "whole-head inputs, parameters, and output_widths must have "
                "equal lengths"
            )
        for index, width in enumerate(output_widths):
            if not isinstance(width, int) or isinstance(width, bool):
                raise TypeError(f"output_widths[{index}] must be an int")
            if width < 1 or width > 128:
                raise ValueError(f"output_widths[{index}] must be in [1, 128]")
            if not isinstance(tail_weights[index], torch.Tensor) or not isinstance(
                tail_biases[index], torch.Tensor
            ):
                raise TypeError("tail_weights and tail_biases must contain tensors")
            if tail_weights[index].dim() != 2 or tail_weights[index].size(0) != width:
                raise ValueError(
                    f"output_widths[{index}] must match tail_weights[{index}]"
                )
            if tail_biases[index].dim() != 1 or tail_biases[index].size(0) != width:
                raise ValueError(
                    f"output_widths[{index}] must match tail_biases[{index}]"
                )
        for name, value in (
            ("worker_groups", worker_groups),
            ("persistent_blocks", persistent_blocks),
        ):
            if not isinstance(value, int) or isinstance(value, bool):
                raise TypeError(f"{name} must be an int")
        if worker_groups < 1 or worker_groups > task_count:
            raise ValueError("worker_groups must be in [1, task_count]")
        if persistent_blocks < 0:
            raise ValueError("persistent_blocks must be >= 0")

        if (shs is None and colors_precomp is None) or (
            shs is not None and colors_precomp is not None
        ):
            raise Exception(
                "Please provide exactly one of either SHs or precomputed colors!"
            )
        if (
            (scales is None or rotations is None) and cov3D_precomp is None
        ) or (
            (scales is not None or rotations is not None)
            and cov3D_precomp is not None
        ):
            raise Exception(
                "Please provide exactly one of either scale/rotation pair or "
                "precomputed 3D covariance!"
            )

        empty = means3D.new_empty((0,))
        if shs is None:
            shs = empty
        if colors_precomp is None:
            colors_precomp = empty
        if scales is None:
            scales = empty
        if rotations is None:
            rotations = empty
        if cov3D_precomp is None:
            cov3D_precomp = empty

        raster_settings = self.raster_settings
        args = (
            raster_settings.bg,
            means3D,
            colors_precomp,
            opacities,
            scales,
            rotations,
            raster_settings.scale_modifier,
            cov3D_precomp,
            raster_settings.viewmatrix,
            raster_settings.projmatrix,
            raster_settings.tanfovx,
            raster_settings.tanfovy,
            raster_settings.image_height,
            raster_settings.image_width,
            shs,
            raster_settings.sh_degree,
            raster_settings.campos,
            raster_settings.prefiltered,
            raster_settings.debug,
            head_inputs,
            first_weights,
            first_biases,
            tail_weights,
            tail_biases,
            worker_groups,
            persistent_blocks,
        )
        (
            _num_rendered,
            color,
            depth,
            radii,
            _geom_buffer,
            _binning_buffer,
            _image_buffer,
            head_outputs,
        ) = _C.rasterize_gaussians_with_whole_heads(*args)
        return color, radii, depth, tuple(head_outputs)

    def forward_with_whole_head(
        self,
        means3D,
        means2D,
        opacities,
        head_input,
        first_weight,
        first_bias,
        tail_weight,
        tail_bias,
        persistent_blocks=0,
        shs=None,
        colors_precomp=None,
        scales=None,
        rotations=None,
        cov3D_precomp=None,
    ):
        """Strict one-task convenience method for the C4 physical backend."""

        result = self.forward_with_whole_heads(
            means3D=means3D,
            means2D=means2D,
            opacities=opacities,
            head_inputs=(head_input,),
            first_weights=(first_weight,),
            first_biases=(first_bias,),
            tail_weights=(tail_weight,),
            tail_biases=(tail_bias,),
            output_widths=(tail_weight.size(0),),
            worker_groups=1,
            persistent_blocks=persistent_blocks,
            shs=shs,
            colors_precomp=colors_precomp,
            scales=scales,
            rotations=rotations,
            cov3D_precomp=cov3D_precomp,
        )
        color, radii, depth, head_outputs = result
        return color, radii, depth, head_outputs[0]
