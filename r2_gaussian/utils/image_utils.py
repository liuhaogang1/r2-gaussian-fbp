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
import sys
import math
import numpy as np
import torch

sys.path.append("./")
from r2_gaussian.utils.loss_utils import ssim


def mse(img1, img2, mask=None):
    """MSE error

    Args:
        img1 (_type_): [b, c, h, w]
        img2 (_type_): [b, c, h, w]
        mask (_type_, optional): [b, c, h, w]. Defaults to None.

    Returns:
        _type_: _description_
    """
    n_channel = img1.shape[1]
    if mask is not None:
        img1 = img1.flatten(1)
        img2 = img2.flatten(1)

        mask = mask.flatten(1).repeat(1, n_channel)
        mask = torch.where(mask != 0, True, False)

        mse = torch.stack(
            [
                (((img1[i, mask[i]] - img2[i, mask[i]])) ** 2).mean(0, keepdim=True)
                for i in range(img1.shape[0])
            ],
            dim=0,
        )

    else:
        mse = (((img1 - img2)) ** 2).reshape(img1.shape[0], -1).mean(1, keepdim=True)
    return mse


def rmse(img1, img2, mask=None):
    """RMSE error

    Args:
        img1 (_type_): [b, c, h, w]
        img2 (_type_): [b, c, h, w]
        mask (_type_, optional): [b, c, h, w]. Defaults to None.

    Returns:
        _type_: _description_
    """
    mse_out = mse(img1, img2, mask)
    rmse = mse_out**0.5
    return rmse


@torch.no_grad()
def psnr(img1, img2, mask=None, pixel_max=1.0):
    """PSNR

    Args:
        img1 (_type_): [b, c, h, w]
        img2 (_type_): [b, c, h, w]
        mask (_type_, optional): [b, c, h, w]. Defaults to None.

    Returns:
        _type_: _description_
    """
    mse_out = mse(img1, img2, mask)
    psnr_out = 10 * torch.log10(pixel_max**2 / mse_out.float())
    if mask is not None:
        if torch.isinf(psnr_out).any():
            print(mse_out.mean(), psnr_out.mean())
            psnr_out = 10 * torch.log10(pixel_max**2 / mse_out.float())
            psnr_out = psnr_out[~torch.isinf(psnr_out)]

    return psnr_out


def _metric_pair(img1, img2):
    device = img1.device if torch.is_tensor(img1) else (img2.device if torch.is_tensor(img2) else "cpu")
    converted = [torch.as_tensor(np.array(value, copy=True) if isinstance(value, np.ndarray) else value,
                                 device=device, dtype=torch.float32) for value in (img1, img2)]
    img1, img2 = converted
    if img1.shape != img2.shape or img1.ndim != 3 or img1.numel() == 0:
        raise ValueError("Metrics require matching nonempty 3D arrays")
    if not torch.isfinite(img1).all() or not torch.isfinite(img2).all():
        raise ValueError("Metrics require finite inputs")
    return img1, img2


@torch.no_grad()
def metric_vol(img1, img2, metric="psnr", pixel_max=1.0):
    """Metrics for volume. img1 must be GT; all slices contribute."""
    assert metric in ["psnr", "ssim"]
    img1, img2 = _metric_pair(img1, img2)

    if metric == "psnr":
        if pixel_max is None:
            pixel_max = max(float(img1.max().item()), 1e-12)
        mse_out = torch.mean((img1 - img2) ** 2)
        psnr_out = 10 * torch.log10(pixel_max**2 / mse_out.float())
        return psnr_out.item(), None
    elif metric == "ssim":
        ssims = []
        for axis in [0, 1, 2]:
            results = []
            count = 0
            n_slice = img1.shape[axis]
            for i in range(n_slice):
                if axis == 0:
                    slice1 = img1[i, :, :]
                    slice2 = img2[i, :, :]
                elif axis == 1:
                    slice1 = img1[:, i, :]
                    slice2 = img2[:, i, :]
                elif axis == 2:
                    slice1 = img1[:, :, i]
                    slice2 = img2[:, :, i]
                else:
                    raise NotImplementedError
                result = ssim(slice1[None, None], slice2[None, None])
                count += 1
                results.append(result)
            if count == 0:
                raise ValueError("Volume metrics require nonempty spatial dimensions")
            results = torch.stack(results)
            mean_results = torch.sum(results) / count
            ssims.append(mean_results.item())
        return float(np.mean(ssims)), ssims


@torch.no_grad()
def metric_proj(img1, img2, metric="psnr", axis=2, pixel_max=1.0,
                normalization="global_reference", data_range=None):
    """Metrics for projection

    Args:
        img1 (_type_): [x, y, z]
        img2 (_type_): [x, y, z]
        pixel_max (float, optional): _description_. Defaults to 1.0.
    """
    # v3 shares one GT scale and aggregates MSE before computing overall PSNR.
    # Explicit reference/legacy_independent modes retain older comparison APIs.
    assert axis in [0, 1, 2]
    assert metric in ["psnr", "ssim"]
    if normalization not in {"global_reference", "reference", "legacy_independent"}:
        raise ValueError("Unknown projection normalization")
    img1, img2 = _metric_pair(img1, img2)
    if normalization == "global_reference":
        scale, _ = projection_reference_scale(img1, data_range)
        if metric == "psnr":
            errors = [float((right.to(torch.float64) - left.to(torch.float64)).square().mean().item())
                      for left, right in zip(img1.unbind(axis), img2.unbind(axis))]
            results = [_psnr_from_mse(value, scale * pixel_max) for value in errors]
            return _psnr_from_mse(math.fsum(errors) / len(errors), scale * pixel_max), results
        results = [float(ssim((left / scale)[None, None], (right / scale)[None, None]).item())
                   for left, right in zip(img1.unbind(axis), img2.unbind(axis))]
        return float(np.mean(results)), results
    if data_range is not None:
        raise ValueError("data_range is only supported for global_reference metrics")
    n_slice = img1.shape[axis]

    results = []
    count = 0
    for i in range(n_slice):
        if axis == 0:
            slice1 = img1[i, :, :]
            slice2 = img2[i, :, :]
        elif axis == 1:
            slice1 = img1[:, i, :]
            slice2 = img2[:, i, :]
        elif axis == 2:
            slice1 = img1[:, :, i]
            slice2 = img2[:, :, i]
        else:
            raise NotImplementedError
        reference_max = slice1.max()
        if normalization == "legacy_independent":
            if reference_max <= 0:
                results.append(0.0)
                continue
            slice1 = slice1 / reference_max
            predicted_max = slice2.max()
            slice2 = slice2 / predicted_max if predicted_max > 0 else slice2
        else:
            # For blank GT retain physical values and the caller's pixel_max.
            scale = reference_max if reference_max > 0 else 1.0
            slice1, slice2 = slice1 / scale, slice2 / scale
        if metric == "psnr":
            result = psnr(slice1[None, None], slice2[None, None], pixel_max=pixel_max)
        else:
            result = ssim(slice1[None, None], slice2[None, None])
        count += 1
        results.append(float(result.item()))
    if count == 0:
        raise ValueError("No evaluable projections under the selected normalization")
    return float(np.sum(results) / count), results


def projection_reference_scale(reference, data_range=None):
    all_zero = not bool(torch.count_nonzero(reference).item())
    if data_range is not None:
        if isinstance(data_range, bool) or not math.isfinite(float(data_range)) or float(data_range) <= 0:
            raise ValueError("projection data_range must be positive and finite")
        return float(data_range), {"data_range_source": "explicit", "all_zero_reference": all_zero}
    maximum = float(reference.max().item())
    if maximum > 0:
        return maximum, {"data_range_source": "global_gt_max", "all_zero_reference": all_zero}
    if not all_zero:
        raise ValueError("Automatic projection data_range requires positive GT or an all-zero reference")
    return 1.0, {"data_range_source": "all_zero_fallback", "all_zero_reference": True}


def _psnr_from_mse(error, data_range):
    if not math.isfinite(float(data_range)) or data_range <= 0:
        raise ValueError("PSNR data_range must be positive and finite")
    return math.inf if error == 0 else 20 * math.log10(data_range) - 10 * math.log10(error)


@torch.no_grad()
def projection_metric_report(reference, prediction, data_range=None, axis=2):
    reference, prediction = _metric_pair(reference, prediction)
    scale, metadata = projection_reference_scale(reference, data_range)
    psnr_value, psnr_views = metric_proj(reference, prediction, "psnr", axis=axis, data_range=scale)
    ssim_value, ssim_views = metric_proj(reference, prediction, "ssim", axis=axis, data_range=scale)
    perfect = [value == math.inf for value in psnr_views]
    return {"psnr_2d": psnr_value, "ssim_2d": ssim_value,
            "psnr_2d_projs": psnr_views, "ssim_2d_projs": ssim_views,
            "psnr_2d_status": "perfect" if psnr_value == math.inf else "finite",
            "psnr_2d_projs_status": ["perfect" if value else "finite" for value in perfect],
            "perfect_view_count": sum(perfect), "projection_data_range": scale,
            "projection_metric_version": "global_reference_v3", **metadata,
            **projection_errors(reference, prediction)}


@torch.no_grad()
def projection_errors(reference, prediction):
    """Amplitude-sensitive errors in the original projection units."""
    absolute, squared = [], []
    for left, right in zip(reference.unbind(-1), prediction.unbind(-1)):
        delta = right.to(torch.float64) - left.to(torch.float64)
        absolute.append(float(delta.abs().sum().item()))
        squared.append(float(delta.square().sum().item()))
    count = reference.numel()
    return {"mae_2d": math.fsum(absolute) / count,
            "rmse_2d": math.sqrt(math.fsum(squared) / count)}
