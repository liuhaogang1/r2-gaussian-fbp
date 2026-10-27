"""Validate actual CUDA rendering against analytic physical Gaussian integrals.

Run from the repository root; --pose also checks the differentiable backend.
Does not load experimental data, update a model, or start training.
"""

import argparse
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def check_render(torch, render, camera, model, pipe, cfg, center, sigma, shift):
    image = render(camera, model, pipe)["render"][0]
    height, width = cfg["nDetector"]
    counts = np.array([width, height])
    pitch = np.array(cfg["dDetector"])[::-1]
    angle = camera.angle
    # Saved projection rows grow downwards; physical +Z maps towards the top.
    axes = np.array([[-np.sin(angle), np.cos(angle), 0], [0, 0, -1]])
    mean = (axes @ center - cfg["offDetector"]) / pitch + (counts - 1) / 2 + shift
    covariance_world = np.diag(sigma**2)
    covariance_camera = axes @ covariance_world @ axes.T
    covariance_pixels = covariance_camera / pitch[:, None] / pitch[None, :]
    uv = np.stack(np.meshgrid(np.arange(width), np.arange(height)), axis=-1)
    delta = uv - mean
    power = -0.5 * np.einsum("...i,ij,...j->...", delta, np.linalg.inv(covariance_pixels), delta)
    amplitude = np.sqrt(
        2 * np.pi * np.linalg.det(covariance_world) / np.linalg.det(covariance_camera)
    )
    expected = amplitude * np.exp(power)
    actual = image.detach().cpu().numpy()
    # CUDA skips contributions below 1e-5 and uses a finite splat footprint.
    relative_l1 = float(np.abs(actual - expected).sum() / expected.sum())
    mass = actual.sum()
    actual_mean = (actual[..., None] * uv).sum(axis=(0, 1)) / mass
    offsets = uv - actual_mean
    actual_covariance = np.einsum("ij,ijk,ijl->kl", actual, offsets, offsets) / mass
    mean_error = float(np.max(np.abs(actual_mean - mean)))
    sigma_error = float(
        np.max(np.abs(np.sqrt(np.diag(actual_covariance) / np.diag(covariance_pixels)) - 1))
    )
    if relative_l1 > 0.025 or mean_error > 0.05 or sigma_error > 0.04:
        raise AssertionError(
            dict(
                relative_l1=relative_l1, mean_error_px=mean_error, sigma_relative_error=sigma_error
            )
        )
    # Exercise real backward for both Gaussian parameters and pose shifts.
    ramp = torch.arange(width, device=image.device, dtype=image.dtype)[None, :] / width
    loss = (image * ramp).sum()
    loss.backward()
    for name in ["_xyz", "_scaling", "_density"]:
        gradient = getattr(model, name).grad
        if gradient is None or not torch.isfinite(gradient).all():
            raise AssertionError(f"Invalid CUDA gradient for {name}")
    if pipe.pose_enabled:
        pose = camera.parallel_pose
        gradient = pose.raw_global_shift.grad
        if gradient is None or not torch.isfinite(gradient).all():
            raise AssertionError("Invalid detector-shift gradient")
        # Independent finite difference in the raw bounded U-shift parameter.
        analytic = float(gradient[0])
        epsilon = 0.002
        values = []
        original = pose.raw_global_shift.detach().clone()
        with torch.no_grad():
            for sign in [-1, 1]:
                pose.raw_global_shift.copy_(original)
                pose.raw_global_shift[0] += sign * epsilon
                probe = render(camera, model, pipe)["render"][0]
                values.append(float((probe * ramp).sum()))
            pose.raw_global_shift.copy_(original)
        numeric = (values[1] - values[0]) / (2 * epsilon)
        if not np.isclose(analytic, numeric, rtol=0.08, atol=0.003):
            raise AssertionError(
                f"Pose shift gradient: analytic={analytic}, finite_difference={numeric}"
            )
    print(
        f"PASS H/W={height}/{width} angle={angle:.2f} pose={pipe.pose_enabled} "
        f"shift={shift.tolist()} mean_error={mean_error:.5f}px L1={relative_l1:.5f}"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pose", action="store_true", help="Also validate the optional pose CUDA extension"
    )
    args = parser.parse_args()
    import torch

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for this renderer validation.")
    from r2_gaussian.dataset.cameras import Camera
    from r2_gaussian.dataset.dataset_readers import angle2pose
    from r2_gaussian.gaussian import GaussianModel, render
    from r2_gaussian.pose import ParallelPose

    for height, width in [(204, 336), (400, 472), (428, 616), (608, 604)]:
        # Use the same normalization as readBlenderInfo: max(sVoxel) -> 2.
        pitch = 2 / max(height, width)
        cfg = dict(
            nDetector=[height, width],
            sDetector=[height * pitch, width * pitch],
            dDetector=[pitch, pitch],
            offDetector=[-2.5 * pitch, 1.25 * pitch],
        )
        center = np.array([0.11, -0.07, 0.09], dtype=np.float32)
        for spherical in [True, False]:
            sigma = np.array([3, 3, 3] if spherical else [3, 5, 4], dtype=np.float32) * pitch
            for angle in [0.0, 0.7, 1.6]:
                for pose_enabled in ([False, True] if args.pose else [False]):
                    for shift in (
                        [np.zeros(2), np.array([1.5, -0.75])] if pose_enabled else [np.zeros(2)]
                    ):
                        w2c = np.linalg.inv(angle2pose(8, angle, cfg["offDetector"]))
                        camera = Camera(
                            colmap_id=0,
                            scanner_cfg=cfg,
                            R=w2c[:3, :3].T,
                            T=w2c[:3, 3],
                            angle=angle,
                            mode=0,
                            FoVx=0.0,
                            FoVy=0.0,
                            image=torch.zeros(1, height, width),
                            image_name="synthetic",
                            uid=0,
                        )
                        if pose_enabled:
                            pose = ParallelPose(
                                1, {"optimize_global_detector_shift": True}, device="cuda"
                            )
                            with torch.no_grad():
                                pose.raw_global_shift.copy_(
                                    torch.as_tensor(
                                        np.arctanh(shift / 3), dtype=torch.float32, device="cuda"
                                    )
                                )
                            camera.attach_parallel_pose(pose, 0)
                        model = GaussianModel()
                        model._xyz = torch.nn.Parameter(
                            torch.as_tensor(center[None], device="cuda")
                        )
                        model._scaling = torch.nn.Parameter(
                            torch.as_tensor(np.log(sigma)[None], device="cuda")
                        )
                        model._rotation = torch.nn.Parameter(
                            torch.tensor([[1.0, 0.0, 0.0, 0.0]], device="cuda")
                        )
                        model._density = torch.nn.Parameter(
                            torch.tensor(
                                [[np.log(np.expm1(1.0))]], dtype=torch.float32, device="cuda"
                            )
                        )
                        pipe = SimpleNamespace(
                            pose_enabled=pose_enabled, compute_cov3D_python=False, debug=False
                        )
                        check_render(torch, render, camera, model, pipe, cfg, center, sigma, shift)
    print("Physical projection validation complete.")


if __name__ == "__main__":
    main()
