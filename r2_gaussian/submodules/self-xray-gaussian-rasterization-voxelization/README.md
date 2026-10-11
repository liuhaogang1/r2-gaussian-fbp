# Optional pose-calibration rasterizer

This is a separate copy of the differentiable rasterizer from
`ct-pose-calibration-master`. It is used only when `pose_enabled: true`; the
existing `xray_gaussian_rasterization_voxelization` extension is unchanged.

Build from this directory in the project's PyTorch/CUDA environment:

```text
python setup.py build_ext --inplace
```
