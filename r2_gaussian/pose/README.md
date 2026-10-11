# Optional pose calibration

The calibration branch is disabled by default.  Use a training config with
`pose_enabled: true` to enable it.  Parallel-beam parameters are controlled
individually by the `parallel_optimize_*` switches.

The branch requires the separately built
`self_xray_gaussian_rasterization_voxelization` extension because camera
matrices must receive gradients.  The existing rasterizer is not replaced.
