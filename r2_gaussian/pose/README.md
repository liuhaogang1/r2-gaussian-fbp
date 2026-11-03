# Optional pose calibration

The calibration branch is disabled by default.  Use a training config with
`pose_enabled: true` to enable it.  Parallel-beam parameters are controlled
individually by the `parallel_optimize_*` switches.

The branch requires the separately built
`self_xray_gaussian_rasterization_voxelization` extension because camera
matrices must receive gradients.  The existing rasterizer is not replaced.

Parallel pose updates are composed in conventional camera-to-world form before
the matrix is transposed for the rasterizer. `delta_angle` changes the scan
angle and the camera center's orbit; `beam_tilt` rotates the ray about the
camera-local U/V axes while keeping its center fixed; `detector_roll` rotates
the detector plane about the ray while keeping both the ray and center fixed.
Detector shifts remain in pixels. Shared corrections are also applied to
held-out cameras; per-view angle and shift corrections are train-view only.

Parallel `pose_offsets.npz` files carry `pose_format_version=2`. Files without
that version were trained with the previous transposed-rotation composition
and are rejected by the loader. Retrain pose calibration before using them.
Pose-aware checkpoints also save and restore the pose model and its optimizer;
Gaussian-only checkpoints cannot resume pose-enabled training.
