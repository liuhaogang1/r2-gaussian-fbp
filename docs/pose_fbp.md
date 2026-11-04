# Gaussian 位姿迁移到 FBP：独立离线工具

入口为 `scripts/apply_pose_fbp.py`。只有显式运行这个脚本才会处理数据；现有训练、测试、初始化、管线、FBP 实现和配置均未修改。工具读取现有投影与位姿快照，结果写入新的独立目录，不覆盖原数据、模型或缓存。

## 两种模式

两种模式都允许 `beam_tilt` 非零，不需要重新训练或将它清零。

| 参数 | 处理方式 | 重建输出 |
|---|---|---|
| 默认 | 应用扫描角、全局/逐视图平移和 detector roll；忽略 beam tilt | 原始基线、简化校正 FBP |
| `--ablation` | 简化模式，按平移、roll、扫描角依次加入校正；始终忽略 beam tilt | 四组 FBP |
| `--apply-beam-tilt` | 应用所有支持的位姿参数，beam tilt 进入三维射线几何 | 仅完整位姿 FBP |

`--apply-beam-tilt` 与 `--ablation` 互斥，同时指定会在读取输入或创建输出之前报错。完整模式失败时不会退回简化模式。该开关属于离线重建工具，与训练时是否优化 beam tilt 是独立的。

## 输入和运行

需要准备好的 parallel 数据集（`meta_data.json` 与 `proj_train` 文件记录），以及指定迭代的 `pose_calibration/iteration_<N>/pose_offsets.npz`。仅支持 `binding_version=1`、`pose_format_version=2` 的绑定快照；旧的无绑定文件不能根据文件名猜测对应关系。

以下为 PowerShell 示例，请替换路径：

```powershell
# 默认简化模式：非零 beam tilt 被忽略并记录，正常重建
python scripts/apply_pose_fbp.py `
  --dataset D:\TEM\data\dataset `
  --pose D:\TEM\model\pose_calibration\iteration_30000\pose_offsets.npz `
  --iteration 30000 `
  --output-dir D:\TEM\results\pose_fbp_partial_30000

# 简化模式分项对照，在同一基础命令后增加 --ablation，使用新的输出目录
python scripts/apply_pose_fbp.py `
  --dataset D:\TEM\data\dataset `
  --pose D:\TEM\model\pose_calibration\iteration_30000\pose_offsets.npz `
  --iteration 30000 `
  --output-dir D:\TEM\results\pose_fbp_ablation_30000 `
  --ablation

# 完整位姿 FBP：beam tilt 通过三维几何应用，不做分项重建
python scripts/apply_pose_fbp.py `
  --dataset D:\TEM\data\dataset `
  --pose D:\TEM\model\pose_calibration\iteration_30000\pose_offsets.npz `
  --iteration 30000 `
  --output-dir D:\TEM\results\pose_fbp_full_30000 `
  --apply-beam-tilt
```

输出目录必须尚不存在，并且位于输入数据集与所选位姿快照目录之外。

默认 `--backend tigre` 使用项目现有的 PyTorch/TIGRE 环境和 CUDA。完整模式重建前会用已知三维倾斜几何和线性探测器图案检查实际安装的 CUDA 反投影内核；仅仅接受三列角度不算验证通过。检查失败会停止，不自动降级。

附加选项：

- `--export-only`：只导出投影、掩膜、角度和所选模式的三维几何；不导入 PyTorch/TIGRE、不重建、不生成切片或 NIfTI。可与任意一种模式配合。
- `--backend cpu`：显式选择小体数据的 CPU 参考实现，可运行两种模式并完成导出。体数据较大时，优先使用默认 GPU 后端。
- `--filter ram_lak|shepp_logan|cosine|hamming|hann`：明确选择实际滤波器。默认 `ram_lak`，与当前 TIGRE 上游 FBP 的实际默认一致。新工具显式传参，防止上游将 `geo.filter` 覆盖为缺省值；不改变主流程的滤波行为。所有对照使用同一选择，报告分别记录 scanner 配置值和实际值。

## 数据绑定和单位

工具复用 `dataset_identity()`，核对 scanner、每张训练投影文件的内容及基础角度；身份与 metadata 记录顺序无关。image ID 按主流程规则生成：去掉目录，再取文件名第一个点号之前的部分。

NPZ 中训练相机可能经过随机打乱，不能直接按位置使用位姿数组。工具按 ID 重排 `delta_angle` 和 `view_shift`，输出保持原 metadata 的 `proj_train` 顺序。重复/缺失 ID、错迭代、错角度、错误形状、非有限值和不一致的像素间距都会拒绝。

仅处理快照绑定的训练视图，不给测试视图外推逐视图校正。投影直接读取磁盘预处理后的浮点数组；scanner 使用原始物理单位，不引入训练时 `scene_scale`。投影与体数据保持 float32，不做显示归一化、不裁剪负值。

坐标约定为：投影数组 `[V,U]`，位移向量 `[U,V]`，角度弧度，体数据 `[X,Y,Z]`。像素间距来自 `sDetector/nDetector`。NIfTI 复用现有 `save_nifti_xyz()`，保留 XYZ 轴向、体素中心、spacing 和 offOrigin，不擅自将数据集单位解释为毫米。

## 像素校正和三维几何

两种模式都把学习平移与 detector roll 合并为一次双线性反向采样：

```text
s = global_shift + view_shift[i]                       # [U,V] pixels
q_input = c + D^-1 R(-roll) D (q_output-c) + s
c = ((W-1)/2, (H-1)/2)
D = diag(pixel_u, pixel_v)
```

输出尺寸和像素间距不变，越界填零。掩膜表示取样坐标位于原图像素中心覆盖域内；它记录缺测区域，不参与 FBP 缺失射线补偿。零平移、零 roll 直接复制数组。

扫描角仅更新标签，不把图片旋转 `delta_angle`。角度和平移/roll 去除后的投影均先落盘；FBP 按 `views.json` 重新读取这些文件。

完整模式保留 beam tilt 改变后的射线方向。移除像素中的 roll 后，剩余相机严格对应主流程：

```text
C_remaining = R_world_Z(delta_angle) @ C_base @ Rx(tilt_u) @ Ry(tilt_v)
```

`C_base` 包含原始 DSO 和静态 offDetector。局部 tilt 改变坐标轴而保持相机中心固定，因此转换后的探测器偏移必须包含整个相机中心在倾斜 U/V 轴上的分量，不能只复制旧 offDetector 两个数。学习平移、roll 已写入像素，不再加入几何。

`geometry_full_pose.npz` 保存相机矩阵、射线/U/V 方向、探测器中心、像素间距、扫描角，以及经过验证的 TIGRE ZYZ 角度与逐视图偏移。TIGRE 的欧拉角轴序和正负号由矩阵转换，不能把两个 tilt 值直接填进角度列。

完整模式采用按真实射线轨迹计算的方向性 ramp 滤波，再进行完整几何反投影。方向来自 `world_Z × ray_direction` 在 U/V 平面的分量，包含倾斜导致的尺度因子。TIGRE 内核只做一次反投影求和，投影行翻转也只做一次；不重复施加普通 FBP 中与 DSO/DSD 抵消的比例因子。CPU 参考使用同一滤波数据和物理三维坐标。

两种模式沿用现有 TIGRE 的等权角度积分归一化，不额外引入非均匀角间距权重；实际视图权重与滤波选择均记录在报告中。完整模式的零 tilt 情况调用标准分支，确保退化到同一基线。

**“完整”指所有支持的位姿参数均已应用。** 完整分支是保存轨迹下的方向性滤波、等权积分的近似 FBP；有限角、非均匀采样和倾斜轨迹缺失的频率不会因此自动恢复。它不使用 Gaussian 合成投影，也不改用迭代重建。学习位姿能否改善真实 FBP，仍需要对照实际数据判断。

## 输出

共用文件：

```text
corrected_projections/000000.npy ...
valid_masks/000000.npy ...
angles_original.npy
angles_corrected.npy
views.json
run_report.json
```

简化模式另存 `fbp_original.npy/.nii.gz`、`fbp_partial.npy/.nii.gz` 与 `comparison_xyz.png`。开启 ablation 时，增加 `fbp_shift_only.npy/.nii.gz`、`fbp_shift_roll.npy/.nii.gz`，对应输入保存在 `ablation/shift_only/`、`ablation/shift_roll/`。四组依次是原始、仅平移、平移+roll、平移+roll+扫描角；全部忽略 tilt。

完整模式另存 `geometry_full_pose.npz`、`fbp_full_pose.npy`、`fbp_full_pose.nii.gz` 与 `fbp_full_pose_slices.png`，不额外重建原始或简化基线。对比简化模式时，用相同输入、相同滤波器和另一个输出目录单独运行简化命令。

所有同次生成的对照体和 XYZ 切片共用一个灰度窗。报告包括数据/位姿身份、迭代、每项校正的应用位置、忽略项、有效区域比例、实际滤波器、物理坐标、版本、实现哈希和状态。运行中失败会留下 `status=failed` 的报告及已完成文件；请换一个新目录重试，不会覆盖或自动重用失败目录。

## 验证

```powershell
python -m pytest tests/test_pose_fbp.py tests/test_pose_fbp_geometry.py -q
```

CPU 测试覆盖生产相机矩阵对照、相机中心与偏移、非方形像素、NPZ/metadata 重排、位姿符号、有效掩膜、模式互斥、失败状态、零位姿一致性、NPY/NIfTI 坐标以及真实保存/重读链路。合成测试使用生产相机和独立的三维 Gaussian 密度解析线积分生成非零倾斜数据，验证完整 FBP 的位置、强度和相对误差。

安装 PyTorch/TIGRE 且有 CUDA 时，同一测试命令还会运行实际 GPU 几何探针、GPU/CPU 完整重建对照和零倾斜退化检查；本机缺少依赖时，这项集成测试会明确显示 skipped。

实现参考：[TIGRE 几何接口](https://github.com/CERN/TIGRE/blob/master/Python/tigre/utilities/geometry.py)、[平行束反投影内核](https://github.com/CERN/TIGRE/blob/master/Common/CUDA/voxel_backprojection_parallel.cu)、[方向倾斜平行束 FBP 研究](https://pubmed.ncbi.nlm.nih.gov/24228274/)。
