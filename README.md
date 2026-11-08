# R2-Gaussian FBP Pipeline

这是一个独立、可复现的单管线版本，代码主体基于 2026-08-26 的旧版 R2-Gaussian。保留的流程只有：

1. TIFF 投影预处理、旋转中心估计和 TIGRE 平行束 FBP
2. 从 FBP 体生成 Gaussian 初始化点云
3. R2-Gaussian 训练
4. 训练结果测试与体重建评估

初始化阶段额外保留点云散点图和 FBP 切片可视化。

## 目录

```text
prepare_fbp_tiff.py                FBP 预处理与数据集生成
scripts/fbp_preprocess.py          TIFF 清洗、角度和预处理
scripts/scan_fbp_center_tomopy.py  TomoPy 旋转中心估计
scripts/scan_fbp_center.py         0/180 端点中心扫描
init_from_fbp.py                   FBP -> Gaussian 初始化与可视化
train.py                           训练
test.py                            测试、投影评估和体重建
export_highres.py                 可选：训练完成后分块导出高分辨率 Gaussian 体
run_pipeline.py                    单命令串联上述四个阶段
r2_gaussian/                       R2-Gaussian 核心实现
```

## 环境

建议使用已有的 CUDA/PyTorch 环境，再安装 `requirements.txt` 中的 Python 依赖。TIGRE、SimpleITK 和 TomoPy 需要在当前环境中可导入；如果已有旋转中心 JSON，可以跳过 TomoPy 中心扫描。

### 可选 TomoPy 条纹校正

条纹校正默认关闭；不配置 `stripe_correction` 时，预处理保持原流程。显式配置后，管线会在原始探测器分辨率、空间降采样之前，对完整投影堆栈调用 `tomopy.remove_all_stripe`，并把同一份结果用于中心估计、FBP、训练投影和测试投影：

```yaml
stripe_correction:
  snr: 3.0
  la_size: 61
  sm_size: 21
  dim: 1
```

也可以在命令行传入 JSON：

```bash
python run_pipeline.py ... \
  --stripe_correction '{"snr":3.0,"la_size":61,"sm_size":21,"dim":1}'
```

开启后，FBP 输出目录会保存 `stripe_sinogram_before.png` 和
`stripe_sinogram_after.png`。两张图使用中间探测器行、相同的灰度范围，便于直接比较校正前后的正弦图。`preprocess_meta.json`、`meta_data.json` 和中心 JSON 会记录算法、参数、TomoPy 版本及预览文件。

## 单管线运行

使用已有中心结果：

```bash
python run_pipeline.py \
  --input_dir /path/to/refcorr \
  --config /path/to/refcorr/refcorr.txt \
  --center_json /path/to/refcorr_center.json \
  --output_dir /path/to/dataset/refcorr_fbp \
  --model_path /path/to/output/refcorr_model \
  --input_type line_integral \
  --pixel_subsample 4 \
  --pixel_size 0.02 \
  --projection_scale 0.125 \
  --nVoxel 128 128 128 \
  --sVoxel 2 2 2 \
  --n_points 50000 \
  --iterations 30000
```

位姿矫正是可选分支，默认关闭；原有训练命令不变。平行束实验可以在训练时额外传入：

```bash
python train.py -s /path/to/dataset -m /path/to/model \
  --ply_path /path/to/init.npy \
  --config configs/pose_parallel.yaml
```
使用 `run_pipeline.py` 时，将同一个配置传给 `--train_config`；若配置开启位姿，
管线会在测试阶段自动加载本次训练生成的 `pose_offsets.npz`。

锥束实验使用 `configs/pose_cone.yaml`。开启位姿分支前，需要先在当前
PyTorch/CUDA 环境中运行 `scripts/build_pose_rasterizer.ps1`；默认流程继续使用
现有的 rasterizer，不受新扩展影响。位姿结果保存在模型目录的
`pose_calibration/` 下，原始 `meta_data.json` 不会被覆盖。

## 可选：抑制异常形状 Gaussian

形状抑制默认关闭。关闭时不计算形状损失、不追踪异常状态，也不改变旧检查点的
保存格式。开启后，训练只惩罚超过阈值的细长或扁平 Gaussian；可选的硬处理在多个
检查周期持续异常后执行。建议先只启用软惩罚并检查训练指标，再尝试修复或删除；
`split` 是实验性操作，可能增加点数并改变局部拟合。

例如只抑制细长 Gaussian，并在训练过程中渐增正则权重：

```bash
python train.py -s /path/to/dataset -m /path/to/model \
  --shape_enabled --shape_needle_enabled \
  --shape_needle_threshold 10 --lambda_shape_needle 0.01 \
  --shape_start_iter 500 --shape_end_iter 15000 --shape_ramp_iters 1000 \
  --shape_hard_action none
```

通过 `run_pipeline.py` 启动时使用 `--train_shape_*` 参数，例如
`--train_shape_enabled --train_lambda_shape_needle 0.01`。硬处理模式为
`none`、`prune_low`、`repair`、`split` 或 `auto`：`auto` 会删除满足低密度或低积分代理
阈值的持续异常点，并修复其余持续异常点；它不会自动拆分。`prune_low` 需要通过
`shape_low_density_threshold` 或 `shape_low_mass_threshold` 设置至少一个正阈值，
否则不会因低贡献条件删除 Gaussian。拆分模式受 `max_num_gaussians` 限制。

`bash scripts/run_shape_grid.sh --input_dir ... --output_dir ... --model_path ... --config ...`
默认在软阈值 10、软权重 0.05 下运行硬触发轴比 12/15/20 的三组 repair 实验。
共用一次预处理和去噪初始化，每组生成一张包含 X/Y/Z 三方向 25%/50%/75% 切片的
3×3 PNG，所有组共享灰度范围。显式添加 `--experiment soft_grid` 可运行原阈值
`[5, 10, 20]` × 权重 `[0.01, 0.05, 0.1]` 九组软约束遍历。详细说明见
[形状 repair 轴比实验](scripts/run_shape_grid.md)。

六组形状消融使用 `bash scripts/run_shape_ablation.sh`，须显式传入针状/片状阈值、
两个损失权重及输入/输出路径；支持核验并复用已有九组实验的数据集与去噪初始化点。
分组、调用示例和复用要求见 [六组形状消融说明](scripts/run_shape_ablation.md)。

如果没有中心 JSON，删除 `--center_json`，脚本会先调用 `scan_fbp_center_tomopy.py`，并把中心结果写到输出目录同级的 `<output_dir_name>_center.json`。这要求当前环境安装 TomoPy。

流程会生成：

```text
<output_dir>/vol_fbp.npy
<output_dir>/meta_data.json
<output_dir>/proj_train/
<output_dir>/proj_test/
<output_dir>/init_<output_dir_name>.npy
<output_dir>/init_<output_dir_name>_preview.png
<output_dir>/init_<output_dir_name>_slices.png
<model_path>/
<model_path>/test/iter_*/
```

## 可选：FBP 初始化前轻度去噪

默认关闭，旧命令、FBP 缓存、归一化、阈值、随机采样和默认初始化文件的
行为保持不变。显式添加以下参数后，才对选定的初始化体执行三维高斯滤波：

```bash
  --init_denoise \
  --init_denoise_sigma 0.8 \
  --init_denoise_save_nii
```

处理顺序为：读取 FBP → 清理非有限值和负值 → Gaussian 滤波
（`sigma=0.8` 体素，`mode=nearest`，`truncate=4`）→ 按去噪体正值的
P99.5 归一化 → 严格大于密度阈值的候选区 → 原有无放回采样。
选点和初始密度均使用去噪体；`init_density_rescale` 是密度缩放，不是空间尺度。
去噪后的 P99.5 会变化，因此阈值相同不代表原始强度界限相同。
候选不足时直接报错，不自动降低阈值或重复采样。

`full_fbp`、`train_fbp` 和 `custom` 均支持；只处理实际选定的初始化来源。
`train_fbp` 仍遵循现有训练视角 FBP 的重建/缓存规则；使用已有缓存前，应确认它
对应当前的投影和视角集合，不能把本功能的输入文件哈希当成旧缓存来源的证明。
不修改训练投影、原始 FBP、原始默认初始化、训练损失或 FBP 参考体监督。
首轮建议保留 `init_density_thresh=0.10`、`init_density_rescale=0.25`、
`init_n_points=100000` 及原训练参数，只比较去噪这一项。

启用后默认输出：

```text
<model_path>/init_denoise/
    init_points.npy
    vol_init_denoised.npy
    vol_init_denoised.nii.gz        # --init_denoise_save_nii 时生成
    comparison_center_{x,y,z}.png   # 原体/去噪体共用灰度窗口，另有差值图
    candidate_mask_center_{x,y,z}.png
    denoise_report.json
    pipeline_manifest.json
    .pipeline_train_config.yml
```

启用时不再额外覆写数据集的 `init_<dataset_name>.npy`，训练的 `--ply_path`
和生成的训练配置均指向本次新点云。可通过 `--init_denoise_output_dir` 指定
产物目录（运行 manifest 仍位于模型的 `init_denoise` 目录），通过
`--init_output` 指定独立的 `.npy` 点云文件。指定 `train_model_path` 时以上模型
目录以它为准。

去噪体保存的是归一化之前的强度。NIfTI 的 I/J/K 对应 NPY 的 X/Y/Z，间距为
`sVoxel/nVoxel`，第一个体素中心为 `offOrigin-sVoxel/2+spacing/2`，
使用恒等 LPS 方向和数据集原长度单位。旧项目导出器使用另一种 XY 显示约定；
不能仅凭旧 NIfTI 的屏幕朝向判断新文件错位，应核对数组轴及几何。
导出使用已有流式 NIfTI 写入器，无需新增依赖。来源体旁必须有有效
`meta_data.json`；不会因为输出目录改变而退回单位立方体坐标。

使用已有数据集时，从初始化开始，并使用新的模型目录：

```bash
CUDA_VISIBLE_DEVICES=0 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python -u run_pipeline.py \
  --input_dir /root/fnvme/data_10_8/1/cata2_8.4kev_400ms_1 \
  --output_dir /root/fnvme/TEST_10.27/stripe_correct/data_1_cata2_40nm_1 \
  --model_path /root/fnvme/TEST_10.27/stripe_correct/models/data_1_cata2_40nm_1_denoise08 \
  --config /root/fnvme/data_10_8/1/cata2_8.4kev_400ms_1/config.txt \
  --start_from init --stop_after train --skip_center --skip_fbp \
  --init_source train_fbp \
  --init_denoise --init_denoise_sigma 0.8 --init_denoise_save_nii \
  --init_n_points 100000 --init_density_thresh 0.10 \
  --init_density_rescale 0.25 --init_normalize_percentile 99.5 \
  --train_config /root/fnvme/r2-gaussian-fbp/configs/pose_parallel.yaml \
  --train_pose_enabled \
  --train_parallel_optimize_angle \
  --train_parallel_optimize_global_detector_shift \
  --no-train_parallel_optimize_per_view_detector_shift \
  --train_parallel_optimize_beam_tilt \
  --train_parallel_optimize_detector_roll \
  --train_max_num_gaussians 300000 --train_iterations 30000 \
  --train_test_iterations 30000 --train_save_iterations 30000 \
  --seed 0
```

命令使用当前数据集中的视角记录，不会恢复此前删除的视角。做对照前应确认
两组使用相同的 `meta_data.json` 和有效 FBP 来源。如果只想查看初始化结果，
把 `--stop_after train` 改为 `--stop_after init`。

不能同时指定 `--skip_init`、`--start_from train`、外部 `--train_ply_path`
或旧 `--train_start_checkpoint`，以免去噪参数被忽略或新点云被替换。
`--resume` 会核对去噪/采样参数、输入体和元数据 SHA256、点云及去噪体 SHA256；
变更后会报错，需用新实验目录重新初始化和训练。关闭去噪时不能续跑同一个
去噪实验。正常重做初始化会从来源 FBP 重新滤波，不把去噪结果当缓存反复处理。

统一 YAML 也可启用，命令行优先：

```yaml
init:
  denoise: true
  denoise_sigma: 0.8
  denoise_save_nii: true
  density_thresh: 0.10
  density_rescale: 0.25
  n_points: 100000
```

单独运行初始化脚本时，对应参数为 `--denoise`、`--denoise_sigma`、
`--denoise_output_dir`、`--denoise_save_nii`；默认产物放在源体目录的
`init_denoise` 下。之后手动训练必须通过 `--ply_path` 指向新点云。

## 自动计算重建视野 sVoxel

`--fbp_sVoxel auto` 显式启用逐数据视野计算，实现位于
`scripts/auto_svoxel.py`；数值形式的 `--fbp_sVoxel X Y Z` 仍然有效。
FBP 始终使用实际探测器几何，Gaussian 平行投影默认采用相同的物理视野。

管线支持 `--fbp_sVoxel auto`（别名 `--sVoxel auto`），单独预处理支持
`python prepare_fbp_tiff.py ... --sVoxel auto`。原有三个数值的写法仍有效；
不传该选项时默认仍为 `2 2 2`，不会自动改变旧命令。YAML 可写 `fbp: {sVoxel: auto}`。

auto 从当前输入目录、当前 projection_glob 选中的第一张投影读取 TIFF 头，
根据训练投影的 pixel_subsample 和现有中心裁剪规则，得到实际保存投影的 H、W：

```text
p_effective = pixel_size × pixel_subsample
L = max(H, W) × p_effective
sVoxel = [L, L, L]
```

当前 pixel_subsample=1 时保留原始矩形图像；大于 1 时旧预处理会缩小后
中心裁成近似正方形（奇数差可能保留一像素），auto 按保留的视野计算。
只用于解析重建的 recon_pixel_subsample 不决定 auto 的视野。
pixel_size 必须是正有限数，单位与其他几何一致；锥束仍沿用脚本约定，
将 pixel_size 解释为旋转轴平面像素间距。nVoxel 只决定采样数，不决定 L。

40 nm、训练投影不降采样时：cata1 的 204×336 对应 0.01344 mm；
cata3_2 的 360×476 对应 0.01904 mm；Cata4 的 608×604 对应 0.02432 mm。
这个规则选择覆盖投影视野的立方体，不会推断样品的真实边界或最佳 ROI。

计算结果和来源会打印到终端，并保存在 meta_data.json 的 scanner.sVoxel、
source.sVoxel_resolution 及 pipeline_manifest.json 的 resolved_arguments 中。
从 train 等后续阶段继续时，auto 优先沿用现有数据集的 scanner.sVoxel，
避免重新推算已存在体数据的坐标；即使原始 TIFF 已移动也可继续。

## Gaussian 物理平行投影

已删除虚拟探测器兼容模式及 `--fbp_coordinate_mode` / `--coordinate_mode` 参数，
旧 YAML 中的 `fbp.coordinate_mode` 也不再接受。无需额外参数启用物理投影。

令 `scene_scale = 2/max(sVoxel)`。Gaussian 使用归一化后的真实
`scanner.sDetector=[V,U]` 构造投影矩阵，其 U/V 对角项分别为
`2/sDetector[1]`、`2/sDetector[0]`。两个 CUDA 后端的协方差输入
`tanfovx/tanfovy` 对应探测器 U/V 半宽，因此中心位置和投影宽度使用同一像素间距。
位姿分支沿用相同比例，学习的 detector shift 仍以像素为单位。
锥束的透视投影公式保持原样。

初始化默认按照实际 `sVoxel` 和 `offOrigin` 映射体素中心，支持非立方体
和非零原点；无 scanner 元数据的独立体仍按中心单位立方体处理。FBP 预览
和常规测试体的 NIfTI 均使用原数据集长度单位；常规测试导出时恢复
`scene_scale`，保留实际体素间距和偏移，便于与初始化及高分辨率输出叠加。

已有物理 FBP 数据集可以继续使用，但需要在新模型目录重新训练。旧投影
约定训练得到的 checkpoint/点云不能直接用于验证新几何。旧兼容数据集若记录
`reconstruction.coordinate_mode: gaussian`，数据读取和初始化会明确拒绝；
请从投影重新生成物理 FBP 和初始化，不要删除元数据标签后复用旧体。

Cata4 的物理几何示例使用新的数据和模型目录：

```bash
bash scripts/run_cata4_physical.sh --stop_after init
bash scripts/run_cata4_physical.sh --start_from train
```

CPU 回归测试包括实际矩形尺寸、偏心点、球体投影半径、位姿像素偏移和
非立方体初始化。服务器上可额外验证实际 CUDA 渲染和反向传播：

```bash
python scripts/validate_physical_projection.py
python scripts/validate_physical_projection.py --pose
```

第二条需要已安装本项目的可微位姿 CUDA 扩展。脚本不启动训练，使用解析
高斯线积分校验实际渲染的中心、宽度和强度，并检查位姿平移梯度。此次
修正没有改动 CUDA 源码，已有扩展不需因这次 Python 参数修正而重编译。

## 分阶段运行

四个脚本也可以单独运行。FBP 输出目录必须是新的空目录；初始化脚本默认使用 `vol_fbp.npy`，并可单独打开 `--visualize` 或 `--visualize_slices`。使用训练视角重建时，可以额外用 `--visualize_fbp_nii` 将完整的稀疏视角 FBP 体导出为 3D Slicer 可读取的 NIfTI：

```bash
python init_from_fbp.py \
  --dataset /path/to/dataset \
  --use_train_fbp \
  --visualize_fbp_nii
```

默认生成 `<dataset>/vol_train_fbp.nii.gz`；也可以用 `--fbp_nii_output /path/to/train_fbp.nii.gz` 指定路径。该文件保存的是完整 FBP 体，不是阈值筛选后的初始化点云。

```bash
python prepare_fbp_tiff.py --help
python init_from_fbp.py --help
python train.py --help
python test.py --help
```

训练和常规测试完成后，可选地对训练好的 Gaussian 连续场进行高分辨率采样。该脚本独立运行，不会被 run_pipeline.py 自动调用，也不会修改原有训练或测试结果：

```bash
python export_highres.py \
  -s /path/to/dataset \
  -m /path/to/model \
  --iteration -1 \
  --resolution 1024 1024 1024 \
  --chunk_size 128 128 64
```

默认结果写入 `<model>/highres/`，以磁盘映射方式逐块写入 `.npy`。每次运行自动生成以下全部文件，不需要额外参数：

- `vol_pred_iter_<迭代>_<NX>x<NY>x<NZ>.npy`：完整体数据，数组顺序为 `[X, Y, Z]`。
- 同名 `.json`：模型、网格、坐标、切片显示范围和输出文件信息。
- 同名 `_center_slices.png`：X/Y/Z 三个方向中心切片的拼图，使用统一灰度窗口，并保持物理长宽比。PNG 的显示窗不会改变体数据。
- 同名 `.nii.gz`：可直接拖入 3D Slicer 的完整体数据，固定为 float32，按 Z 切片压缩写出，不额外复制完整体到内存。

默认块大小为 `128 128 64`、NPY 输出类型为 float32；可通过 `--dtype float16` 减小 NPY 体积，但 NIfTI 仍为 float32，以便 Slicer 兼容。使用 `--output` 时，其他三个文件沿用指定文件的目录和主文件名。任一目标文件已存在时，脚本会在查询前停止，不覆盖已有结果。

两个 CUDA 查询后端均使用 `8×8×8` 体素块。为使分块查询与整体验证使用相同的块边界，每个查询块边长按 8 的倍数向下取整，最小为 8，再限制到对应轴的分辨率；末尾允许不足整块。默认块大小不变，`8 9 10` 实际使用 `8 8 8`。JSON 的 `chunk_size_xyz` 保留请求值，`query_chunk_size_xyz` 记录实际值。

所有新导出的 FBP、初始化去噪、常规测试和高分辨率 NIfTI 统一采用 XYZ：I/J/K 对应 NPY 的 X/Y/Z，尺寸为 `(NX, NY, NZ)`，等价于 `sitk.GetImageFromArray(volume.transpose(2, 1, 0))`。内部数组、点云坐标和查询仍为 XYZ。切片预览统一为横纵轴递增的显示方向；初始化与去噪预览增加轴标签，显示调整不改变 NPY 数据。

NIfTI 使用数据集 `meta_data.json` 的原始长度坐标：`spacing = sVoxel / shape_xyz`，首体素中心 `origin = offOrigin - sVoxel/2 + spacing/2`，采用单位方向 LPS，并转换为 NIfTI 的 RAS sform。常规测试导出会恢复训练时的归一化尺度；高分辨率 JSON 顶层 `spacing_xyz/center_xyz/size_xyz` 仍为模型归一化坐标，`nifti_geometry` 记录原数据集空间几何及 `format_version: 2`。新文件描述中写入 `r2g axes=XYZ` 标记，供掩膜回导识别。数据集未声明长度单位，因此不会擅自标记为毫米；Slicer 中的测量数值应结合实际输入单位解读。

此修改只影响此后新生成的输出，不会自动改写旧文件。已有 NPY 可继续作为 XYZ 数据使用；旧 NIfTI 的轴顺序、原点及尺度不会随代码更新，需重新导出后再与新文件比较。

从 Slicer 回导掩膜时，`--reference` 应指向当时用于分割的原始完整 NIfTI；裁剪或重定向后的 labelmap 会按该参考的物理坐标重采样。新文件通常可自动识别，也可显式指定：

```bash
python apply_mask_to_volume.py \
  --volume /path/to/dataset/vol_fbp.npy \
  --mask /path/to/sample_labelmap.nii.gz \
  --reference /path/to/dataset/vol_fbp_preview.nii.gz \
  --mask_axes xyz \
  --output /path/to/dataset/original_masked.npy
```

如果分割使用的是旧版 FBP 或测试导出的 YXZ NIfTI，请保留并传入那份旧参考，同时改用 `--mask_axes legacy_yxz`。旧版去噪输出的载荷已为 XYZ，应选择 `xyz`。无标记且形状不能区分 X/Y 的参考（例如立方体）必须显式选择，程序会拒绝含糊的自动推断；不要用新导出的参考替换当时分割使用的旧参考。

掩膜输出 NPY 始终为 XYZ。程序默认读取 `--volume` 同目录的 `meta_data.json`，也可通过 `--metadata` 指定；有元数据时输出 NIfTI 使用上述数据集几何，无元数据时保留原参考的物理映射，仅重排载荷索引。后者无法补回旧文件缺失的数据集偏移或尺度。

输出是对训练 Gaussian 场的直接密集采样，并非额外训练或超分辨率推断。NIfTI 强度值与 NPY 一致，不随空间坐标恢复而缩放。float32 的 1024³ NPY 约占 4 GiB，NIfTI 还需要额外磁盘空间和压缩时间；其压缩大小取决于数据。原有训练、测试和管线入口保持不变。

本目录不包含原始 TIFF、训练输出或 Git 元数据；数据和模型应放在代码目录之外。
