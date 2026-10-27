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
保留原输入长度单位；常规测试体仍按现有流程导出归一化单位，叠加前应换算。

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

NIfTI 的轴顺序与原流程 `test.py` 保持一致：等价于 `sitk.GetImageFromArray(volume.transpose(2, 0, 1))`，因此 NIfTI I/J/K 对应 NPY 的 Y/X/Z，文件尺寸为 `(NY, NX, NZ)`。仅 NIfTI 写出时交换 X/Y，不对数据做 180° 旋转；分块查询、NPY 和中心切片 PNG 仍保持原来的 XYZ 约定。

本次轴顺序兼容不改变现有 spacing、原点或尺度：NIfTI 仍使用数据集 `meta_data.json` 的原始长度坐标，原点对应第一个 voxel 的中心，采用单位方向 LPS 并转换为 NIfTI 的 RAS sform。spacing 和原点按原有顺序设置，不随体素轴交换。JSON 原有的 `spacing_xyz/center_xyz/size_xyz` 仍是模型归一化坐标，`nifti_geometry` 记录 NIfTI 实际尺寸、轴映射及空间几何。原流程测试体采用归一化尺度且原点为零，故方向兼容不代表两份文件可以直接空间叠加。由于数据集未声明长度单位，NIfTI 不擅自标记为毫米；如果原始参数并非毫米，Slicer 中的测量数值需要结合实际单位解读。

此修改只影响此后新导出的 NIfTI，不会自动改写之前生成的文件。

输出是对训练 Gaussian 场的直接密集采样，并非额外训练或超分辨率推断。NIfTI 强度值与 NPY 一致，不随空间坐标恢复而缩放。float32 的 1024³ NPY 约占 4 GiB，NIfTI 还需要额外磁盘空间和压缩时间；其压缩大小取决于数据。原有训练、测试和管线入口保持不变。

本目录不包含原始 TIFF、训练输出或 Git 元数据；数据和模型应放在代码目录之外。
