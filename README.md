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
