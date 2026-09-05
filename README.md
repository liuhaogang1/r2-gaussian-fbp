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

四个脚本也可以单独运行。FBP 输出目录必须是新的空目录；初始化脚本默认使用 `vol_fbp.npy`，并可单独打开 `--visualize` 或 `--visualize_slices`。

```bash
python prepare_fbp_tiff.py --help
python init_from_fbp.py --help
python train.py --help
python test.py --help
```

本目录不包含原始 TIFF、训练输出或 Git 元数据；数据和模型应放在代码目录之外。
