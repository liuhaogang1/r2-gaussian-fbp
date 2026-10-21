# 可选：批量运行 FBP pixel_size 候选

这是额外的可选脚本，不接入 `run_pipeline.py`，也不修改现有管线。
它**不自行预处理、不调用旋转中心扫描脚本、不实现另一套 FBP**，而是每个候选值
启动一次未修改的 `prepare_fbp_tiff.py`。因此平场校正、预处理、FBP、结果裁剪、
预览、训练/测试投影拆分和元数据都沿用原脚本。

## 示例

在项目根目录和现有 FBP/CUDA 环境下运行。`--` 后面的参数会传给原脚本：

```powershell
python scripts/sweep_fbp_pixel_size.py `
  --pixel_sizes 0.1 0.01 0.001 0.00001 `
  --output_dir "D:/your_results/pixel_size_sweep" `
  -- `
  --input_dir "D:/your_data" `
  --input_type line_integral `
  --flat_field `
  --projection_glob "Tomo_*.tiff" `
  --flat_glob "Flat_*.tiff" `
  --shift_v 0 `
  --pixel_subsample 4 `
  --projection_scale 0.125 `
  --nVoxel 128 128 128 `
  --sVoxel 2 2 2
```

示例参数仅作展示。除 `--pixel_size` 和 `--output_dir` 外，原脚本所需的其他参数
均应通过 `--` 传入，并使用与你要复现的原流程一致的值。平场相关选项（
`--flat_field`、`--flat_glob`、`--dark_value`、`--flat_eps`）也由原脚本逐组执行。
`--config` 指采集参数 TXT，不是 pipeline YAML。

批量脚本自己管理 `--pixel_sizes` 和根 `--output_dir`。禁止在 `--` 后再次指定
`--pixel_size`、`--output_dir`、`--center_json` 或 `--offDetector`，避免候选被覆盖或
悄悄使用与当前像素尺寸不匹配的中心偏移。

## 旋转中心

批量扫描阶段**不估计/校正旋转中心**，也不调用 `scan_fbp_center_tomopy.py`；
每组传给原脚本的是默认零 `offDetector`。等选定 `pixel_size` 后，再单独运行中心
扫描脚本，并用选定值执行正式的原始管线。

## 输出与执行行为

根输出目录必须新建或为空；已有非空目录不会被覆盖。每个候选使用隔离目录：

```text
pixel_size_sweep/
  commands.json
  summary.csv
  comparison_windows.json
  comparison_[xyz]_[shared|auto]_p01_c01.png
  001_ps_0.1/
    prepare.log
    fbp/                    prepare_fbp_tiff.py 的完整原始输出
      vol_fbp.npy
      vol_fbp_raw.npy
      meta_data.json
      proj_train/
      proj_test/
      ...
```

候选按给定顺序串行运行；默认一组失败后继续，`--fail_fast` 可首次失败即停。
检查 `summary.csv` 和各组 `prepare.log`。原脚本仍会生成它自己的数据集输出；
批量脚本不会调用初始化、训练或测试阶段。

对比图直接读取每组 `fbp/vol_fbp.npy`，按三轴的相同相对切片位置显示。默认
`--slice_fractions 0.25 0.5 0.75`；每行一个成功候选。`shared` 统一灰度窗，
`auto` 每组独立灰度窗；不要单凭亮度判优。候选多时按每页最多 8 组和 6 个切片
位置分页。

固定 `sVoxel/nVoxel` 时改变 `pixel_size` 会改变探测器相对重建网格的覆盖范围。
结果供人工比较，不自动选优，也不能替代真实物理尺寸标定。选定后再做旋转中心
扫描，并使用原管线生成正式数据集。

## 检查

```powershell
python scripts/sweep_fbp_pixel_size.py --help
python -m unittest discover -s tests -p test_sweep_fbp_pixel_size.py
```
