# 六组 Gaussian 形状约束消融

在项目 Linux 训练环境的项目根目录运行。阈值、权重、输入/输出/模型目录和采集配置
都必须显式指定。该脚本不会修改训练损失实现及现有固定种子行为。

## 新建一轮完整实验

```bash
bash scripts/run_shape_ablation.sh \
  --input_dir /root/fnvme/data_10_8/1/cata2_8.4kev_400ms_1 \
  --config /root/fnvme/data_10_8/1/cata2_8.4kev_400ms_1/config.txt \
  --output_dir /root/fnvme/TEST_11.4/shape_ablation \
  --model_path /root/fnvme/TEST_11.4/models/shape_ablation \
  --needle_threshold 10 \
  --sheet_threshold 10 \
  --lambda_needle 0.05 \
  --lambda_sheet 0.05
```

以上路径为示例，需替换成实际路径。两个输出根目录须全新或为空。阈值必须有限且
大于 1，权重必须有限且非负。权重为 0 时相应软惩罚没有作用。

| 组别 | 针状软权重 | 片状软权重 | 硬处理 |
|---|---|---|---|
| A0_baseline | 0，总开关关闭 | 0，总开关关闭 | none |
| A1_needle | lambda_needle | 0，片状分支关闭 | none |
| A2_sheet | 0，针状分支关闭 | lambda_sheet | none |
| A3_soft_both | lambda_needle | lambda_sheet | none |
| A4_hard_repair | 0，分支开启 | 0，分支开启 | repair |
| A5_soft_repair | lambda_needle | lambda_sheet | repair |

各组均记录传入的两个目标阈值。A0 不运行形状分支；A4 保留形状总开关及两个分支，
使两个软损失权重为 0，硬修复仍然生效。

## 复用已有九组实验的预处理和初始化

在同一条命令末尾同时追加下面两个参数：

```bash
  --shared_dataset_dir /root/fnvme/TEST_11.4/shape_grid_9/shared/dataset \
  --shared_init_path /root/fnvme/TEST_11.4/shape_grid_9/shared/init/init_points.npy
```

六组训练直接读取这些外部产物，跳过中心估计、FBP、去噪及点采样，不向被复用的
目录写入文件。新实验的两个输出根目录仍须全新或为空。`--input_dir`、`--config`
仍需指向原始采集，供核对来源。

复用需要初始化目录内的 `pipeline_manifest.json`、`denoise_report.json` 和其中引用的
原始产物仍在原位置。校验包括：原流程 center/FBP/init 均完成；数值参数与固定流程
相同；元数据、投影和体数据均有保存的哈希且未变化；去噪输入与指定数据集一致；
100000 个初始化点、sigma=0.8、阈值=0.25、rescale=0.2、percentile=99.5、seed=0。
若外层九组/六组清单保存了采集配置哈希，也会核对该哈希。

此校验不重算旧流水线的代码签名，因此添加本脚本本身不会让已有数据无法复用。
原始 TIFF 不重新处理；没有完整来源记录、移动/删除了引用产物或参数不一致时会报错，
不自动混用。校验是只读操作，可能需要一次读取较大的产物文件。

## 固定参数

共用输入与九组流程完全相同：Tomo/Flat TIFF、平场、dark=0、flat/log eps=1e-6、
条纹校正 snr=3/la_size=61/sm_size=21/dim=1、transmission、parallel、shift_v=0，
像素/重建下采样均为 1，resize_order=1，projection_scale=0.125，pixel_size=0.00004，
nVoxel=256³、sVoxel=auto、offOrigin=0，DSD=7、DSO=5、accuracy=0.5、hann、pair 中心估计，
训练/测试视角数为 360/0。初始化采用 train_fbp，sigma=0.8 并保存 NIfTI，100000 点、
density_thresh=0.25、density_rescale=0.2、normalize_percentile=99.5。

训练固定使用 `configs/pose_parallel.yaml`、GPU 0、expandable_segments:True，开启 pose，
30000 步、形状约束 500–30000 步、ramp=1000、最多 300000 点。学习率、TV、DSSIM 和
常规增密/剪枝保持已有流程设置；seed=0，训练内部也保持现有固定为 0 的行为。
测试使用 iteration=-1，跳过测试投影渲染，保留训练投影评价和体重建。

硬处理固定：needle/sheet 严重异常触发轴比均为 50，每 100 步检查，连续 3 次异常，
每次最多处理 32 个点；密度补偿上限为 4，低密度/低质量阈值均为 0。传入的
needle_threshold/sheet_threshold 是软惩罚阈值及修复目标阈值。阈值或样本形状使硬处理
没有触发时，A4/A5 的效果可能很小，需结合统计与训练日志判断。

## 图片与汇总

`output_dir/images/` 下输出六张独立 PNG，每张都是：行 X/Y/Z，列 25%/50%/75%。
256³ 的索引为从 0 开始的 64/128/192。所有成功组统一灰度范围，坐标方向和物理比例
相同；图头标记实际开关、两个目标阈值、两个实际权重、硬处理方式和最终迭代数。
原始九个浮点切片保存于 `images/slices/`，显示设置保存于 `images/display_settings.json`。

每组模型位于 `model_path/A0_baseline` 等目录，最终体为
`test/iter_30000/vol_pred.npy`；日志位于 `output_dir/logs/`，完整训练 YAML 位于
`output_dir/cases/`。清单为 `shape_ablation_manifest.json`，汇总为 `summary.csv`，包含
实际分支与参数、状态、耗时、最终点数、训练投影指标、相对 FBP 的三维指标和形状统计。
因测试视角数为 0，训练投影指标用于衡量拟合；FBP 参考体也不是独立真实体。

加 `--dry_run` 可校验并打印全部命令，不创建输出文件。加 `--plot_only` 可从六组清单的
成功结果重新生成 PNG，不执行预处理、训练或测试；路径和四个形状参数必须与原命令
一致，此时不要求原始 TIFF 仍挂载。图片标题始终使用保存的实际参数。

共用阶段失败时停止；某组失败时默认继续其余组，成功组仍会出图，清单标记 partial，
返回非零。`--fail_fast` 可在第一组失败后停止。当前不执行断点续训，完整重跑需新的
输出目录。
