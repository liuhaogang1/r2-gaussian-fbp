# Gaussian repair 硬触发轴比实验

在项目的 Linux 训练环境运行。脚本仅组织已有的 `run_pipeline.py`、`train.py`、
`test.py`，不修改训练算法或随机种子行为。输入、数据/图片输出、模型输出、采集配置
均须显式指定。下面路径只是调用示例，可替换为实际路径：

```bash
bash scripts/run_shape_grid.sh \
  --input_dir /root/fnvme/data_10_8/1/cata2_8.4kev_400ms_1 \
  --output_dir /root/fnvme/TEST_11.8/shape_repair_ratios \
  --model_path /root/fnvme/TEST_11.8/models/shape_repair_ratios \
  --config /root/fnvme/data_10_8/1/cata2_8.4kev_400ms_1/config.txt
```

两个输出根目录需要全新或为空，原有实验不会被自动复用或覆盖。默认读取项目内的
`configs/pose_parallel.yaml`，也可显式传入 `--train_config /path/to/pose_parallel.yaml`。
路径中有空格时请加引号。可在以上命令末尾加 `--dry_run` 查看全部命令；该模式不创建
文件、不执行预处理或训练，也不需要 CUDA，但仍检查输入文件是否存在。

## 实验与执行顺序

默认模式为 `--experiment repair_ratios`，直接运行三组：

| 组别 | 针状/片状软阈值 | 两个软权重 | 硬触发轴比 | 方式 |
|---|---|---|---|---|
| H12_repair | 10 / 10 | 0.05 / 0.05 | 12 / 12 | repair |
| H15_repair | 10 / 10 | 0.05 / 0.05 | 15 / 15 | repair |
| H20_repair | 10 / 10 | 0.05 / 0.05 | 20 / 20 | repair |

硬触发轴比是开始离散处理的条件；修复目标仍为软阈值 10。检查间隔 100、连续检查
次数 3、每次最多处理 32 个、密度补偿增益上限 4。两个低贡献阈值固定为 0，让所有
进入处理队列的异常候选都走修复分支。该模式不启用形状删除分支，常规增密/剪枝照常。

1. GPU 0 上运行一次中心估计、FBP、训练视角 FBP 初始化及 sigma=0.8 去噪。
2. 三组训练都独立读取同一个 `shared/init/init_points.npy`，按 12、15、20 串行运行。
3. 每组调用 `test.py`，自动读取该组最终迭代的 pose 快照，导出最终 Gaussian 重建体。
4. 全部训练结束后，按所有成功组的统一灰度范围分别出图。

预处理参数沿用原命令：`Tomo_*.tiff`/`Flat_*.tiff`，平场开启，dark=0，flat/log eps=1e-6，
条纹校正 snr=3/la_size=61/sm_size=21/dim=1，transmission，parallel，shift_v=0，
像素/重建下采样均为 1，resize_order=1，projection_scale=0.125，pixel_size=0.00004，
nVoxel=256³，sVoxel=auto，offOrigin=0，DSD=7，DSO=5，accuracy=0.5，hann，pair 中心估计，
360 个训练视角、0 个测试视角。初始化为 train_fbp，sigma=0.8 并保存 NIfTI，
100000 点、density_thresh=0.25、density_rescale=0.2、normalize_percentile=99.5。

训练保留 pose 开启、30000 步、shape_start=500、shape_end=30000、ramp=1000、
hard_action=repair、最多 300000 点及 seed=0。除模型路径和两个硬触发轴比参数外，
三组训练配置相同。现有训练内部固定随机种子为 0，本脚本保持该行为。
测试保留 `iteration=-1` 和 `skip_render_test`，仍执行训练投影评价与体重建。

## 输出

```text
output_dir/
  shared/center.json
  shared/dataset/                  # 共用投影、FBP、meta_data.json
  shared/init/init_points.npy      # 共用去噪初始化点
  shared/init/denoise_report.json
  cases/H12_repair/train_config.yml # 每组实际交给 train.py 的完整配置
  logs/shared.log
  logs/H12_repair_train.log
  logs/H12_repair_test.log
  images/H12_repair.png            # 每组一张 3×3 PNG，共三张
  images/H15_repair.png
  images/H20_repair.png
  images/slices/H12_repair.npz     # 该组九张原始浮点切片
  images/display_settings.json     # 共享灰度、索引、方向和对应重建体
  shape_grid_manifest.json         # 全部命令、解析参数、状态、共用输入哈希
  summary.csv

model_path/
  H12_repair/point_cloud/iteration_30000/point_cloud.pickle
  H12_repair/test/iter_30000/vol_pred.npy
  H12_repair/test/iter_30000/vol_pred.nii.gz
  ...其余两组独立模型目录
```

每张 PNG 的行依次为 X/Y/Z 法向，列依次为 25%/50%/75%。256³ 重建体取从 0 开始的
64/128/192 层。X 切片为 YZ 面、Y 为 XZ 面、Z 为 XY 面；显示时转置面内数组并采用
`origin=lower`，坐标轴使用元数据的原始物理长度和同一比例，不对各组单独旋转、翻转
或拉伸。颜色条为重建密度。

所有组使用同一个灰度窗口：每个重建体等间隔抽样至最多约 20 万点，合并后取
0.5%–99.5% 分位数，退化时回退到 min/max。显示范围只用于 PNG，不改变原始密度。
窗口值写入图片和 `display_settings.json`；不对单个面板自动归一化。

`summary.csv` 包含运行状态、耗时、最终点数、训练投影 PSNR/SSIM、相对 FBP 的三维指标，
以及实际硬触发轴比、针状/片状轴比 P99、超过固定参考阈值 10/12/15/20/50 的点数比例
和积分质量代理占比。图片标题分别标记软阈值、软权重、硬触发轴比和最终迭代数。
统计会先恢复 Gaussian 尺度与密度的激活函数。因 n_test=0，表中没有留出投影指标；
三维指标的参考是 FBP，不能视为独立真实体。

## 日志、失败与重新出图

控制台打印每个阶段的命令、日志路径与每组状态。训练进度可另开终端查看：

```bash
tail -f /your/output_dir/logs/H12_repair_train.log
```

预处理失败立即停止。某一训练/测试组失败时默认继续其余组；加 `--fail_fast` 可立即
停止。失败组不进入图片，成功组会统一出图，清单标记 `partial`，进程返回非零。
绘图失败也返回非零，可修复原因后用原命令追加 `--plot_only`，从现有成功组的浮点体
重新出图，不执行预处理、训练或测试。此时 PNG/切片文件会按该清单重新生成。

当前脚本不执行断点续训；重新执行完整实验时须指定新的输出根目录。中断运行不能
被视为完成结果，最终 Gaussian、最终重建体和完成状态均会检查。

## 原九组软约束遍历

如需运行原来的阈值 `[5, 10, 20]` × 权重 `[0.01, 0.05, 0.1]` 九组实验，在命令末尾
显式添加 `--experiment soft_grid`。该模式 hard_action=none，组名仍为 `T5_L0p01` 等。
未传入该选项时默认运行上面的三组 repair 实验。`--plot_only` 按保存清单中的实际分组
重新出图，因此已有九组结果也可以继续使用。
