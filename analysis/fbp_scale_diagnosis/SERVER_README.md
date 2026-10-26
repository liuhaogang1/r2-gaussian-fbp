# Gaussian 视野兼容修正

在服务器 r2-gaussian-fbp 仓库根目录 git pull 获取更新。
修改涉及 prepare_fbp_tiff.py、init_from_fbp.py、run_pipeline.py、
独立可选模块 scripts/auto_svoxel.py、scripts/gaussian_fbp.py、
scripts/fbp_geometry.py、参数解析边界 scripts/fbp_options.py 和回归测试。
未修改 Gaussian 渲染器、训练器或位姿配置。无需重新编译 CUDA 扩展。

示例现使用 --fbp_sVoxel auto，根据当前训练投影视野和 pixel_size 计算，
不再固定 cata1 的 0.01344 mm。Cata4 不降采样时为 0.02432 mm。
auto 不会开启 gaussian 兼容模式；两个参数相互独立。
不传新选项时，旧命令沿用修改前 964783a 的流程，包括旧缓存复用、
点坐标映射和物理单位预览。数据集元数据不会自动开启 gaussian。
下方示例脚本显式开启两个功能，运行该脚本即选择启用。

在服务器仓库根目录先运行：

```bash
bash scripts/run_cata4_gaussian_fbp.sh --stop_after init
```

查看新目录中的：

```text
/root/fnvme/TEST_10.21/data_1_cata4_40nm_gaussian_fbp_auto/
  vol_fbp_preview.nii.gz
  vol_train_fbp_gaussian.nii.gz
  init_data_1_cata4_40nm_gaussian_fbp_auto_preview.png
  init_data_1_cata4_40nm_gaussian_fbp_auto_slices.png
```

确认初始化后继续：

```bash
bash scripts/run_cata4_gaussian_fbp.sh --start_from train
```

如需一次完成全部阶段，首次运行不添加 --stop_after 或 --start_from。
命令使用新输出目录。若首阶段已经完成，应使用 --start_from train，
不能直接再次运行默认全流程，因为 FBP 要求输出目录为空。

自行调整原命令时，关键参数为：

```bash
--fbp_coordinate_mode gaussian \
--fbp_sVoxel auto \
--init_source train_fbp \
--init_visualize_fbp_nii
```

新模式让 FBP 从投影重新反投影到现有 Gaussian 的单位探测器坐标。
训练读取的 pixel_size、sDetector、offDetector 等元数据仍保留原值。
初始点云改变会影响优化轨迹；保留的是投影尺度约定，并不承诺最终密度逐点相同。
gaussian 模式下的 NIfTI 预览和 test.py 一样使用归一化场景单位，
不能将其间距直接当作毫米标定。默认 physical 模式仍可用于原物理几何重建。

本地验证包括已知圆盘的 CPU 滤波反投影、位置/半径恢复、完整/训练 FBP
路径检查、Gaussian 专用缓存失效检查，以及与 964783a 的默认流程对比。
默认流程对比覆盖阶段命令、manifest、输出元数据与数组、几何、预览参数、
非立方体点云和旧缓存复用；未启用功能时禁止导入可选实现模块。
这些测试使用 CPU 重建替身验证流程，不等于真实 GPU 数值一致性测试。
当前本地无 TIGRE/CUDA；SimpleITK 读取测试因缺少依赖跳过，
尚未在真实数据上重跑 GPU 重建或训练。
