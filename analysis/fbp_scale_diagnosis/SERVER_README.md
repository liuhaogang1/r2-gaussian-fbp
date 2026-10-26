# Gaussian 视野兼容修正

在服务器 r2-gaussian-fbp 仓库根目录 git pull 获取更新。
修改涉及 prepare_fbp_tiff.py、init_from_fbp.py、run_pipeline.py、
scripts/fbp_geometry.py、scripts/run_cata4_gaussian_fbp.sh 和回归测试。
未修改 Gaussian 渲染器、训练器或位姿配置。无需重新编译 CUDA 扩展。

示例现使用 --fbp_sVoxel auto，根据当前训练投影视野和 pixel_size 计算，
不再固定 cata1 的 0.01344 mm。Cata4 不降采样时为 0.02432 mm。
auto 不会开启 gaussian 兼容模式；两个参数相互独立。

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

本地验证：25 项测试中 23 项通过，2 项 SimpleITK 读取测试因缺少依赖跳过。
包括已知圆盘的 CPU 滤波反投影、位置/半径恢复、完整/训练 FBP 路径参数检查、
元数据和投影保持检查、缓存失效检查。当前本地无 TIGRE/CUDA 运行环境，
尚未在真实数据上重跑 GPU 重建或训练。
