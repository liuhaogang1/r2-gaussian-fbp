# 物理投影更新

原虚拟探测器兼容模式已删除，不再使用 gaussian_fbp.py 或 fbp_geometry.py。
FBP 使用真实探测器宽高，Gaussian 平行投影默认匹配该物理视野。

服务器同步更新后，已有物理 FBP 可保留，使用新模型目录重新训练；旧模型
不能直接用于验证新投影。旧兼容模式产生的体需要重新生成，不能仅改 metadata。
移除命令和 YAML 中旧的 coordinate_mode 配置。sVoxel auto 保留。

示例：

```bash
bash scripts/run_cata4_physical.sh --stop_after init
bash scripts/run_cata4_physical.sh --start_from train
```

实际 CUDA 验收，不启动训练：

```bash
python scripts/validate_physical_projection.py
python scripts/validate_physical_projection.py --pose
```

本地 CPU 测试覆盖几何计算、初始化、旧模式拒绝和预处理回归；本地无
PyTorch/CUDA/TIGRE，实际渲染脚本需要在服务器运行。具体公式及迁移说明见根目录 README。
