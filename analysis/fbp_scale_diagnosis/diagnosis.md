# FBP 与 Gaussian 尺度不一致分析

仅使用 D:/TEM/data/output 内真实的 vol_gt*.nii.gz 与 vol_pred*.nii.gz；其他原始 PNG 不作证据。未修改重建或训练代码。

## 结论

FBP 和 Gaussian 的平行束投影尺度约定不一致。正确 pixel_size 只保证 TIGRE 接收到正确的像素物理尺寸；Gaussian 平行束中心投影没有把 sDetector 转换成归一化视野。其单位投影矩阵和固定 tanfov=1 隐含探测器覆盖 [-1,1]²。位姿分支只添加 shift，没有补充视野比例，因此也存在该问题。

## 文件证据

8 组 cata 配对文件均为 256³、spacing=0.0078125，配对 affine 相同。差异发生在数组内结构的位置和尺寸。vol_gt 来自 meta_data.json 指定的参考体，在当前流程中是 FBP，不是独立实验真值。

| 实验 | 横截面 FBP/Gaussian 尺度（两方向） | 缩放配准后的 NCC |
|---|---|---|
| cata1_2 | 1.397, 1.413 | 0.900 |
| cata1 | 0.995, 0.991 | 0.996 |
| cata2_2 | 1.221, 1.226 | 0.878 |
| cata2_3 | 1.201, 1.202 | 0.905 |
| cata3_1 | 1.617, 1.622 | 0.705 |
| cata3_2 | 1.415, 1.418 | 0.861 |
| cata4 | 1.804, 1.815 | 0.796 |

这些值是中心切片经独立强度归一化、平滑后的缩放和平移配准估计，不是物理校准值。cata2_1 对应切片缺乏稳定匹配，排除定量结论。cata4 有截断、噪声与位姿差异，纵向切片的倍率不稳定。

## 尺度关系

记 L=max(sVoxel)，s=2/L，处理后投影为 H×W，像素间距 p（包含 subsample）。TIGRE 的探测器视野为 [Hp,Wp]，归一化后为 [sHp,sWp]。物理一致的平行投影应为 u_ndc=2*x_camera/(sWp)、v_ndc=2*y_camera/(sHp)。当前中心投影则是 u_ndc=x_camera、v_ndc=y_camera。

对相同投影特征，理论 FBP/Gaussian 尺度比为横向 Wp/L、轴向 Hp/L；当 Wp>L，FBP 在归一化网格里就显得更大。矩形投影还会造成横向和轴向比例不同。

例如 cata3_2 横向约 1.416、轴向约 1.067，比值约 1.327；本地原始投影宽高比 476/360=1.322，与该机制吻合。这里只作机制佐证，不假定本地和远端全部参数一致。

## 用户提供的 Cata4 命令

parallel、pixel_subsample=1、recon_pixel_subsample=1、p=0.00004 mm、nVoxel=256³、立方体 sVoxel=FOV_MM、DSD=7、DSO=5、train_fbp 初始化、parallel pose 开启。

本地 Tomo TIFF 为 608×604，因此对应探测器视野为 V=0.02432 mm、U=0.02416 mm。L=FOV_MM 未知；命令不足以计算确定倍率。NIfTI 的 extent=2 是 test.py 用归一化 scanner_cfg 导出形成的，不能从该文件反推出原始 FOV_MM。现存 cata4 的横向约 1.81、纵向约 1.21–1.25，不能由 608×604 和单个 L 单独解释，因此这条命令与这组结果之间还缺历史 meta_data/实际输入形状等信息。不能把拟合结果当作 FOV_MM 的确定反算。

## 初始化影响与修正方向

init_from_fbp.py 在阈值通过的 FBP 体素中抽样，再将每轴映射到 [-1,1]。在中心立方体情况下，这一映射与 FBP 网格一致，但不能补偿 FBP 与 Gaussian 投影模型之间的尺度差异。因此初始点云相对最终 Gaussian 可以过大、过小或各轴不一致。训练会优化坐标、尺度和密度，并进行增密与剪枝，最终外观正常不能证明初始化或物理尺度正确。

物理一致的修正应同时覆盖：平行投影矩阵 diag=[2/sDetector_u,2/sDetector_v]（使用归一化 sDetector）；高斯投影协方差的 tanfovx=sDetector_u/2、tanfovy=sDetector_v/2；位姿分支保留这些尺度并验证 shift 单位；非立方体或非零 offOrigin 的初始化坐标应使用 s*(offOrigin-sVoxel/2+(index+0.5)*sVoxel/nVoxel)。修改后需用正投影与解析体例子检查质心、半径和各轴尺寸，再重建并重新初始化训练。不能直接用旧 checkpoint 评价新几何。

若目的是沿用既有 Gaussian 的显示外观，可把 FBP 重采样到其实际采用的单位探测器坐标后初始化，但这只能保持旧坐标约定，不能证明原始物理尺度正确。

需要历史数据集中的 meta_data.json 和 pipeline_manifest.json（resolved_arguments.fbp_sVoxel）恢复实际物理 FOV；若缓存 vol_train_fbp.npy 曾在修改参数后复用，还需排除旧几何缓存，因为当前缓存只检查形状。

## 验证范围

完成 8 对 NIfTI 头信息和中心切片分析、缩放配准、对应代码审查与纯 NumPy 投影尺度代数复现。当前 Python 环境没有 TIGRE/CUDA 重建依赖，因此未重跑 FBP 或训练。没有修改真实实验结果，也没有修改主流程代码。
