import os
import glob
import numpy as np
import tifffile as tiff
import cv2
from skimage.filters import median
from skimage.morphology import disk

input_dir = r'D:\TEM\3D-GS\SiO2'
output_dir = r'D:\SiO2_8bit'
os.makedirs(output_dir, exist_ok=True)

file_list = sorted(glob.glob(os.path.join(input_dir, 'Tomo*.tif*')))
print(f"找到 {len(file_list)} 张投影 TIFF。")

for file_path in file_list[:1]:
    file_name = os.path.basename(file_path)
    img_32bit = tiff.imread(file_path).astype(np.float32)
    
    """# 1. 百分位数截断 (极度重要：砍掉极端亮暗噪点)
    # 这里的 0.5 和 99.5 可以根据图像对比度微调，比如改成 1.0 和 99.0 会对比度更强
    p_min = np.percentile(img_32bit, 0.5)
    p_max = np.percentile(img_32bit, 99.5)
    
    # 将图像裁剪到有效灰度区间
    img_clipped = np.clip(img_32bit, p_min, p_max)
    
    # 2. 线性映射并量化为 8-bit (0-255)
    img_normalized = (img_clipped - p_min) / (p_max - p_min)
    img_8bit = (img_normalized * 255.0).astype(np.uint8)
    
    # 3. 安全且保守的去噪：中值滤波 (Median Filter)
    # disk(2) 表示使用半径为 2 个像素的圆盘去寻找邻域中值，能抹平大部分离散噪点同时保留边缘
    img_denoised = median(img_8bit, disk(2))
    
    # 4. 导出为 PNG (或者 8-bit TIFF)，专供 FBP 初始化使用
    out_name = file_name.replace('.tif', '.png').replace('.tiff', '.png')
    out_path = os.path.join(output_dir, out_name)
    cv2.imwrite(out_path, img_denoised)"""

    # 1. 稍微放宽百分位数，保留更多原始对比度
    p_min = np.percentile(img_32bit, 0.1) 
    p_max = np.percentile(img_32bit, 99.9)
    
    img_clipped = np.clip(img_32bit, p_min, p_max)
    
    # 2. 线性映射并量化为 8-bit (0-255)
    img_normalized = (img_clipped - p_min) / (p_max - p_min)
    img_8bit = (img_normalized * 255.0).astype(np.uint8)
    
    # 3. 直接跳过去噪步骤，保留所有量化后的高频边缘
    # img_denoised = median(img_8bit, disk(2))  <-- 注释掉这一行
    
    # 4. 直接导出 8-bit 图像
    out_name = os.path.splitext(file_name)[0] + '.png'
    out_path = os.path.join(output_dir, out_name)
    cv2.imwrite(out_path, img_8bit)
    
    
    print(f"已生成高对比度初始化切片: {out_name}")
