import numpy as np
import open3d as o3d
import h5py
import matplotlib.pyplot as plt
import os
from glob import glob
from PIL import Image  # 新增PIL库用于图像拼接

# 配置路径
pred_root =  "D:/spt_vis/kitti_0_62.8" # 包含 batch_0~batch_60 的预测结果
true_root = "D:/spt_vis/kitti_y"  # 新增真实标签路径
original_roots = [
    "D:/spt_vis/kitti_val/2013_05_28_drive_0000_sync",
    "D:/spt_vis/kitti_val/2013_05_28_drive_0002_sync",
    "D:/spt_vis/kitti_val/2013_05_28_drive_0003_sync",
    "D:/spt_vis/kitti_val/2013_05_28_drive_0004_sync", 
    "D:/spt_vis/kitti_val/2013_05_28_drive_0005_sync",
    "D:/spt_vis/kitti_val/2013_05_28_drive_0006_sync",
    "D:/spt_vis/kitti_val/2013_05_28_drive_0007_sync",
    "D:/spt_vis/kitti_val/2013_05_28_drive_0009_sync",
    "D:/spt_vis/kitti_val/2013_05_28_drive_0010_sync"
]  # 验证集的9个原始数据目录
output_root = "kitti_val_62.8"
os.makedirs(output_root, exist_ok=True)
def label_to_color(labels, colormap="viridis"):
    cmap = plt.get_cmap(colormap)
    return cmap(labels % 256)[:, :3]
def save_view(geometry, save_path, width=1920, height=1080):
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=width, height=height, visible=True)
    vis.add_geometry(geometry)
    vis.poll_events()
    vis.update_renderer()
    vis.capture_screen_image(save_path)
    vis.destroy_window()
def process_batch(batch_id, original_h5_path, output_dir):
    try:
        # ========== 数据加载部分 ==========
        # 加载预测数据
        pred_h5_path = os.path.join(pred_root, f"batch_{batch_id}.h5")
        with h5py.File(pred_h5_path, "r") as f:
            pos_pred = f["partition_0/pos"][:]
            semantic_pred = f["partition_0/semantic_pred"][:]

        # 加载真实标签
        true_h5_path = os.path.join(true_root, f"{batch_id}_y.h5")
        with h5py.File(true_h5_path, "r") as f:
            tensor_data = f["tensor_data"][:]  # (N, 16)
        semantic_true = np.argmax(tensor_data, axis=1)  # 真实标签

        # 加载原始点云
        with h5py.File(original_h5_path, "r") as f:
            pos_original = f["partition_0/pos"][:]
            rgb = f["partition_0/rgb"][:] / 255.0

        assert np.allclose(pos_pred, pos_original), "坐标不一致！"

        # ========== 输出目录准备 ==========
        batch_output_dir = os.path.join(output_dir, f"batch_{batch_id}")
        os.makedirs(batch_output_dir, exist_ok=True)

        # ========== 基础可视化部分 ==========
        # 保存RGB点云
        pcd_rgb = o3d.geometry.PointCloud()
        pcd_rgb.points = o3d.utility.Vector3dVector(pos_original)
        pcd_rgb.colors = o3d.utility.Vector3dVector(rgb)
        o3d.io.write_point_cloud(os.path.join(batch_output_dir, "rgb.ply"), pcd_rgb)
        save_view(pcd_rgb, os.path.join(batch_output_dir, "rgb_view.png"))

        # 保存预测结果
        colors_pred = label_to_color(semantic_pred, "tab20")
        pcd_pred = o3d.geometry.PointCloud()
        pcd_pred.points = o3d.utility.Vector3dVector(pos_pred)
        pcd_pred.colors = o3d.utility.Vector3dVector(colors_pred)
        o3d.io.write_point_cloud(os.path.join(batch_output_dir, "prediction.ply"), pcd_pred)
        save_view(pcd_pred, os.path.join(batch_output_dir, "prediction_view.png"))

        # 保存真实标签
        colors_true = label_to_color(semantic_true, "tab20")
        pcd_true = o3d.geometry.PointCloud()
        pcd_true.points = o3d.utility.Vector3dVector(pos_original)
        pcd_true.colors = o3d.utility.Vector3dVector(colors_true)
        o3d.io.write_point_cloud(os.path.join(batch_output_dir, "true.ply"), pcd_true)
        save_view(pcd_true, os.path.join(batch_output_dir, "true_view.png"))

        # ========== 新增差异对比可视化 ==========
        # 计算差异区域
        diff_mask = semantic_pred != semantic_true
        
        # 创建对比颜色数组（绿色表示一致，红色表示差异）
        comparison_colors = np.full_like(colors_pred, [0.0, 1.0, 0.0])  # 初始化为绿色
        comparison_colors[diff_mask] = [1.0, 0.0, 0.0]  # 差异区域设为红色

        # 创建对比点云
        pcd_compare = o3d.geometry.PointCloud()
        pcd_compare.points = o3d.utility.Vector3dVector(pos_original)
        pcd_compare.colors = o3d.utility.Vector3dVector(comparison_colors)
        
        # 保存对比结果
        o3d.io.write_point_cloud(os.path.join(batch_output_dir, "comparison.ply"), pcd_compare)
        save_view(pcd_compare, os.path.join(batch_output_dir, "comparison_view.png"))

        # ========== 复合对比视图 ==========
        # 创建并排对比图像
        true_img = Image.open(os.path.join(batch_output_dir, "true_view.png"))
        pred_img = Image.open(os.path.join(batch_output_dir, "prediction_view.png"))
        comp_img = Image.open(os.path.join(batch_output_dir, "comparison_view.png"))
        
        # 水平拼接三张图
        total_width = true_img.width + pred_img.width + comp_img.width
        max_height = max(true_img.height, pred_img.height, comp_img.height)
        
        comparison_combined = Image.new("RGB", (total_width, max_height))
        comparison_combined.paste(true_img, (0, 0))
        comparison_combined.paste(pred_img, (true_img.width, 0))
        comparison_combined.paste(comp_img, (true_img.width + pred_img.width, 0))
        comparison_combined.save(os.path.join(batch_output_dir, "combined_comparison.png"))

        print(f"Batch {batch_id} 处理完成，差异点比例：{diff_mask.mean():.2%}")
    except Exception as e:
        print(f"处理 Batch {batch_id} 失败: {str(e)}")

# 合并所有原始数据文件（按目录顺序和文件名排序）
original_files = []
for root in original_roots:
    files = sorted(glob(os.path.join(root, "*.h5")))
    original_files.extend(files)
    print(f"目录 {root} 中找到 {len(files)} 个文件")

# 检查文件总数是否匹配
total_batches = 61  # batch_0~batch_60
if len(original_files) < total_batches:
    print(f"警告: 原始文件数量不足！需要 {total_batches} 个，找到 {len(original_files)} 个")

# 处理所有批次
for batch_id in range(total_batches):
    if batch_id >= len(original_files):
        print(f"警告: 没有对应的原始文件用于 batch_{batch_id}")
        continue
    
    original_h5_path = original_files[batch_id]
    process_batch(batch_id, original_h5_path, output_root)

print("所有批次处理完成！")