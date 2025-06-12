import numpy as np
import open3d as o3d
import h5py
import matplotlib.pyplot as plt
import os

# 创建保存结果的目录
output_dir = "dales0_vis"
os.makedirs(output_dir, exist_ok=True)

def process_batch(batch_id):
    """处理单个 batch 的可视化并保存结果"""
    try:
        # 生成文件路径
        h5_file_path = f"D:/spt_vis/predictions/test/400/batch_{batch_id}.h5"  # 根据实际路径调整
        label_path = f"D:/spt_vis/dales_y/{batch_id}_y.h5"  # 根据实际标签路径调整

        # 加载数据
        with h5py.File(h5_file_path, "r") as f:
            pos = f["partition_0/pos"][:]
            semantic_pred = f["partition_0/semantic_pred"][:]
            x = f["partition_0/x"][:]
            intensity = x[:, 0]

        # 创建强度点云
        pcd_intensity = o3d.geometry.PointCloud()
        pcd_intensity.points = o3d.utility.Vector3dVector(pos)
        # 强度归一化
        if intensity.max() == intensity.min():
            intensity_normalized = np.zeros_like(intensity)
        else:
            intensity_normalized = (intensity - intensity.min()) / (intensity.max() - intensity.min())
        colors_intensity = plt.cm.viridis(intensity_normalized)[:, :3]
        pcd_intensity.colors = o3d.utility.Vector3dVector(colors_intensity)

        # 创建预测标签点云
        colors_pred = label_to_color(semantic_pred, colormap="tab20")
        pcd_pred = o3d.geometry.PointCloud()
        pcd_pred.points = o3d.utility.Vector3dVector(pos)
        pcd_pred.colors = o3d.utility.Vector3dVector(colors_pred)

        # 加载真实标签
        with h5py.File(label_path, "r") as f:
            label = f["tensor_data"][:]
        real_labels = np.argmax(label, axis=1)
        colors_real = label_to_color(real_labels, colormap="tab20")
        pcd_real = o3d.geometry.PointCloud()
        pcd_real.points = o3d.utility.Vector3dVector(pos)
        pcd_real.colors = o3d.utility.Vector3dVector(colors_real)

        # 对比标签差异
        diff_labels = semantic_pred != real_labels
        comparison_colors = np.zeros_like(colors_pred)
        comparison_colors[diff_labels] = [1, 0, 0]
        comparison_colors[~diff_labels] = [0, 1, 0]
        pcd_comparison = o3d.geometry.PointCloud()
        pcd_comparison.points = o3d.utility.Vector3dVector(pos)
        pcd_comparison.colors = o3d.utility.Vector3dVector(comparison_colors)

        # ==== 保存结果 ====
        batch_dir = os.path.join(output_dir, f"batch_{batch_id}")
        os.makedirs(batch_dir, exist_ok=True)

        # 保存点云文件
        o3d.io.write_point_cloud(os.path.join(batch_dir, "intensity.ply"), pcd_intensity)
        o3d.io.write_point_cloud(os.path.join(batch_dir, "predicted.ply"), pcd_pred)
        o3d.io.write_point_cloud(os.path.join(batch_dir, "ground_truth.ply"), pcd_real)
        o3d.io.write_point_cloud(os.path.join(batch_dir, "comparison.ply"), pcd_comparison)

        # 保存截图
        def capture_view(geometry, name):
            vis = o3d.visualization.Visualizer()
            vis.create_window(visible=True)
            vis.add_geometry(geometry)
            vis.poll_events()
            vis.update_renderer()
            vis.capture_screen_image(os.path.join(batch_dir, f"{name}.png"))
            vis.destroy_window()

        capture_view(pcd_intensity, "intensity_view")
        capture_view(pcd_pred, "prediction_view")
        capture_view(pcd_real, "ground_truth_view")
        capture_view(pcd_comparison, "comparison_view")

        print(f"Batch {batch_id} 处理完成")

    except Exception as e:
        print(f"处理 Batch {batch_id} 时出错: {str(e)}")

def label_to_color(labels, colormap="viridis"):
    """标签转颜色函数"""
    cmap = plt.get_cmap(colormap)
    return cmap(labels % 256)[:, :3]

# 循环处理所有 batch
for batch_id in range(25):  # 0到24
    process_batch(batch_id)

print("所有批次处理完成！")