import numpy as np
import open3d as o3d
import h5py
import matplotlib.pyplot as plt
import os

# 创建保存目录
output_dir = "s3dis0_vis_68.98"
os.makedirs(output_dir, exist_ok=True)

def save_point_cloud_and_view(pcd, name):
    """保存点云文件及窗口截图"""
    # 保存点云
    o3d.io.write_point_cloud(os.path.join(output_dir, f"{name}.ply"), pcd)
    # 保存截图
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=True)
    vis.add_geometry(pcd)
    vis.poll_events()
    vis.update_renderer()
    vis.capture_screen_image(os.path.join(output_dir, f"{name}_view.png"))
    vis.destroy_window()

# 加载数据
h5_file_path = "D:/spt_vis/s3dis5_0_68.898/predictions/test/2000/batch_0.h5"
original_path = "I:/S3DIS_test/Area_5.h5"
label_path = "I:/data.h5"

# ------------------------------------------------------------------
# 1. 保存原始RGB点云
# ------------------------------------------------------------------
with h5py.File(original_path, "r") as area5:
    partition = area5["partition_0"]
    pos = partition["pos"][:]  # 坐标 [N,3]
    rgb = partition["rgb"][:] / 255.0  # RGB颜色 [N,3]

# 创建RGB点云对象
pcd_rgb = o3d.geometry.PointCloud()
pcd_rgb.points = o3d.utility.Vector3dVector(pos)
pcd_rgb.colors = o3d.utility.Vector3dVector(rgb)

# 保存RGB点云及截图
save_point_cloud_and_view(pcd_rgb, "rgb")

# ------------------------------------------------------------------
# 2. 保存预测标签点云
# ------------------------------------------------------------------
with h5py.File(h5_file_path, "r") as f:
    semantic_pred = f["partition_0/semantic_pred"][:]  # 预测标签 [N,]

def label_to_color(labels, colormap="viridis"):
    cmap = plt.get_cmap(colormap)
    return cmap(labels % 256)[:, :3]

# 创建预测标签点云
colors_pred = label_to_color(semantic_pred, colormap="tab20")
pcd_pred = o3d.geometry.PointCloud()
pcd_pred.points = o3d.utility.Vector3dVector(pos)
pcd_pred.colors = o3d.utility.Vector3dVector(colors_pred)

# 保存预测结果
save_point_cloud_and_view(pcd_pred, "prediction")

# ------------------------------------------------------------------
# 3. 保存真实标签点云
# ------------------------------------------------------------------
with h5py.File(label_path, "r") as f:
    label = f["tensor_data"][:]  # 真实标签 [N,14]

real_labels = np.argmax(label, axis=1)  # 转换为 [N,]
colors_real = label_to_color(real_labels, colormap="tab20")

# 创建真实标签点云
pcd_real = o3d.geometry.PointCloud()
pcd_real.points = o3d.utility.Vector3dVector(pos)
pcd_real.colors = o3d.utility.Vector3dVector(colors_real)

# 保存真实结果
save_point_cloud_and_view(pcd_real, "ground_truth")

# ------------------------------------------------------------------
# 4. 保存对比结果点云
# ------------------------------------------------------------------
diff_labels = semantic_pred != real_labels
comparison_colors = np.zeros_like(colors_pred)
comparison_colors[diff_labels] = [1, 0, 0]  # 红色：预测错误
comparison_colors[~diff_labels] = [0, 1, 0]  # 绿色：预测正确

# 创建对比点云
pcd_comparison = o3d.geometry.PointCloud()
pcd_comparison.points = o3d.utility.Vector3dVector(pos)
pcd_comparison.colors = o3d.utility.Vector3dVector(comparison_colors)

# 保存对比结果
save_point_cloud_and_view(pcd_comparison, "comparison")

print("所有结果已保存至 vis_results 目录！")
def split_and_save_all(pcd_rgb, pcd_pred, pcd_real, pcd_compare, grid_size=3.0):
    """切割并保存所有相关点云（RGB、预测、真实、对比）"""
    points = np.asarray(pcd_rgb.points)
    x_min, y_min, z_min = np.min(points, axis=0)
    x_max, y_max, z_max = np.max(points, axis=0)
    
    x_bins = np.arange(x_min, x_max + grid_size, grid_size)
    y_bins = np.arange(y_min, y_max + grid_size, grid_size)
    z_bins = np.arange(z_min, z_max + grid_size, grid_size)

    for i in range(len(x_bins)-1):
        for j in range(len(y_bins)-1):
            for k in range(len(z_bins)-1):
                mask = (points[:,0] >= x_bins[i]) & (points[:,0] < x_bins[i+1]) & \
                       (points[:,1] >= y_bins[j]) & (points[:,1] < y_bins[j+1]) & \
                       (points[:,2] >= z_bins[k]) & (points[:,2] < z_bins[k+1])
                
                if np.sum(mask) == 0:
                    continue
                
                # 创建区域目录
                region_dir = os.path.join(output_dir, f"region_{i}_{j}_{k}")
                os.makedirs(region_dir, exist_ok=True)

                # 保存所有模态数据
                def save_sub_data(original_pcd, name):
                    sub_pcd = o3d.geometry.PointCloud()
                    sub_pcd.points = o3d.utility.Vector3dVector(np.asarray(original_pcd.points)[mask])
                    if original_pcd.has_colors():
                        sub_pcd.colors = o3d.utility.Vector3dVector(np.asarray(original_pcd.colors)[mask])
                    # 保存点云文件
                    o3d.io.write_point_cloud(os.path.join(region_dir, f"{name}.ply"), sub_pcd)
                    # 保存截图
                    save_view(sub_pcd, os.path.join(region_dir, f"{name}_view.png"))

                save_sub_data(pcd_rgb, "rgb")
                save_sub_data(pcd_pred, "prediction")
                save_sub_data(pcd_real, "ground_truth")
                save_sub_data(pcd_compare, "comparison")
def split_and_save_by_bbox(pcd, base_name):
    """交互式选择边界框切割"""
    # 可视化点云并选择区域
    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window()
    vis.add_geometry(pcd)
    vis.run()  # 按 'K' 键绘制边界框
    
    # 获取选择的区域索引
    bbox = vis.get_cropped_geometry()
    if not bbox:
        return
    
    # 提取边界框内点云
    bbox_points = np.asarray(bbox.get_box_points())
    min_bound = np.min(bbox_points, axis=0)
    max_bound = np.max(bbox_points, axis=0)
    
    points = np.asarray(pcd.points)
    mask = np.all((points >= min_bound) & (points <= max_bound), axis=1)
    
    # 创建子点云
    sub_pcd = o3d.geometry.PointCloud()
    sub_pcd.points = o3d.utility.Vector3dVector(points[mask])
    if pcd.has_colors():
        sub_pcd.colors = o3d.utility.Vector3dVector(np.asarray(pcd.colors)[mask])
    
    # 保存子区域
    save_point_cloud_and_view(sub_pcd, os.path.join("split_results", f"{base_name}_bbox"))
    vis.destroy_window()

# 在所有保存操作之后添加切割代码
split_output_dir = os.path.join(output_dir, "split_results")
os.makedirs(split_output_dir, exist_ok=True)


split_and_save_all(
    pcd_rgb=pcd_rgb,          # RGB点云
    pcd_pred=pcd_pred,        # 预测标签点云
    pcd_real=pcd_real,        # 真实标签点云
    pcd_compare=pcd_comparison,  # 对比结果点云
    base_name="area5",        # 基础名称（可选）
    grid_size=10             # 切割网格大小（米）
)
# # 方法二：交互式边界框切割（示例切割预测点云）
# split_and_save_by_bbox(pcd_pred, "pred_split")



