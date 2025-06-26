import numpy as np
import open3d as o3d
import h5py
import matplotlib.pyplot as plt

# 加载数据
# h5_file_path = "I:/Dales_pred/test/0/batch_0.h5"
h5_file_path = "D:/spt_vis/predictions/test/400/batch_0.h5"
label_path = "D:/spt_vis/dales_y/0_y.h5"
# Dales可视化
with h5py.File(h5_file_path, "r") as f:
    pos = f["partition_0/pos"][:]          # 坐标 (N, 3)
    semantic_pred = f["partition_0/semantic_pred"][:]  # 语义标签 (N,)
    x = f["partition_0/x"][:]              # 特征 (N, C)
    intensity = x[:, 0]  # 修正1：取一维数组 (N,)

# 创建 Open3D 点云对象（强度可视化）
pcd_intensity = o3d.geometry.PointCloud()
pcd_intensity.points = o3d.utility.Vector3dVector(pos)

# 将强度归一化到 0-1 范围并赋值为颜色
if intensity.max() == intensity.min():
    intensity_normalized = np.zeros_like(intensity)
else:
    intensity_normalized = (intensity - intensity.min()) / (intensity.max() - intensity.min())

colors_intensity = plt.cm.viridis(intensity_normalized)[:, :3]  # Viridis colormap
pcd_intensity.colors = o3d.utility.Vector3dVector(colors_intensity)

# 可视化点云
#o3d.visualization.draw_geometries([pcd_intensity], window_name="Point Cloud Visualization") 
def label_to_color(labels, colormap="viridis"):
    # 使用 matplotlib 的 colormap 生成颜色
    import matplotlib.pyplot as plt
    cmap = plt.get_cmap(colormap)
    colors = cmap(labels % 256)[:, :3]  # 取 RGB 通道，忽略 Alpha
    return colors

# 生成颜色（若标签类别较多，建议使用固定调色板）
colors = label_to_color(semantic_pred, colormap="tab20")

# 创建点云对象
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(pos)
pcd.colors = o3d.utility.Vector3dVector(colors)

# 可视化
#o3d.visualization.draw_geometries([pcd], window_name="Point Cloud Visualization")

with h5py.File(label_path, "r") as f:
    label = f["tensor_data"][:]          

real_labels = np.argmax(label, axis=1)
# 生成真实标签的颜色
real_colors = label_to_color(real_labels, colormap="tab20")

pcd_real = o3d.geometry.PointCloud()
pcd_real.points = o3d.utility.Vector3dVector(pos)
pcd_real.colors = o3d.utility.Vector3dVector(real_colors)  # 真实标签的颜色

# 对比可视化：同时显示预测标签和真实标签
#o3d.visualization.draw_geometries([pcd_real], window_name="Point Cloud real")
# 比较预测标签与真实标签
diff_labels = semantic_pred != real_labels

# 为不同标签点设置颜色
comparison_colors = np.zeros_like(colors)
comparison_colors[diff_labels] = [1, 0, 0]  # 不同标签的点显示为红色
comparison_colors[~diff_labels] = [0, 1, 0]  # 相同标签的点显示为绿色

# 创建点云对象，显示标签差异
pcd_comparison = o3d.geometry.PointCloud()
pcd_comparison.points = o3d.utility.Vector3dVector(pos)
pcd_comparison.colors = o3d.utility.Vector3dVector(comparison_colors)

# 可视化对比结果
#o3d.visualization.draw_geometries([pcd_comparison], window_name="Prediction vs Ground Truth")



# ==== 保存点云文件 ====
# 预测标签
o3d.io.write_point_cloud("vis/predicted0.ply", pcd)
# 真实标签
o3d.io.write_point_cloud("vis/ground_truth0.ply", pcd_real)
# 对比结果
o3d.io.write_point_cloud("vis/comparison0.ply", pcd_comparison)

o3d.io.write_point_cloud("intensity0.ply", pcd_intensity)
# ==== 保存窗口截图 ====
def capture_and_save_view(geometry, filename, width=1920, height=1080):
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=width, height=height, visible=True)
    vis.add_geometry(geometry)
    vis.poll_events()
    vis.update_renderer()
    vis.capture_screen_image(filename)
    vis.destroy_window()

# 预测标签截图
capture_and_save_view(pcd, "vis/prediction_view0.png")
# 真实标签截图
capture_and_save_view(pcd_real, "vis/ground_truth_view0.png")
# 对比结果截图
capture_and_save_view(pcd_comparison, "vis/comparison_view0.png")

capture_and_save_view(pcd_intensity, "vis/intensity_view0.png")