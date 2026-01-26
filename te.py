import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
import sys

# Import các hàm cần thiết từ các file khác
# Giả sử các file io.py, process.py, target.py đã có trong cùng thư mục

from utils import read_points, read_label, read_calib
from utils import bbox_camera2lidar, bbox3d2bevcorners
from models import Target
import torch

# ===== CẤU HÌNH =====
# Đường dẫn tới các file dữ liệu
POINT_CLOUD_PATH = r"datasets\point_cloud_reduced\training\003500.bin"
LABEL_PATH = r"KITTICrowd\training\label_2\003500.txt"
CALIB_PATH = r"KITTICrowd\training\camera.json"

# Config từ file config.py
point_cloud_range = [0, -15.36, -3, 30.72, 15.36, 2]
voxel_size = [0.12, 0.12, 4]
classes = {'Pedestrian': 0}

# ===== ĐỌC DỮ LIỆU =====
print("Đang đọc dữ liệu...")

# Đọc point cloud
pts = read_points(POINT_CLOUD_PATH).astype(np.float32)
print(f"Số lượng điểm: {len(pts)}")

# Đọc labels
annos = read_label(LABEL_PATH)
names = annos['names']
locations = annos['locations']
dimensions = annos['dimensions']
rotation_y = annos['rotation_y']
print(f"Số lượng objects: {len(names)}")

# Đọc calibration
calib_info = read_calib(CALIB_PATH)
tr_velo_to_cam = calib_info['tr_velo_to_cam']
r0_rect = calib_info['r0_rect']

# ===== CHUYỂN ĐỔI BBOX SANG LIDAR COORDINATE =====
print("Chuyển đổi bounding boxes sang LiDAR coordinate...")

# Tạo gt_bboxes trong camera coordinate
gt_bboxes = np.concatenate([locations, dimensions, rotation_y[:, None]], axis=1)  # (m, 7)

# Chuyển sang LiDAR coordinate
gt_bboxes_3d = bbox_camera2lidar(gt_bboxes, tr_velo_to_cam, r0_rect)
print(f"Shape của gt_bboxes_3d: {gt_bboxes_3d.shape}")

# ===== TẠO HEATMAP VÀ OFFSETMAP =====
print("Tạo heatmap và offsetmap...")

# Khởi tạo Target generator
target_generator = Target(point_cloud_range, voxel_size)

# Chuyển gt_bboxes_3d sang tensor
batch_gt_bboxes = [torch.from_numpy(gt_bboxes_3d).float()]
batch_size = 1
device = torch.device('cpu')

# Tạo heatmap
gt_heatmap = target_generator.get_heatmap(batch_gt_bboxes, batch_size, device)
print(f"Shape của heatmap: {gt_heatmap.shape}")

# Tạo offsetmap
gt_offsetmap = target_generator.get_offsetmap(batch_gt_bboxes, batch_size, device)
print(f"Shape của offsetmap: {gt_offsetmap.shape}")

# ===== VISUALIZATION =====
print("Đang tạo visualization...")

# Lọc points trong point_cloud_range
mask = (pts[:, 0] >= point_cloud_range[0]) & (pts[:, 0] <= point_cloud_range[3]) & \
       (pts[:, 1] >= point_cloud_range[1]) & (pts[:, 1] <= point_cloud_range[4]) & \
       (pts[:, 2] >= point_cloud_range[2]) & (pts[:, 2] <= point_cloud_range[5])
pts_filtered = pts[mask]

# Tính offset cho mỗi point
offsetmap_np = gt_offsetmap[0].numpy()  # (2, H, W)
heatmap_np = gt_heatmap[0, 0].numpy()  # (H, W)

# Chuyển point coordinates sang grid coordinates
pts_x = ((pts_filtered[:, 0] - target_generator.x_offset) / target_generator.vx).astype(np.int32)
pts_y = ((pts_filtered[:, 1] - target_generator.y_offset) / target_generator.vy).astype(np.int32)

# Lọc points nằm trong grid
valid_mask = (pts_x >= 0) & (pts_x < target_generator.x_grid) & \
             (pts_y >= 0) & (pts_y < target_generator.y_grid)

# Tính bounding box corners
bev_corners = bbox3d2bevcorners(gt_bboxes_3d)  # (n, 4, 2)

# ===== HÌNH 1: Point Cloud + Bounding Boxes (BEV) =====
fig1, ax1 = plt.subplots(1, 1, figsize=(8, 8))

# Vẽ point cloud (BEV: x-y plane)
ax1.scatter(pts_filtered[:, 0], pts_filtered[:, 1], s=0.1, c='blue', alpha=0.5)

# Vẽ bounding boxes với linewidth nhỏ hơn
for corners in bev_corners:
    polygon = patches.Polygon(corners, fill=False, edgecolor='red', linewidth=0.5)
    ax1.add_patch(polygon)

ax1.set_xlim(point_cloud_range[0], point_cloud_range[3])
ax1.set_ylim(point_cloud_range[1], point_cloud_range[4])
ax1.set_aspect('equal')
ax1.set_xlabel('X (m)', fontsize=12)
ax1.set_ylabel('Y (m)', fontsize=12)
ax1.set_title('Point Cloud + Bounding Boxes (BEV)', fontsize=14)
ax1.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('1_pointcloud_bboxes.png', dpi=150, bbox_inches='tight')
print("Đã lưu hình 1 vào '1_pointcloud_bboxes.png'")
plt.close()

# ===== HÌNH 2: Heatmap (BEV) =====
fig2, ax2 = plt.subplots(1, 1, figsize=(8, 8))

im = ax2.imshow(heatmap_np, cmap='hot', origin='lower', 
                extent=[point_cloud_range[0], point_cloud_range[3],
                        point_cloud_range[1], point_cloud_range[4]])
plt.colorbar(im, ax=ax2, label='Heatmap Value')

ax2.set_xlabel('X (m)', fontsize=12)
ax2.set_ylabel('Y (m)', fontsize=12)
ax2.set_title(f'Ground Truth Heatmap (BEV) - {heatmap_np.shape[0]}x{heatmap_np.shape[1]}', fontsize=14)

plt.tight_layout()
plt.savefig('2_heatmap.png', dpi=150, bbox_inches='tight')
print("Đã lưu hình 2 vào '2_heatmap.png'")
plt.close()

# ===== HÌNH 3: Point Cloud với offset (đỏ) và không offset (xanh) =====
fig3, ax3 = plt.subplots(1, 1, figsize=(8, 8))

# Lấy offset tương ứng cho valid points
pts_x_valid = pts_x[valid_mask]
pts_y_valid = pts_y[valid_mask]
pts_valid = pts_filtered[valid_mask]

offset_x = offsetmap_np[0, pts_y_valid, pts_x_valid]  # (N,)
offset_y = offsetmap_np[1, pts_y_valid, pts_x_valid]  # (N,)

# Xác định points có offset (không phải zero)
has_offset_mask = (offset_x != 0) | (offset_y != 0)

# Points không có offset - màu xanh
pts_no_offset = pts_valid[~has_offset_mask]
if len(pts_no_offset) > 0:
    ax3.scatter(pts_no_offset[:, 0], pts_no_offset[:, 1], 
                s=0.1, c='blue', alpha=0.5, label='No Offset')

# Points có offset - dịch chuyển và tô màu đỏ
pts_with_offset = pts_valid[has_offset_mask].copy()
pts_with_offset[:, 0] += offset_x[has_offset_mask]
pts_with_offset[:, 1] += offset_y[has_offset_mask]
if len(pts_with_offset) > 0:
    ax3.scatter(pts_with_offset[:, 0], pts_with_offset[:, 1], 
                s=0.1, c='red', alpha=0.5, label='Shifted Points')

# Points ngoài grid - màu xanh
pts_outside_grid = pts_filtered[~valid_mask]
if len(pts_outside_grid) > 0:
    ax3.scatter(pts_outside_grid[:, 0], pts_outside_grid[:, 1], 
                s=0.1, c='blue', alpha=0.5)

# Vẽ bounding boxes với linewidth nhỏ
for corners in bev_corners:
    polygon = patches.Polygon(corners, fill=False, edgecolor='orange', linewidth=0.5)
    ax3.add_patch(polygon)

ax3.set_xlim(point_cloud_range[0], point_cloud_range[3])
ax3.set_ylim(point_cloud_range[1], point_cloud_range[4])
ax3.set_aspect('equal')
ax3.set_xlabel('X (m)', fontsize=12)
ax3.set_ylabel('Y (m)', fontsize=12)
ax3.set_title('Point Cloud: Shifted (Red) vs No Offset (Blue)', fontsize=14)
ax3.legend(loc='upper right')
ax3.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('3_pointcloud_offset.png', dpi=150, bbox_inches='tight')
print("Đã lưu hình 3 vào '3_pointcloud_offset.png'")
plt.close()

# ===== THỐNG KÊ =====
print("\n===== THỐNG KÊ =====")
print(f"Heatmap max value: {heatmap_np.max():.4f}")
print(f"Heatmap min value: {heatmap_np.min():.4f}")
print(f"Offsetmap X range: [{offsetmap_np[0].min():.4f}, {offsetmap_np[0].max():.4f}]")
print(f"Offsetmap Y range: [{offsetmap_np[1].min():.4f}, {offsetmap_np[1].max():.4f}]")
print(f"Số points có offset: {np.sum(has_offset_mask)}/{len(pts_valid)}")
print(f"Grid size: {target_generator.x_grid} x {target_generator.y_grid}")

print("\nĐã tạo xong 3 hình ảnh riêng biệt!")