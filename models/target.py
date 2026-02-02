import numpy as np
import torch
import cv2
from utils import bbox3d2bevcorners

class Target:
    def __init__(self, point_cloud_range, voxel_size):
        self.vx, self.vy = voxel_size[:2]
        self.x_offset = self.vx / 2 + point_cloud_range[0]
        self.y_offset = self.vy / 2 + point_cloud_range[1]
        self.x_grid = int((point_cloud_range[3] - point_cloud_range[0]) / self.vx)
        self.y_grid = int((point_cloud_range[4] - point_cloud_range[1]) / self.vy)
        self._gaussian_cache = {}

    def _get_gaussian_kernel(self, r, sigma, device):
        key = (r, sigma, device)
        if key not in self._gaussian_cache:
            d = 2 * r + 1
            m = (d - 1) / 2
            y, x = torch.meshgrid(
                torch.arange(-m, m + 1, device=device),
                torch.arange(-m, m + 1, device=device),
                indexing="ij",
            )
            self._gaussian_cache[key] = torch.exp(-(x**2 + y**2) / (2 * sigma**2))
        return self._gaussian_cache[key]

    def get_heatmap(self, batch_gt_bboxes, batch_size, batched_gt_numpoints, device, sigma0=1.5, alpha=1.0):
        r = max(1, int(min(0.6 / self.vx, 0.6 / self.vy) / 2))
        heatmap = torch.zeros((batch_size, 1, self.y_grid, self.x_grid), device=device)
        for b in range(batch_size):
            gt = batch_gt_bboxes[b]
            if gt.numel() == 0: continue
            centers = gt[:, :2].to(device)
            cx = ((centers[:, 0] - self.x_offset) / self.vx).long()
            cy = ((centers[:, 1] - self.y_offset) / self.vy).long()
            valid = (cx >= 0) & (cx < self.x_grid) & (cy >= 0) & (cy < self.y_grid)
            if not valid.any(): continue
            cx = cx[valid]; cy = cy[valid]
            num_points = batched_gt_numpoints[b]
            if not torch.is_tensor(num_points): num_points = torch.from_numpy(num_points)
            num_points = num_points.to(device)[valid]
            
            for i in range(cx.shape[0]):
                x = int(cx[i]); y = int(cy[i])
                rho_hat = torch.log1p(num_points[i] / 0.36)
                sigma_b = sigma0 / (1.0 + alpha * rho_hat)
                gaussian = self._get_gaussian_kernel(r, float(sigma_b), device)
                lx = max(0, x - r); rx = min(self.x_grid, x + r + 1)
                ty = max(0, y - r); by = min(self.y_grid, y + r + 1)
                gl = r - (x - lx); gr = r + (rx - x)
                gt_ = r - (y - ty); gb = r + (by - y)
                heatmap[b, 0, ty:by, lx:rx] = torch.maximum(heatmap[b, 0, ty:by, lx:rx], gaussian[gt_:gb, gl:gr])
                
        return heatmap #(B, 1, H, W)

    def get_offsetmap(self, batch_gt_bboxes, batch_size, device):
        offset = torch.zeros((batch_size, 2, self.y_grid, self.x_grid), device=device)
        offset_mask = torch.zeros((batch_size, 1, self.y_grid, self.x_grid), device=device, dtype=torch.bool)
        dist_map = torch.full((batch_size, self.y_grid, self.x_grid), float("inf"), device=device)

        for b in range(batch_size):
            gt = batch_gt_bboxes[b]
            if len(gt) == 0:
                continue

            gt_np = gt.cpu().numpy() if torch.is_tensor(gt) else gt
            centers = gt[:, :2].to(device) if torch.is_tensor(gt) else torch.from_numpy(gt_np[:, :2]).to(device)
            polys = bbox3d2bevcorners(gt_np)

            for poly, center in zip(polys, centers):
                min_x, min_y = poly.min(0)
                max_x, max_y = poly.max(0)

                x0 = max(0, int((min_x - self.x_offset) / self.vx))
                x1 = min(self.x_grid, int((max_x - self.x_offset) / self.vx) + 1)
                y0 = max(0, int((min_y - self.y_offset) / self.vy))
                y1 = min(self.y_grid, int((max_y - self.y_offset) / self.vy) + 1)
                if x0 >= x1 or y0 >= y1:
                    continue

                xs = torch.arange(x0, x1, device=device) * self.vx + self.x_offset
                ys = torch.arange(y0, y1, device=device) * self.vy + self.y_offset
                gy, gx = torch.meshgrid(ys, xs, indexing="ij")

                px = (poly[:, 0] - self.x_offset) / self.vx - x0
                py = (poly[:, 1] - self.y_offset) / self.vy - y0
                mask_np = np.zeros((y1 - y0, x1 - x0), np.uint8)
                cv2.fillPoly(mask_np, [np.stack([px, py], 1).astype(np.int32)], 1)
                mask = torch.from_numpy(mask_np).to(device, dtype=torch.bool)

                dx = center[0] - gx
                dy = center[1] - gy
                dist = dx**2 + dy**2

                d_old = dist_map[b, y0:y1, x0:x1]
                upd = mask & (dist < d_old)

                if upd.any():
                    offset[b, 0, y0:y1, x0:x1][upd] = dx[upd]
                    offset[b, 1, y0:y1, x0:x1][upd] = dy[upd]
                    offset_mask[b, 0, y0:y1, x0:x1][upd] = True
                    d_old[upd] = dist[upd]

        return offset, offset_mask # (B, 2, H, W)
