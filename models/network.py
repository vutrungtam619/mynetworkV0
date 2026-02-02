import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from configs import config
from packages import Voxelization
from .target import Target

class PillarLayer(nn.Module):
    def __init__(self, voxel_size, point_cloud_range, max_points, max_voxels):
        super(PillarLayer, self).__init__()
        self.voxel_layer = Voxelization(voxel_size=voxel_size, point_cloud_range=point_cloud_range, max_num_points=max_points, max_voxels=max_voxels)

    @torch.no_grad()
    def forward(self, batched_pts):
        ''' Generate pillar from points
        Args:
            batched_pts [list torch.tensor float32, (N, 4)]: list of batch points, each batch have shape (N, 4)
                        
        Returns:
            pillars [torch.tensor float32, (p1 + p2 + ... + pb, N, c)]: c is number of features per point
            coors_batch [torch.tensor int64, (p1 + p2 + ... + pb, 1 + 3)]: coordinate of each pillar, 1 is the batch index
            num_points_per_pillar [torch.tensor int64, (p1 + p2 + ... + pb, )]: number of points in each pillar
        '''
        pillars, coors, npoints_per_pillar = [], [], []
        for i, pts in enumerate(batched_pts):
            voxels_out, coors_out, num_points_per_voxel_out = self.voxel_layer(pts) 
            # voxels_out: (max_voxel, num_points, c), coors_out: (max_voxel, 3)
            # num_points_per_voxel_out: (max_voxel, )
            pillars.append(voxels_out)
            coors.append(coors_out.long())
            npoints_per_pillar.append(num_points_per_voxel_out)
        
        pillars = torch.cat(pillars, dim=0) # (p1 + p2 + ... + pb, num_points, c)
        npoints_per_pillar = torch.cat(npoints_per_pillar, dim=0) # (p1 + p2 + ... + pb, )
        coors_batch = []
        for i, cur_coors in enumerate(coors):
            coors_batch.append(F.pad(cur_coors, (1, 0), value=i))
        coors_batch = torch.cat(coors_batch, dim=0) # (p1 + p2 + ... + pb, 1 + 3)

        return pillars, coors_batch, npoints_per_pillar
    
class PillarEncoder(nn.Module):
    def __init__(self, voxel_size, point_cloud_range, out_channels):
        super(PillarEncoder, self).__init__()
        self.out_channels = out_channels
        self.vx, self.vy = voxel_size[0], voxel_size[1]
        self.x_offset = voxel_size[0] / 2 + point_cloud_range[0]
        self.y_offset = voxel_size[1] / 2 + point_cloud_range[1]
        self.x_grid = int((point_cloud_range[3] - point_cloud_range[0]) / voxel_size[0])
        self.y_grid = int((point_cloud_range[4] - point_cloud_range[1]) / voxel_size[1])
        
        self.point_encoder = nn.Sequential(
            nn.Conv1d(6, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(out_channels),          
        )

    def forward(self, pillars, coors_batch, npoints_per_pillar, batch_size):
        ''' Encode pillars into BEV feature map 
        Args:
            pillars [torch.tensor float32, (p1 + p2 + ... + pb, N, c)]: c is number of features per point
            coors_batch [torch.tensor int64, (p1 + p2 + ... + pb, 1 + 3)]: coordinate of each pillar, 1 is the batch index
            num_points_per_pillar [torch.tensor int64, (p1 + p2 + ... + pb, )]: number of points in each pillar
            
        Returns: 
            
        '''
        device = pillars.device
        # 1. calculate offset to the points center (in each pillar)
        offset_pt_center = pillars[:, :, :3] - torch.sum(pillars[:, :, :3], dim=1, keepdim=True) / npoints_per_pillar[:, None, None] # (p1 + p2 + ... + pb, num_points, 3)

        # 2. calculate offset to the pillar center
        x_offset_pi_center = pillars[:, :, 0:1] - (coors_batch[:, None, 1:2] * self.vx + self.x_offset) # (p1 + p2 + ... + pb, num_points, 1)
        y_offset_pi_center = pillars[:, :, 1:2] - (coors_batch[:, None, 2:3] * self.vy + self.y_offset) # (p1 + p2 + ... + pb, num_points, 1)

        # 3. encoder
        features = torch.cat([pillars[:, :, 2:3], offset_pt_center, x_offset_pi_center, y_offset_pi_center], dim=-1) # (p1 + p2 + ... + pb, num_points, 6)

        # 4. find mask for (0, 0, 0) and update the encoded features
        # a very beautiful implementation
        voxel_ids = torch.arange(0, pillars.size(1)).to(device) # (num_points, )
        mask = voxel_ids[:, None] < npoints_per_pillar[None, :] # (num_points, p1 + p2 + ... + pb)
        mask = mask.permute(1, 0).contiguous()  # (p1 + p2 + ... + pb, num_points)
        features *= mask[:, :, None]

        # 5. embedding
        features = features.permute(0, 2, 1).contiguous() # (p1 + p2 + ... + pb, 6, num_points)
        features = self.point_encoder(features)  # (p1 + p2 + ... + pb, out_channels, num_points)
        pooling_features = torch.max(features, dim=-1)[0] # (p1 + p2 + ... + pb, out_channels)

        # 6. pillar scatter
        batched_canvas = []
        for i in range(batch_size):
            cur_coors_idx = coors_batch[:, 0] == i
            cur_coors = coors_batch[cur_coors_idx, :]
            cur_features = pooling_features[cur_coors_idx]

            canvas = torch.zeros((self.x_grid, self.y_grid, self.out_channels), dtype=torch.float32, device=device)
            canvas[cur_coors[:, 1], cur_coors[:, 2]] = cur_features
            canvas = canvas.permute(2, 1, 0).contiguous()
            batched_canvas.append(canvas)
        pseudo_image = torch.stack(batched_canvas, dim=0) # (B, C, H, W)
        return pseudo_image
    
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=True) 

        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True) 
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        identity = self.shortcut(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out) 

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity       
        out = self.relu2(out) 
        return out

class Backbone(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Backbone, self).__init__()
        
        self.enc0 = nn.Sequential(
            ResidualBlock(in_channels, in_channels),
            ResidualBlock(in_channels, in_channels),
        )
        
        self.enc1 = nn.Sequential(
            ResidualBlock(in_channels, in_channels*2, stride=2),
            ResidualBlock(in_channels*2, in_channels*2),
        )
        
        self.enc2 = nn.Sequential(
            ResidualBlock(in_channels*2, in_channels*4, stride=2),
            ResidualBlock(in_channels*4, in_channels*4),
        )
        
        self.up1 = nn.ConvTranspose2d(in_channels*4, in_channels*2, kernel_size=2, stride=2)
        self.dec1 = nn.Sequential(
            ResidualBlock(in_channels*4, in_channels*2),
            ResidualBlock(in_channels*2, in_channels*2),
        )
        
        self.up2 = nn.ConvTranspose2d(in_channels*2, in_channels, kernel_size=2, stride=2)
        self.dec2 = nn.Sequential(
            ResidualBlock(in_channels*2, in_channels),
            ResidualBlock(in_channels, in_channels),
        )

        self.out = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        
    def forward(self, x):
        # Encoder path
        x0 = self.enc0(x)   # (B, C, H, W)
        x1 = self.enc1(x0)  # (B, 2C, H/2, W/2)
        x2 = self.enc2(x1)  # (B, 4C, H/4, W/4)
        
        # Decoder path with skip connections
        u1 = self.up1(x2)                 # (B, 2C, H/2, W/2)
        c1 = torch.cat([u1, x1], dim=1)   # (B, 4C, H/2, W/2)
        d1 = self.dec1(c1)                # (B, 2C, H/2, W/2)
        
        u2 = self.up2(d1)                 # (B, C, H, W)
        c2 = torch.cat([u2, x0], dim=1)   # (B, 2C, H, W)
        d2 = self.dec2(c2)                # (B, C, H, W)
        
        out = self.out(d2)                # (B, out_channels, H, W)
        
        return out
        
        
class Head(nn.Module):
    def __init__(self, in_channel, n_classes):
        super(Head, self).__init__()
        
        self.conv_heat = nn.Conv2d(in_channel, n_classes, 1)
        self.conv_offset = nn.Conv2d(in_channel, n_classes*2, 1)
        
        # in consitent with mmdet3d
        conv_layer_id = 0
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0, std=0.01)
                if conv_layer_id == 0:
                    prior_prob = 0.01
                    bias_init = float(-np.log((1 - prior_prob) / prior_prob))
                    nn.init.constant_(m.bias, bias_init)
                else:
                    nn.init.constant_(m.bias, 0)
                conv_layer_id += 1
        
    def forward(self, x):
        heat_map = self.conv_heat(x)
        offset_map = self.conv_offset(x)
        
        return heat_map, offset_map      

class MyNet(nn.Module):
    def __init__(
        self,
        nclasses=config['num_classes'], 
        voxel_size=config['voxel_size'],
        point_cloud_range=config['point_cloud_range'],
        max_points=config['max_points'],
        max_voxels=config['max_voxels'],
    ):
        super(MyNet, self).__init__()
        
        self.nclasses = nclasses
        
        self.pillar_layer = PillarLayer(
            voxel_size=voxel_size,
            point_cloud_range=point_cloud_range,
            max_points=max_points,
            max_voxels=max_voxels,
        )
        
        self.pillar_encoder = PillarEncoder(
            voxel_size=voxel_size,
            point_cloud_range=point_cloud_range,
            out_channels=64,
        )
        
        self.backbone = Backbone(
            in_channels=64, 
            out_channels=128,
        )
        
        self.head = Head(
            in_channel=128, 
            n_classes=nclasses
        )
        
    def forward(self, mode, batched_pts, batched_gt_bboxes, batched_gt_numpoints):
        batch_size = len(batched_pts)
        
        pillars, coors_batch, npoints_per_pillar = self.pillar_layer(batched_pts)
        
        pillar_features = self.pillar_encoder(pillars, coors_batch, npoints_per_pillar, batch_size) # (B, C, H, W)
        
        features_map = self.backbone(pillar_features) # (B, 2C, H, W)
        
        heatmap, offsetmap = self.head(features_map) # (B, 1, H, W) (B, 2, H, W)
        
        device = heatmap.device
        
        if mode == 'train' or mode == 'val':   
            target_generator = Target(point_cloud_range=config['point_cloud_range'], voxel_size=config['voxel_size'])
            gt_heatmap = target_generator.get_heatmap(
                batch_gt_bboxes=batched_gt_bboxes,
                batch_size=batch_size,
                batched_gt_numpoints = batched_gt_numpoints,
                device=device
            )
            gt_offsetmap, gt_offsetmask = target_generator.get_offsetmap(
                batch_gt_bboxes=batched_gt_bboxes,
                batch_size=batch_size,
                device=device                
            )
            return {
                'pred_heatmap': heatmap,
                'pred_offsetmap': offsetmap,
                'gt_heatmap': gt_heatmap,
                'gt_offsetmap': gt_offsetmap,   
                'gt_offsetmask': gt_offsetmask,            
            }
        
        if mode == 'test':
            return {
                'pred_heatmap': heatmap,
                'pred_offsetmap': offsetmap,                
            }
        else:
            raise ValueError 