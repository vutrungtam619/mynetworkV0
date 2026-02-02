import numpy as np
from utils import limit_period, remove_outside_points, remove_outside_bboxes
    
def global_rot_scale_trans(data_dict, rot_range, scale_ratio_range, translation_std, ratio):
    if np.random.rand() < ratio:
        pts = data_dict['pts']
        gt_bboxes_3d = data_dict['gt_bboxes_3d']
        
        # rotation
        rot_angle = np.random.uniform(rot_range[0], rot_range[1])
        rot_cos, rot_sin = np.cos(rot_angle), np.sin(rot_angle)
        rot_mat = np.array([[rot_cos, rot_sin], [-rot_sin, rot_cos]])
        
        gt_bboxes_3d[:, :2] = gt_bboxes_3d[:, :2] @ rot_mat.T
        gt_bboxes_3d[:, 6] += rot_angle
        pts[:, :2] = pts[:, :2] @ rot_mat.T
        
        # scale
        scale_factor = np.random.uniform(scale_ratio_range[0], scale_ratio_range[1])
        gt_bboxes_3d[:, :6] *= scale_factor
        pts[:, :3] *= scale_factor
        
        # translate
        trans_factor = np.random.normal(scale=translation_std, size=(1, 3))
        gt_bboxes_3d[:, :3] += trans_factor
        pts[:, :3] += trans_factor
        
        data_dict.update({
            'pts': pts, 
            'gt_bboxes_3d': gt_bboxes_3d
        })
    
    return data_dict

def random_flip(data_dict, ratio):
    if np.random.rand() < ratio:
        pts = data_dict['pts']
        gt_bboxes_3d = data_dict['gt_bboxes_3d']
        # Flip along Y axis
        pts[:, 1] = -pts[:, 1]
        gt_bboxes_3d[:, 1] = -gt_bboxes_3d[:, 1]
        gt_bboxes_3d[:, 6] = -gt_bboxes_3d[:, 6] + np.pi

        data_dict.update({
            'pts': pts, 
            'gt_bboxes_3d': gt_bboxes_3d
        })
    
    return data_dict

def points_range_filter(data_dict, point_cloud_range):
    pts = data_dict['pts']
    keep_mask = np.all([
        pts[:, 0] > point_cloud_range[0], pts[:, 0] < point_cloud_range[3],
        pts[:, 1] > point_cloud_range[1], pts[:, 1] < point_cloud_range[4],
        pts[:, 2] > point_cloud_range[2], pts[:, 2] < point_cloud_range[5]
    ], axis=0)
    
    data_dict.update({
        'pts': pts[keep_mask]
    })
    
    return data_dict

def bboxes_range_filter(data_dict, point_cloud_range):
    gt_bboxes_3d = data_dict['gt_bboxes_3d']
    gt_numpoints = data_dict.get('gt_numpoints', None)

    keep_mask = np.all([
        gt_bboxes_3d[:, 0] >= point_cloud_range[0],
        gt_bboxes_3d[:, 0] <= point_cloud_range[3],
        gt_bboxes_3d[:, 1] >= point_cloud_range[1],
        gt_bboxes_3d[:, 1] <= point_cloud_range[4],
    ], axis=0)

    gt_bboxes_3d = gt_bboxes_3d[keep_mask]
    gt_bboxes_3d[:, 6] = limit_period(gt_bboxes_3d[:, 6], 0.5, 2 * np.pi)

    data_dict['gt_bboxes_3d'] = gt_bboxes_3d

    if gt_numpoints is not None:
        data_dict['gt_numpoints'] = gt_numpoints[keep_mask]

    return data_dict

def points_frustum_filter(data_dict):
    calib = data_dict['calib_info']
    points = data_dict['pts']
    points = remove_outside_points(
        points=points, 
        r0_rect=calib['r0_rect'], 
        tr_velo_to_cam=calib['tr_velo_to_cam'], 
        P2=calib['P0'], 
        image_shape=data_dict['image_info']['image_shape']
    )
    data_dict.update({
        'pts': points
    })
    
    return data_dict   

def bboxes_frustum_filter(data_dict):
    bboxes = data_dict['gt_bboxes_3d']
    gt_numpoints = data_dict.get('gt_numpoints', None)
    calib = data_dict['calib_info']
    bboxes, keep_mask = remove_outside_bboxes(
        bboxes=bboxes,
        r0_rect=calib['r0_rect'],
        tr_velo_to_cam=calib['tr_velo_to_cam'],
        P2=calib['P0'],
        image_shape=data_dict['image_info']['image_shape'],
        return_mask=True
    )
    data_dict['gt_bboxes_3d'] = bboxes
    if gt_numpoints is not None:
        data_dict['gt_numpoints'] = gt_numpoints[keep_mask]

    return data_dict

def points_shuffle(data_dict):
    pts = data_dict['pts']
    indices = np.arange(0, len(pts))
    np.random.shuffle(indices)
    pts = pts[indices]
    
    data_dict.update({
        'pts': pts
    })
    
    return data_dict

def train_data_aug(data_dict, data_aug_config):    
    global_rot_scale_trans_cfg = data_aug_config['global_rot_scale_trans']
    data_dict = global_rot_scale_trans(
        data_dict=data_dict,
        rot_range=global_rot_scale_trans_cfg['rot_range'],
        scale_ratio_range=global_rot_scale_trans_cfg['scale_ratio_range'],
        translation_std=global_rot_scale_trans_cfg['translation_std'],
        ratio=0.5,
    )

    data_dict = random_flip(
        data_dict=data_dict,
        ratio=0.5,
    )
    
    data_dict = points_range_filter(
        data_dict=data_dict,
        point_cloud_range=data_aug_config['point_cloud_range'],
    )
    
    data_dict = bboxes_range_filter(
        data_dict=data_dict,
        point_cloud_range=data_aug_config['point_cloud_range'],
    )
    
    data_dict = points_frustum_filter(
        data_dict=data_dict,
    )
    
    data_dict = bboxes_frustum_filter(
        data_dict=data_dict
    )
    
    data_dict = points_shuffle(
        data_dict=data_dict,
    )   
    
    return data_dict

def val_data_aug(data_dict, data_aug_config):
    data_dict = points_range_filter(
        data_dict=data_dict,
        point_cloud_range=data_aug_config['point_cloud_range'],
    )
    
    data_dict = bboxes_range_filter(
        data_dict=data_dict,
        point_cloud_range=data_aug_config['point_cloud_range'],
    )
    
    data_dict = points_frustum_filter(
        data_dict=data_dict,
    )
    
    data_dict = bboxes_frustum_filter(
        data_dict=data_dict
    )
    
    return data_dict