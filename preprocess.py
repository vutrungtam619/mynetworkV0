import argparse
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
from configs import config
from concurrent.futures import ProcessPoolExecutor, as_completed
from utils import (
    read_calib,
    read_points,
    read_label,
    write_pickle,
    write_points,
    remove_out_range_points,
    get_points_num_in_bbox,
)

root = Path(__file__).parent

def remove_min_points_bboxes(points, annotation_dict, calib_dict, min_points=10):
    points_num = get_points_num_in_bbox(
        points=points,
        r0_rect=calib_dict['r0_rect'],
        tr_velo_to_cam=calib_dict['tr_velo_to_cam'],
        dimensions=annotation_dict['dimensions'],
        locations=annotation_dict['locations'],
        rotation_y=annotation_dict['rotation_y'],
        names=annotation_dict['names']
    )

    keep_mask = points_num >= min_points

    filtered_dict = {}
    for key, val in annotation_dict.items():
        if isinstance(val, np.ndarray) and val.shape[0] == keep_mask.shape[0]:
            filtered_dict[key] = val[keep_mask]
        else:
            filtered_dict[key] = val

    return filtered_dict

def process_one_idx(idx, data_root, split, label, lidar_reduced_folder):
    cur_info_dict = {}

    image_path = Path(data_root) / split / 'image_2' / f'{idx}.jpg'
    lidar_path = Path(data_root) / split / 'velodyne' / f'{idx}.bin'
    calib_path = Path(data_root) / split / 'camera.json'
    
    image = cv2.imread(str(image_path))
    image_shape = image.shape[:2]
    cur_info_dict['image'] = {
        'image_shape': image_shape,
        'image_path': Path(*image_path.parts[-3:])
    }

    calib_dict = read_calib(calib_path)
    cur_info_dict['calib'] = calib_dict
    
    # read points 
    lidar_points = read_points(lidar_path)
    
    # reduce points out of point cloud range for lower the size of file
    reduced_points = remove_out_range_points(pts=lidar_points, point_range=config['point_cloud_range'])
    lidar_reduced_path = Path(lidar_reduced_folder) / f'{idx}.bin'
    write_points(lidar_reduced_path, reduced_points)

    cur_info_dict['lidar'] = {
        'lidar_total': reduced_points.shape[0],
        'lidar_path': lidar_reduced_path,
    }
     
    if label:
        label_path = Path(data_root) / split / 'label_2' / f'{idx}.txt'
        annotation_dict = read_label(label_path)
        # remove bboxes that have minimum points
        annotation_dict = remove_min_points_bboxes(
            points=reduced_points,
            annotation_dict=annotation_dict,
            calib_dict=calib_dict,
            min_points=10
        )  
        cur_info_dict['annos'] = annotation_dict

    return int(idx), cur_info_dict
        
   
def create_data_info_pkl(data_root, data_type, label):
    print(f"Processing {data_type} data into pkl file....")

    # read id (000000, 000001, ...) in index file
    index_files = Path(root) / 'index' / f'{data_type}.txt'
    ids = index_files.read_text(encoding="utf-8").splitlines()
    
    # create split
    if data_type == 'train':
        split = 'training'
    else:
        split = 'validating'
    
    # create folder for reduced lidar
    lidar_reduced_folder = Path(root) / 'datasets' / 'point_cloud_reduced' / f'{split}'
    lidar_reduced_folder.mkdir(parents=True, exist_ok=True)
    
    # information about split
    info_dict = {}
    
    # process for each id
    with ProcessPoolExecutor() as executor:
        futures = {
            executor.submit(process_one_idx, idx, data_root, split, label, lidar_reduced_folder): idx
            for idx in ids
        }
        for future in tqdm(as_completed(futures), total=len(futures)):
            idx, cur_info_dict = future.result()
            # each idx have a dict inlcude: image, calib, lidar, annos
            info_dict[idx] = cur_info_dict

    save_pkl_path = Path(root) / 'datasets' / f'infos_{data_type}.pkl'
    write_pickle(save_pkl_path, info_dict)
    
    return None      

def main(args):
    data_root = args.data_root
    # create train information
    create_data_info_pkl(data_root, data_type='train', label=True)
    # create val information
    create_data_info_pkl(data_root, data_type='val', label=True)
    print("......Processing finished!!!")  
  
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Dataset information')
    parser.add_argument('--data_root', default=config['data_root'])
    args = parser.parse_args()
    main(args)