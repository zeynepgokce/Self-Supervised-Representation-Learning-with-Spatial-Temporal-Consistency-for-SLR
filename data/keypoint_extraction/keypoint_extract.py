import os
import cv2
import numpy as np
import pickle as pkl
from tqdm import tqdm
from mmpose.apis.inference import inference_topdown
from mmpose.apis import init_model
# 🔧 MODEL AYARLARI
config_file = './td-hm_hrnet-w48_dark-8xb32-210e_coco-wholebody-384x288.py'
checkpoint_file = './hrnet_w48_coco_wholebody_384x288_dark-f5726563_20200918.pth'
device = "cuda" #'cuda:0'  # veya 'cpu'

# 🔧 Modeli başlat
model = init_model(config_file, checkpoint_file, device=device)
def extract_pose_from_frames(frame_dir):
    frame_files = sorted([f for f in os.listdir(frame_dir) if f.endswith(('.jpg', '.bmp', '.png'))])
    keypoints = []
    img_list = []

    for fname in frame_files:
        img_path = os.path.join(frame_dir, fname)

        result = inference_topdown(model, img_path)

        if len(result) == 0 or not hasattr(result[0], 'pred_instances'):
            #keypoints.append([])
            keypoints.append(np.zeros((133, 3)))
        else:

            kps = result[0].pred_instances.keypoints
            score = result[0].pred_instances.keypoint_scores  # (1, 133)
            if len(kps) > 0 and len(score) > 0:
                kps_full = np.concatenate([kps[0], score[0][:, None]], axis=1)  # (133, 3)
                keypoints.append(kps_full)
            else:
                keypoints.append(np.zeros((133, 3)))  # boşsa 0'larla doldur
        img_list.append(fname)

    return {'keypoints': keypoints, 'img_list': img_list}

def run_pose_extraction(split_file, frame_root, save_root):
    os.makedirs(save_root, exist_ok=True)
    with open(split_file, 'r') as f:
        video_names = [line.strip().split()[0] for line in f.readlines()]
    for video_name in tqdm(video_names, desc='Extracting poses'):
        vid_id = video_name.split('.')[0]
        frame_dir = os.path.join(frame_root, vid_id)
        if not os.path.exists(frame_dir):
            print(f"Frame path not found: {frame_dir}")
            continue
        pose_dict = extract_pose_from_frames(frame_dir)
        save_path = os.path.join(save_root, vid_id + '.pkl')
        with open(save_path, 'wb') as f:
            pkl.dump(pose_dict, f)

# 🔧 Örnek kullanımı
split_file = '../WLASL/WLASL100_64x64/val.txt'
frame_root = '/home/zeynep/Thesis/datasets/wlasl100_64x64_640x480_PIL/val'
save_root = '../WLASL/WLASL100_64x64/Keypoints_2d_mmpose/val'

run_pose_extraction(split_file, frame_root, save_root)