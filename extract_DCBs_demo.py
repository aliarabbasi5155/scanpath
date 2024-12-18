import torch
import torch.nn.functional as F
import numpy as np
import os

from PIL import Image, ImageFilter
from detectron2.config import get_cfg
from detectron2.modeling import build_backbone
from detectron2.engine import DefaultPredictor

device = torch.device("cuda")


def pred2feat(seg, info):
    seg = seg.cpu()
    feat = torch.zeros([80 + 54, 320, 512])
    for pred in info:
        mask = (seg == pred['id']).float()
        if pred['isthing']:
            feat[pred['category_id'], :, :] = mask * pred['score']
        else:
            feat[pred['category_id'] + 80, :, :] = mask
    return F.interpolate(feat.unsqueeze(0), size=[20, 32]).squeeze(0)


def get_DCBs(img_path, predictor, radius=1):
    high = Image.open(img_path).convert('RGB').resize((512, 320))
    low = high.filter(ImageFilter.GaussianBlur(radius=radius))
    high_panoptic_seg, high_segments_info = predictor(
        np.array(high))["panoptic_seg"]
    low_panoptic_seg, low_segments_info = predictor(
        np.array(low))["panoptic_seg"]
    high_feat = pred2feat(high_panoptic_seg, high_segments_info)
    low_feat = pred2feat(low_panoptic_seg, low_segments_info)
    return high_feat, low_feat


if __name__ == '__main__':
    # Load pretrained panoptic_fpn
    cfg = get_cfg()
    cfg.merge_from_file(
        './detectron2/configs/COCO-PanopticSegmentation/panoptic_fpn_R_50_3x.yaml'
    )
    cfg.MODEL.WEIGHTS = 'detectron2://COCO-PanopticSegmentation/panoptic_fpn_R_50_3x/139514569/model_final_c10459.pkl'
    model = build_backbone(cfg).to(device)
    model.eval()

    predictor = DefaultPredictor(cfg)

    # Specify directories
    img_dir = 'files/Stroop_DataSet/Stroop 512x320'
    hr_dir = 'files/Stroop_DataSet/DCBs/HR'
    lr_dir = 'files/Stroop_DataSet/DCBs/LR'
    
    # Ensure output directories exist
    os.makedirs(hr_dir, exist_ok=True)
    os.makedirs(lr_dir, exist_ok=True)

    # Process each image file in the directory recursively
    for root, dirs, files in os.walk(img_dir):
        for img_filename in files:
            # Check if the file is an image
            if img_filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(root, img_filename)

                # Get high and low resolution features
                high_feat, low_feat = get_DCBs(img_path, predictor)
                
                # Create relative path and ensure the output directories exist
                relative_path = os.path.relpath(root, img_dir)
                hr_output_dir = os.path.join(hr_dir, relative_path)
                lr_output_dir = os.path.join(lr_dir, relative_path)
                
                os.makedirs(hr_output_dir, exist_ok=True)
                os.makedirs(lr_output_dir, exist_ok=True)

                # Save features to HR and LR directories
                image_filename = f'{img_filename.replace(".png", "").replace(" ", "")}.pth.tar'
                torch.save(high_feat, os.path.join(hr_output_dir, image_filename))
                torch.save(low_feat, os.path.join(lr_output_dir, image_filename))

                print(f"Processed and saved features for {img_filename}")
