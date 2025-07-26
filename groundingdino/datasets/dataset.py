import os
import csv
import random
import torch
from collections import defaultdict
from torch.utils.data import Dataset
from groundingdino.util.train import load_image
from groundingdino.util.vl_utils import build_captions_and_token_span

class GroundingDINODataset(Dataset):
    def __init__(self, img_dir, ann_file, transforms=None, negative_sampling_rate=0.0):
        """
        Args:
            img_dir (str): Path to image directory
            ann_file (str): Path to annotation CSV
            transforms: Optional transform to be applied
            negative_sampling_rate (float): Rate of negative samples to include in captions (0.0 to 1.0)
 
        """
        self.img_dir = img_dir
        self.transforms = transforms
        self.negative_sampling_rate = negative_sampling_rate
        self.annotations = self.read_dataset(img_dir, ann_file)
        self.image_paths = list(self.annotations.keys())
        # Collect all unique categories for negative sampling
        self.all_categories = set()
        for img_path in self.annotations:
            self.all_categories.update(self.annotations[img_path]['phrases'])
        self.all_categories = list(self.all_categories)

    def read_dataset(self, img_dir, ann_file):
        """
        Read dataset annotations and convert to [x,y,w,h] format
        """
        ann_dict = defaultdict(lambda: defaultdict(list))
        with open(ann_file) as file_obj:
            ann_reader = csv.DictReader(file_obj)
            for row in ann_reader:
                img_path = os.path.join(img_dir, row['image_name'])
                # Store in [x,y,w,h] format directly
                x = int(row['bbox_x'])
                y = int(row['bbox_y'])
                w = int(row['bbox_width'])
                h = int(row['bbox_height'])
                
                # Convert to center format [cx,cy,w,h]
                cx = x + w/2
                cy = y + h/2
                ann_dict[img_path]['boxes'].append([cx, cy, w, h])
                ann_dict[img_path]['phrases'].append(row['label_name'])
        return ann_dict
    
    def sample_negative_categories(self, positive_categories, num_negative=None):
        """
        Sample negative categories that are not present in the current image
        """
        if num_negative is None:
            # Default to same number as positive categories or 1, whichever is larger
            num_negative = max(1, len(positive_categories))
        
        # Get candidates that are not in positive categories
        candidates = [cat for cat in self.all_categories if cat not in positive_categories]
        
        # Handle case where there are no candidates
        if not candidates:
            # If no candidates, return empty list or duplicate some categories
            return []
        
        # Sample negative categories
        if len(candidates) >= num_negative:
            negative_categories = random.sample(candidates, num_negative)
        else:
            # If not enough candidates, sample with replacement
            negative_categories = random.choices(candidates, k=num_negative)
            
        return negative_categories

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        # Load and transform image
        image_source, image = load_image(img_path)
        h, w = image_source.shape[0:2]  

        boxes = torch.tensor(self.annotations[img_path]['boxes'], dtype=torch.float32)
        str_cls_lst = self.annotations[img_path]['phrases']
        
        # Sample negative categories if needed
        if self.negative_sampling_rate > 0 and len(self.all_categories) > len(str_cls_lst):
            # Determine number of negative samples based on rate
            num_negative = max(1, int(len(str_cls_lst) * self.negative_sampling_rate))
            negative_categories = self.sample_negative_categories(str_cls_lst, num_negative)
            # Combine positive and negative categories
            combined_categories = str_cls_lst + negative_categories
        else:
            combined_categories = str_cls_lst
        
        # Create caption mapping and format
        caption_dict = {item: idx for idx, item in enumerate(str_cls_lst)}  # Only for positive categories
        captions, cat2tokenspan = build_captions_and_token_span(combined_categories, force_lowercase=True)
        # Labels for positive categories only
        classes = torch.tensor([caption_dict[p] for p in str_cls_lst], dtype=torch.int64)

        target = {
            'boxes': boxes,  # Already in [cx,cy,w,h] format
            'size': torch.as_tensor([int(h), int(w)]),
            'orig_img': image_source,  
            'str_cls_lst': str_cls_lst,  # Positive categories only
            'all_categories': combined_categories,  # Positive + negative categories
            'caption': captions,
            'labels': classes, 
            'cat2tokenspan': cat2tokenspan
        }

        return image, target