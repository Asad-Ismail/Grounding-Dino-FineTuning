"""
Demo script to show negative sampling in action
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '.'))

from groundingdino.datasets.dataset import GroundingDINODataset
from torch.utils.data import DataLoader

def demo_negative_sampling():
    print("=== Negative Sampling Demo ===")
    
    # Create dataset with different negative sampling rates
    dataset_no_neg = GroundingDINODataset(
        'multimodal-data/fashion_dataset_subset/images/train',
        'multimodal-data/fashion_dataset_subset/train_annotations.csv',
        negative_sampling_rate=0.0  # No negative sampling
    )
    
    dataset_some_neg = GroundingDINODataset(
        'multimodal-data/fashion_dataset_subset/images/train',
        'multimodal-data/fashion_dataset_subset/train_annotations.csv',
        negative_sampling_rate=0.5  # 50% negative sampling
    )
    
    dataset_full_neg = GroundingDINODataset(
        'multimodal-data/fashion_dataset_subset/images/train',
        'multimodal-data/fashion_dataset_subset/train_annotations.csv',
        negative_sampling_rate=1.0  # 100% negative sampling
    )
    
    # Compare samples from each dataset
    for i, (name, dataset) in enumerate([
        ("No negative sampling", dataset_no_neg),
        ("50% negative sampling", dataset_some_neg),
        ("100% negative sampling", dataset_full_neg)
    ]):
        print(f"\n--- {name} ---")
        image, target = dataset[0]
        print(f"Positive categories: {target['str_cls_lst']}")
        print(f"All categories: {target['all_categories']}")
        print(f"Number of positive categories: {len(target['str_cls_lst'])}")
        print(f"Number of all categories: {len(target['all_categories'])}")
        print(f"Increase due to negative sampling: {len(target['all_categories']) - len(target['str_cls_lst'])}")

if __name__ == "__main__":
    demo_negative_sampling()