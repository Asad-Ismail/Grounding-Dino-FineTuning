import sys
import os
import unittest
import torch
from torch.utils.data import DataLoader
from groundingdino.datasets.dataset import GroundingDINODataset

class TestTrainingIntegration(unittest.TestCase):
    
    def test_data_loader_with_negative_sampling(self):
        """Test that DataLoader works correctly with negative sampling"""
        # Create dataset with negative sampling
        dataset = GroundingDINODataset(
            'multimodal-data/fashion_dataset_subset/images/train',
            'multimodal-data/fashion_dataset_subset/train_annotations.csv',
            negative_sampling_rate=0.5  # 50% negative sampling
        )
        
        # Create data loader
        dataloader = DataLoader(
            dataset,
            batch_size=2,
            shuffle=True,
            num_workers=1,
            collate_fn=lambda x: tuple(zip(*x))
        )
        
        # Get a batch
        for images, targets in dataloader:
            # Check batch size
            self.assertEqual(len(images), 2)
            self.assertEqual(len(targets), 2)
            
            # Check that targets have the expected structure
            for target in targets:
                self.assertIn('str_cls_lst', target)
                self.assertIn('all_categories', target)
                self.assertGreaterEqual(len(target['all_categories']), len(target['str_cls_lst']))
            
            # If we got here, the test passes
            break
    
    def test_dataset_without_negative_sampling(self):
        """Test that dataset works correctly without negative sampling"""
        # Create dataset without negative sampling
        dataset = GroundingDINODataset(
            'multimodal-data/fashion_dataset_subset/images/train',
            'multimodal-data/fashion_dataset_subset/train_annotations.csv',
            negative_sampling_rate=0.0  # No negative sampling
        )
        
        # Get a sample
        image, target = dataset[0]
        
        # Check that all_categories equals str_cls_lst when no negative sampling
        self.assertEqual(target['all_categories'], target['str_cls_lst'])

if __name__ == '__main__':
    unittest.main()