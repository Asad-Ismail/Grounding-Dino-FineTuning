import sys
import os
import unittest
import torch
from groundingdino.datasets.dataset import GroundingDINODataset

class TestNegativeSampling(unittest.TestCase):
    
    def setUp(self):
        self.dataset = GroundingDINODataset(
            'multimodal-data/fashion_dataset_subset/images/train',
            'multimodal-data/fashion_dataset_subset/train_annotations.csv',
            negative_sampling_rate=1.0  # 100% negative sampling
        )
    
    def test_dataset_initialization(self):
        """Test that dataset initializes correctly with negative sampling"""
        self.assertGreater(len(self.dataset.all_categories), 0)
        self.assertEqual(self.dataset.negative_sampling_rate, 1.0)
    
    def test_negative_sampling_function(self):
        """Test the sample_negative_categories function"""
        # Get a sample
        image, target = self.dataset[0]
        positive_cats = target['str_cls_lst']
        
        # Sample negative categories
        negative_cats = self.dataset.sample_negative_categories(positive_cats, 2)
        
        # Verify no overlap (if we have negative categories)
        if negative_cats:
            for cat in negative_cats:
                self.assertNotIn(cat, positive_cats)
    
    def test_get_item_with_negative_sampling(self):
        """Test that __getitem__ correctly handles negative sampling"""
        image, target = self.dataset[0]
        
        # Check that we have the expected keys
        expected_keys = {'boxes', 'size', 'orig_img', 'str_cls_lst', 'all_categories', 'caption', 'labels', 'cat2tokenspan'}
        self.assertEqual(set(target.keys()), expected_keys)
        
        # Check that all_categories contains at least as many items as str_cls_lst
        self.assertGreaterEqual(len(target['all_categories']), len(target['str_cls_lst']))
        
        # Check that positive categories are a subset of all categories
        for cat in target['str_cls_lst']:
            self.assertIn(cat, target['all_categories'])

if __name__ == '__main__':
    unittest.main()