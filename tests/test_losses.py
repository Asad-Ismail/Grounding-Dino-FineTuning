import sys
import os
import unittest
import torch
from groundingdino.util.class_loss import BCEWithLogitsLoss, MultilabelFocalLoss

class TestLossFunctions(unittest.TestCase):
    
    def setUp(self):
        self.bce_loss = BCEWithLogitsLoss(eos_coef=0.1)
        self.focal_loss = MultilabelFocalLoss(eos_coef=0.1)
    
    def test_bce_loss_with_negatives(self):
        """Test BCEWithLogitsLoss with both positive and negative examples"""
        # Create mock predictions and targets
        preds = torch.randn(2, 5, 10)  # batch_size=2, num_queries=5, num_classes=10
        targets = torch.zeros(2, 5, 10)
        # Mark some as positive (matched queries)
        targets[0, 0, 0] = 1.0
        targets[0, 1, 1] = 1.0
        targets[1, 0, 2] = 1.0
        
        # Create valid mask
        valid_mask = torch.ones(2, 5, 10, dtype=torch.bool)
        # Mark some text tokens as invalid
        valid_mask[:, :, 5:] = False
        
        # Create text mask
        text_mask = torch.ones(2, 10, dtype=torch.bool)
        text_mask[:, 5:] = False
        
        # Compute loss
        loss = self.bce_loss(preds, targets, valid_mask, text_mask)
        
        # Check that we get a scalar loss value
        self.assertIsInstance(loss, torch.Tensor)
        self.assertEqual(loss.dim(), 0)  # Scalar
        self.assertGreaterEqual(loss.item(), 0)  # Non-negative
    
    def test_focal_loss_with_negatives(self):
        """Test MultilabelFocalLoss with both positive and negative examples"""
        # Create mock predictions and targets
        preds = torch.randn(2, 5, 10)  # batch_size=2, num_queries=5, num_classes=10
        targets = torch.zeros(2, 5, 10)
        # Mark some as positive (matched queries)
        targets[0, 0, 0] = 1.0
        targets[0, 1, 1] = 1.0
        targets[1, 0, 2] = 1.0
        
        # Create valid mask
        valid_mask = torch.ones(2, 5, 10, dtype=torch.bool)
        # Mark some text tokens as invalid
        valid_mask[:, :, 5:] = False
        
        # Create text mask
        text_mask = torch.ones(2, 10, dtype=torch.bool)
        text_mask[:, 5:] = False
        
        # Compute loss
        loss = self.focal_loss(preds, targets, valid_mask, text_mask)
        
        # Check that we get a scalar loss value
        self.assertIsInstance(loss, torch.Tensor)
        self.assertEqual(loss.dim(), 0)  # Scalar
        self.assertGreaterEqual(loss.item(), 0)  # Non-negative

if __name__ == '__main__':
    unittest.main()