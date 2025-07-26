"""
Demo script to show how negative sampling would work with a more diverse dataset
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '.'))

from groundingdino.datasets.dataset import GroundingDINODataset

# Mock implementation to demonstrate negative sampling concept
class MockDataset:
    def __init__(self):
        # Simulate a dataset with many categories
        self.all_categories = [
            'shirt', 'pants', 'bag', 'shoes', 'hat', 'jacket', 
            'dress', 'skirt', 'shorts', 'socks', 'gloves', 'scarf'
        ]
    
    def sample_negative_categories(self, positive_categories, num_negative=None):
        """
        Sample negative categories that are not present in the current image
        """
        if num_negative is None:
            num_negative = max(1, len(positive_categories))
        
        # Get candidates that are not in positive categories
        candidates = [cat for cat in self.all_categories if cat not in positive_categories]
        
        # Handle case where there are no candidates
        if not candidates:
            return []
        
        # Sample negative categories
        import random
        if len(candidates) >= num_negative:
            negative_categories = random.sample(candidates, num_negative)
        else:
            # If not enough candidates, sample with replacement
            negative_categories = [random.choice(candidates) for _ in range(num_negative)]
            
        return negative_categories

def demo_negative_sampling():
    print("=== Negative Sampling Concept Demo ===")
    
    # Create mock dataset
    dataset = MockDataset()
    
    # Simulate different images with different positive categories
    test_cases = [
        (["shirt", "pants"], "Person wearing top and bottom"),
        (["dress"], "Person wearing a dress"),
        (["shirt", "shorts", "shoes"], "Casual outfit"),
        (["jacket", "hat", "gloves", "scarf"], "Winter outfit")
    ]
    
    for positive_cats, description in test_cases:
        print(f"\n--- {description} ---")
        print(f"Positive categories: {positive_cats}")
        
        # Sample negative categories
        negative_cats = dataset.sample_negative_categories(positive_cats, len(positive_cats))
        print(f"Negative categories: {negative_cats}")
        
        # Combined categories for training
        all_cats = positive_cats + negative_cats
        print(f"All categories for training: {all_cats}")

if __name__ == "__main__":
    demo_negative_sampling()