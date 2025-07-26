# Testing Grounding DINO with and without Negative Sampling

This document provides instructions for training and testing Grounding DINO with and without negative sampling.

## Training Commands

### 1. Training WITHOUT Negative Sampling

```bash
# Training WITHOUT negative sampling
cd /home/asad/dev/Grounding-Dino-FineTuning
python train.py configs/train_config_no_neg_sampling.yaml
```

### 2. Training WITH Negative Sampling

```bash
# Training WITH negative sampling (using existing config)
cd /home/asad/dev/Grounding-Dino-FineTuning
python train.py configs/train_config.yaml
```

## Testing Commands

### 1. Testing the Trained Model

After training, you can test your model using the existing test script:

```bash
# Test model trained WITHOUT negative sampling
cd /home/asad/dev/Grounding-Dino-FineTuning
python test.py --config configs/train_config_no_neg_sampling.yaml --weights weights/checkpoint_epoch_5.pth

# Test model trained WITH negative sampling
cd /home/asad/dev/Grounding-Dino-FineTuning
python test.py --config configs/train_config.yaml --weights weights/checkpoint_epoch_5.pth
```

### 2. Inference on a Single Image

You can also run inference on a single image using the demo script:

```bash
# Inference without negative sampling model
cd /home/asad/dev/Grounding-Dino-FineTuning
python demo/inference_on_a_image.py \
  --config configs/GroundingDINO_SwinT_OGC.py \
  --weights weights/checkpoint_epoch_5.pth \
  --image_path /path/to/your/image.jpg \
  --text_prompt "shirt . pants . bag ."

# Inference with negative sampling model
cd /home/asad/dev/Grounding-Dino-FineTuning
python demo/inference_on_a_image.py \
  --config configs/GroundingDINO_SwinT_OGC.py \
  --weights weights/checkpoint_epoch_5.pth \
  --image_path /path/to/your/image.jpg \
  --text_prompt "shirt . pants . bag . shoes . hat . jacket"
```

### 3. Running Unit Tests

To verify that the implementation works correctly:

```bash
# Run all unit tests
cd /home/asad/dev/Grounding-Dino-FineTuning
python -m pytest tests/ -v
```

### 4. Comparing Results

To compare the results of models trained with and without negative sampling:

```bash
# Run evaluation on COCO format dataset for model WITHOUT negative sampling
cd /home/asad/dev/Grounding-Dino-FineTuning
python demo/test_ap_on_coco.py \
  --config configs/GroundingDINO_SwinT_OGC.py \
  --weights weights/checkpoint_epoch_5.pth \
  --anno_path multimodal-data/fashion_dataset_subset/val_annotations.csv \
  --image_dir multimodal-data/fashion_dataset_subset/images/val

# Run evaluation on COCO format dataset for model WITH negative sampling
cd /home/asad/dev/Grounding-Dino-FineTuning
python demo/test_ap_on_coco.py \
  --config configs/GroundingDINO_SwinT_OGC.py \
  --weights weights/checkpoint_epoch_5.pth \
  --anno_path multimodal-data/fashion_dataset_subset/val_annotations.csv \
  --image_dir multimodal-data/fashion_dataset_subset/images/val
```

## Additional Notes

1. Make sure to adjust the checkpoint epoch number based on your actual saved checkpoints.
2. You might want to modify the number of epochs, learning rate, or other hyperparameters based on your specific requirements.
3. For a fair comparison, use the same random seed for both training runs.
4. When testing on new categories not seen during training, the model with negative sampling should perform better at identifying when those categories are not present in the image.