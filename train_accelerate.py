# This script uses HuggingFace Accelerate for multi-GPU training.
import os
import yaml
from datetime import datetime
from collections import defaultdict
from accelerate import Accelerator
import torch
from config import ConfigurationManager
from train import setup_model, setup_data_loaders, GroundingDINOVisualizer, freeze_model_layers, verify_only_lora_trainable, print_frozen_status, GroundingDINOTrainer


def train_accelerate(config_path: str, save_dir: str = None) -> None:
    accelerator = Accelerator()
    data_config, model_config, training_config = ConfigurationManager.load_config(config_path)

    model = setup_model(model_config, training_config.use_lora)
    train_loader, val_loader = setup_data_loaders(data_config)
    steps_per_epoch = len(train_loader.dataset) // data_config.batch_size

    if save_dir:
        training_config.save_dir = save_dir
    save_dir = os.path.join(
        training_config.save_dir,
        datetime.now().strftime("%Y%m%d_%H%M")
    )
    if accelerator.is_main_process:
        os.makedirs(save_dir, exist_ok=True)
        config_save_path = os.path.join(save_dir, 'config.yaml')
        with open(config_save_path, 'w') as f:
            yaml.dump({
                'data': vars(data_config),
                'model': vars(model_config),
                'training': vars(training_config)
            }, f, default_flow_style=False)

    visualizer = GroundingDINOVisualizer(save_dir=save_dir)

    if not training_config.use_lora:
        if accelerator.is_main_process:
            print("Freezing most of model except few layers!")
        freeze_model_layers(model)
    else:
        if accelerator.is_main_process:
            print(f"Is only Lora trainable?  {verify_only_lora_trainable(model)} ")
    if accelerator.is_main_process:
        print_frozen_status(model)

    # Trainer is used for loss, optimizer, scheduler, etc., but we will manage optimizer/step with Accelerator
    trainer = GroundingDINOTrainer(
        model,
        num_steps_per_epoch=steps_per_epoch,
        num_epochs=training_config.num_epochs,
        warmup_epochs=training_config.warmup_epochs,
        learning_rate=training_config.learning_rate,
        use_lora=training_config.use_lora,
        use_gradient_clipping=training_config.use_gradient_clipping,
        max_grad_norm=training_config.max_grad_norm
    )

    # Prepare with accelerator
    model, optimizer, train_loader, val_loader = accelerator.prepare(
        trainer.model, trainer.optimizer, train_loader, val_loader
    )
    trainer.model = model
    trainer.optimizer = optimizer

    for epoch in range(training_config.num_epochs):
        if accelerator.is_main_process and epoch % training_config.visualization_frequency == 0:
            visualizer.visualize_epoch(model, val_loader, epoch, trainer.prepare_batch, box_th=0.3, txt_th=0.2)

        epoch_losses = defaultdict(list)
        for batch_idx, batch in enumerate(train_loader):
            trainer.model.train()
            trainer.optimizer.zero_grad()
            images, targets, captions = trainer.prepare_batch(batch)
            outputs = trainer.model(images, captions=captions)
            loss_dict = trainer.criterion(outputs, targets, captions=captions, tokenizer=trainer.model.tokenizer)
            total_loss = sum(loss_dict[k] * trainer.weights_dict_loss[k] for k in loss_dict.keys() if k in trainer.weights_dict_loss)
            accelerator.backward(total_loss)
            loss_dict['total_loss'] = total_loss
            if training_config.use_gradient_clipping:
                accelerator.clip_grad_norm_(trainer.model.parameters(), training_config.max_grad_norm)
            trainer.optimizer.step()
            if trainer.scheduler is not None:
                trainer.scheduler.step()
            if trainer.use_ema:
                trainer.ema_model.update()
            # Only main process prints
            if accelerator.is_main_process and batch_idx % 5 == 0:
                loss_str = ", ".join(f"{k}: {v.item() if isinstance(v, torch.Tensor) else v:.4f}" for k, v in loss_dict.items())
                print(f"Epoch {epoch+1}/{training_config.num_epochs}, Batch {batch_idx}/{len(train_loader)}, {loss_str}")
            for k, v in loss_dict.items():
                epoch_losses[k].append(v.item() if isinstance(v, torch.Tensor) else v)
        if accelerator.is_main_process:
            avg_losses = {k: sum(v)/len(v) for k, v in epoch_losses.items()}
            print(f"Epoch {epoch+1} complete. Average losses:", ", ".join(f"{k}: {v:.4f}" for k, v in avg_losses.items()))
            if (epoch + 1) % training_config.save_frequency == 0:
                trainer.save_checkpoint(
                    os.path.join(save_dir, f'checkpoint_epoch_{epoch+1}.pth'),
                    epoch,
                    avg_losses,
                    use_lora=training_config.use_lora
                )

if __name__ == "__main__":
    import sys
    config_path = sys.argv[1] if len(sys.argv) > 1 else 'configs/train_config.yaml'
    train_accelerate(config_path) 