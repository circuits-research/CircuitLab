"""
Entirely made by Claude
"""

import pytest
import torch
import torch.nn as nn
from clt.config import CLTConfig, CLTTrainingRunnerConfig
from clt.clt import CLT
from clt.training.clt_trainer import CLTTrainer
from tests.utils import FakeActivationsStore
from pathlib import Path


def dummy_save_fn(trainer, checkpoint_name):
    """Dummy save function for testing"""
    pass


def test_gradient_accumulation_basic():
    """Test that gradient accumulation correctly accumulates gradients"""
    
    # Create a simple config
    cfg = CLTTrainingRunnerConfig(
        device="cpu",
        dtype="float32",
        seed=42,
        model_name="gpt2",
        d_in=64,
        d_latent=128,
        context_size=8,
        n_batches_in_buffer=2,
        store_batch_size_prompts=2,
        total_training_tokens=1024,
        train_batch_size_tokens=32,
        gradient_accumulation_steps=4,
        lr=1e-3,
        l0_coefficient=0.1,
        wandb_id="test_grad_accum",
        log_to_wandb=False,
        logger_verbose=False,
    )
    
    # Create CLT
    clt_cfg = cfg.create_sub_config(CLTConfig, n_layers=4)
    clt = CLT(clt_cfg)
    
    # Create fake activations
    batch_size = cfg.train_batch_size_tokens
    n_layers = 4
    x = torch.randn(batch_size, n_layers, cfg.d_in)
    y = torch.randn_like(x)
    fake_store = FakeActivationsStore(x, y)
    
    # Create trainer
    trainer = CLTTrainer(
        clt=clt,
        activations_store=fake_store,
        cfg=cfg,
        save_checkpoint_fn=dummy_save_fn,
    )
    
    # Test that n_training_steps only increments after full accumulation cycle
    initial_steps = trainer.n_training_steps
    
    # Process 4 micro-batches (1 full accumulation cycle)
    for i in range(4):
        loss_metrics = trainer._compute_training_step_loss(x, y)
        
        # Check accumulation_step cycles correctly
        expected_accum_step = (i + 1) % 4
        assert trainer.accumulation_step == expected_accum_step, \
            f"Step {i}: accumulation_step should be {expected_accum_step}, got {trainer.accumulation_step}"
    
    # After 4 micro-batches, we should have completed 1 optimizer step
    # But n_training_steps is incremented in fit(), not in _compute_training_step_loss
    # So we test it indirectly by checking accumulation_step reset
    assert trainer.accumulation_step == 0, "accumulation_step should reset to 0 after full cycle"


def test_gradient_accumulation_vs_no_accumulation():
    """Test that gradient accumulation with N steps gives similar results to 1 step with N*batch_size"""
    
    torch.manual_seed(42)
    
    # Config WITHOUT gradient accumulation (larger batch)
    cfg_no_accum = CLTTrainingRunnerConfig(
        device="cpu",
        dtype="float32",
        seed=42,
        model_name="gpt2",
        d_in=64,
        d_latent=128,
        context_size=8,
        n_batches_in_buffer=2,
        store_batch_size_prompts=2,
        total_training_tokens=1024,
        train_batch_size_tokens=128,  # 4x larger
        gradient_accumulation_steps=1,
        lr=1e-3,
        l0_coefficient=0.1,
        wandb_id="test_no_accum",
        log_to_wandb=False,
        logger_verbose=False,
    )
    
    # Create CLT and data
    clt_cfg = cfg_no_accum.create_sub_config(CLTConfig, n_layers=4)
    clt_no_accum = CLT(clt_cfg)
    
    # Large batch
    x_large = torch.randn(128, 4, 64)
    y_large = torch.randn_like(x_large)
    
    fake_store = FakeActivationsStore(x_large, y_large)
    trainer_no_accum = CLTTrainer(
        clt=clt_no_accum,
        activations_store=fake_store,
        cfg=cfg_no_accum,
        save_checkpoint_fn=dummy_save_fn,
    )
    
    # Get initial weights
    initial_W_enc_no_accum = clt_no_accum.W_enc.clone()
    
    # One training step with large batch
    loss_metrics_no_accum = trainer_no_accum._compute_training_step_loss(x_large, y_large)
    
    # Config WITH gradient accumulation (4 smaller batches)
    torch.manual_seed(42)  # Reset seed
    cfg_accum = CLTTrainingRunnerConfig(
        device="cpu",
        dtype="float32",
        seed=42,
        model_name="gpt2",
        d_in=64,
        d_latent=128,
        context_size=8,
        n_batches_in_buffer=2,
        store_batch_size_prompts=2,
        total_training_tokens=1024,
        train_batch_size_tokens=32,  # 4x smaller
        gradient_accumulation_steps=4,
        lr=1e-3,
        l0_coefficient=0.1,
        wandb_id="test_accum",
        log_to_wandb=False,
        logger_verbose=False,
    )
    
    clt_cfg = cfg_accum.create_sub_config(CLTConfig, n_layers=4)
    clt_accum = CLT(clt_cfg)
    
    # Copy weights to match initial state
    clt_accum.load_state_dict(clt_no_accum.state_dict())
    
    fake_store_accum = FakeActivationsStore(x_large[:32], y_large[:32])
    trainer_accum = CLTTrainer(
        clt=clt_accum,
        activations_store=fake_store_accum,
        cfg=cfg_accum,
        save_checkpoint_fn=dummy_save_fn,
    )
    
    # Four training steps with smaller batches (gradient accumulation)
    for i in range(4):
        x_mini = x_large[i*32:(i+1)*32]
        y_mini = y_large[i*32:(i+1)*32]
        loss_metrics_accum = trainer_accum._compute_training_step_loss(x_mini, y_mini)
    
    # The weight updates should be similar (not exactly same due to loss scaling and potential numerical differences)
    # But the direction should be similar
    delta_no_accum = clt_no_accum.W_enc - initial_W_enc_no_accum
    delta_accum = clt_accum.W_enc - initial_W_enc_no_accum
    
    # Check that both produced non-zero updates
    assert delta_no_accum.abs().max() > 1e-6, "No accumulation should produce weight updates"
    assert delta_accum.abs().max() > 1e-6, "With accumulation should produce weight updates"
    
    # Check that updates are in similar direction (cosine similarity > 0.5)
    delta_no_accum_flat = delta_no_accum.flatten()
    delta_accum_flat = delta_accum.flatten()
    cos_sim = torch.nn.functional.cosine_similarity(
        delta_no_accum_flat.unsqueeze(0),
        delta_accum_flat.unsqueeze(0)
    )
    
    assert cos_sim > 0.5, f"Weight updates should be in similar direction, got cosine similarity {cos_sim}"
    
    print(f"✓ Gradient accumulation test passed! Cosine similarity: {cos_sim.item():.4f}")


def test_scheduler_steps_correctly():
    """Test that schedulers only step after full accumulation cycle"""
    
    cfg = CLTTrainingRunnerConfig(
        device="cpu",
        dtype="float32",
        seed=42,
        model_name="gpt2",
        d_in=64,
        d_latent=128,
        context_size=8,
        n_batches_in_buffer=2,
        store_batch_size_prompts=2,
        total_training_tokens=1024,
        train_batch_size_tokens=32,
        gradient_accumulation_steps=4,
        lr=1e-3,
        lr_warm_up_steps=5,
        l0_coefficient=0.1,
        l0_warm_up_steps=5,
        wandb_id="test_scheduler",
        log_to_wandb=False,
        logger_verbose=False,
    )
    
    clt_cfg = cfg.create_sub_config(CLTConfig, n_layers=4)
    clt = CLT(clt_cfg)
    
    x = torch.randn(32, 4, cfg.d_in)
    y = torch.randn_like(x)
    fake_store = FakeActivationsStore(x, y)
    
    trainer = CLTTrainer(
        clt=clt,
        activations_store=fake_store,
        cfg=cfg,
        save_checkpoint_fn=dummy_save_fn,
    )
    
    initial_lr = trainer.lr_scheduler.get_lr()
    initial_l0 = trainer.l0_scheduler.get_lr()
    
    # Process 3 micro-batches (incomplete cycle)
    for i in range(3):
        trainer._compute_training_step_loss(x, y)
    
    # Schedulers should NOT have stepped yet
    assert trainer.lr_scheduler.current_step == 0, "LR scheduler should not step during accumulation"
    assert trainer.l0_scheduler.current_step == 0, "L0 scheduler should not step during accumulation"
    
    # Complete the cycle with 4th micro-batch
    trainer._compute_training_step_loss(x, y)
    
    # NOW schedulers should have stepped once
    assert trainer.lr_scheduler.current_step == 1, "LR scheduler should step after full accumulation"
    assert trainer.l0_scheduler.current_step == 1, "L0 scheduler should step after full accumulation"
    
    print("✓ Scheduler stepping test passed!")


if __name__ == "__main__":
    test_gradient_accumulation_basic()
    test_scheduler_steps_correctly()
    test_gradient_accumulation_vs_no_accumulation()
    print("\n✅ All gradient accumulation tests passed!")
