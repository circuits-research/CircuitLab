#!/usr/bin/env python3
"""
Test script for CLT Activation Compression functionality.

This script tests:
1. Quantization error for different compression configs
2. Compression ratios and file sizes
3. Save/load cycle accuracy
4. Gradient error when training with compressed activations
5. Training dynamics comparison
6. ActivationsStore integration

Usage:
    python scripts/test_compression.py [--quick] [--no-training] [--device cuda]
"""

import sys
import os
import argparse
import tempfile
import time
from pathlib import Path

# Add src to path
SCRIPT_DIR = Path(__file__).parent.absolute()
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import torch
import numpy as np
from tqdm import tqdm

# Import CLT modules
from clt.training.compressed_activations_store import (
    CompressionConfig,
    CompressedActivationsStore,
    QuantizedTensor,
    estimate_compression_savings,
)
from clt.config import CLTConfig


def print_header(title: str):
    """Print a formatted section header."""
    print("\n" + "=" * 80)
    print(f" {title}")
    print("=" * 80)


def print_table(data: list[dict], title: str = None):
    """Print a formatted table from list of dicts."""
    if title:
        print(f"\n{title}")
        print("-" * 60)

    if not data:
        print("  No data")
        return

    # Get column widths
    keys = list(data[0].keys())
    widths = {k: max(len(str(k)), max(len(str(row.get(k, ""))) for row in data)) for k in keys}

    # Print header
    header = " | ".join(f"{k:{widths[k]}}" for k in keys)
    print(header)
    print("-" * len(header))

    # Print rows
    for row in data:
        print(" | ".join(f"{str(row.get(k, '')):{widths[k]}}" for k in keys))


def generate_synthetic_activations(
    batch_size: int = 32,
    context_size: int = 128,
    n_layers: int = 12,
    d_model: int = 768,
    device: str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Generate synthetic activations for testing.

    Returns:
        act_in: [batch_size * context_size, n_layers, d_model]
        act_out: [batch_size * context_size, n_layers, d_model]
    """
    n_tokens = batch_size * context_size

    # Generate activations with realistic statistics
    # act_in is typically normalized (mean~0, std~1)
    act_in = torch.randn(n_tokens, n_layers, d_model, device=device)

    # act_out has slightly larger variance
    act_out = torch.randn(n_tokens, n_layers, d_model, device=device) * 1.5

    return act_in, act_out


def test_quantization_error(act_in: torch.Tensor, act_out: torch.Tensor, configs: list[CompressionConfig]) -> list[dict]:
    """Test quantization and reconstruction error for each config."""
    print_header("QUANTIZATION ERROR TEST")

    results = []

    for config in tqdm(configs, desc="Testing quantization"):
        store = CompressedActivationsStore(config)

        # Quantize
        q_act_in, q_act_out = store.quantize_activations(act_in, act_out)

        # Dequantize
        recon_act_in = q_act_in.dequantize().to(act_in.dtype)
        recon_act_out = q_act_out.dequantize().to(act_out.dtype)

        # Compute errors
        in_mae = (act_in.cpu() - recon_act_in).abs().mean().item()
        in_rmse = ((act_in.cpu() - recon_act_in) ** 2).mean().sqrt().item()
        in_max = (act_in.cpu() - recon_act_in).abs().max().item()

        out_mae = (act_out.cpu() - recon_act_out).abs().mean().item()
        out_rmse = ((act_out.cpu() - recon_act_out) ** 2).mean().sqrt().item()

        config_name = f"{config.quantization}+{config.compression}"
        results.append({
            "Config": config_name,
            "In MAE": f"{in_mae:.6f}",
            "In RMSE": f"{in_rmse:.6f}",
            "In Max": f"{in_max:.6f}",
            "Out MAE": f"{out_mae:.6f}",
            "Out RMSE": f"{out_rmse:.6f}",
        })

    print_table(results)
    return results


def test_compression_ratio(act_in: torch.Tensor, act_out: torch.Tensor, configs: list[CompressionConfig]) -> list[dict]:
    """Test actual compression ratios by saving to disk."""
    print_header("COMPRESSION RATIO TEST")

    results = []
    original_size = (act_in.element_size() * act_in.numel() + act_out.element_size() * act_out.numel())

    for config in tqdm(configs, desc="Testing compression"):
        store = CompressedActivationsStore(config)

        with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
            temp_path = Path(f.name)

        try:
            file_size = store.save_compressed(temp_path, act_in.cpu(), act_out.cpu())
            ratio = original_size / file_size
            savings = (1 - 1/ratio) * 100

            config_name = f"{config.quantization}+{config.compression}"
            results.append({
                "Config": config_name,
                "Original (MB)": f"{original_size / 1024 / 1024:.2f}",
                "Compressed (MB)": f"{file_size / 1024 / 1024:.2f}",
                "Ratio": f"{ratio:.2f}x",
                "Savings": f"{savings:.1f}%",
            })
        finally:
            temp_path.unlink(missing_ok=True)

    print_table(results)
    return results


def test_save_load_cycle(act_in: torch.Tensor, act_out: torch.Tensor, configs: list[CompressionConfig]) -> list[dict]:
    """Test complete save/load cycle accuracy."""
    print_header("SAVE/LOAD CYCLE TEST")

    results = []
    tokens = torch.randint(0, 50000, (act_in.shape[0],), dtype=torch.long)

    for config in tqdm(configs, desc="Testing save/load"):
        store = CompressedActivationsStore(config)

        with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
            temp_path = Path(f.name)

        try:
            # Save
            start_save = time.time()
            file_size = store.save_compressed(
                temp_path, act_in.cpu(), act_out.cpu(),
                tokens=tokens, metadata={"test": True}
            )
            save_time = time.time() - start_save

            # Load
            start_load = time.time()
            loaded_in, loaded_out, loaded_tokens, metadata = store.load_compressed(temp_path)
            load_time = time.time() - start_load

            # Verify
            in_error = (act_in.cpu() - loaded_in).abs().mean().item()
            out_error = (act_out.cpu() - loaded_out).abs().mean().item()
            tokens_match = torch.equal(tokens, loaded_tokens) if loaded_tokens is not None else False
            meta_ok = metadata.get("test") == True

            config_name = f"{config.quantization}+{config.compression}"
            results.append({
                "Config": config_name,
                "Size (MB)": f"{file_size / 1024 / 1024:.2f}",
                "Save (s)": f"{save_time:.3f}",
                "Load (s)": f"{load_time:.3f}",
                "In MAE": f"{in_error:.8f}",
                "Tokens OK": "Yes" if tokens_match else "No",
                "Meta OK": "Yes" if meta_ok else "No",
            })
        except Exception as e:
            config_name = f"{config.quantization}+{config.compression}"
            results.append({
                "Config": config_name,
                "Error": str(e)[:40],
            })
        finally:
            temp_path.unlink(missing_ok=True)

    print_table(results)
    return results


def test_gradient_error(act_in: torch.Tensor, act_out: torch.Tensor, configs: list[CompressionConfig], device: str = "cpu") -> list[dict]:
    """Test gradient error when training with compressed activations."""
    print_header("GRADIENT ERROR TEST")

    # Create a simple CLT model for gradient testing
    clt_config = CLTConfig(
        device=device,
        dtype="float32",
        seed=42,
        model_name="gpt2",
        n_layers=12,
        d_in=768,
        d_latent=768 * 4,
        jumprelu_bandwidth=0.001,
        jumprelu_init_threshold=0.001,
        normalize_decoder=False,
        dead_feature_window=250,
        cross_layer_decoders=False,
        context_size=128,
        l0_coefficient=1e-4,
    )

    from clt.clt import CLT
    clt_model = CLT(clt_config)
    clt_model.to(device)

    # Use smaller batch for gradient test
    test_in = act_in[:1024].to(device)
    test_out = act_out[:1024].to(device)

    # Get original gradients
    clt_model.zero_grad()
    loss_orig, _ = clt_model(test_in, test_out, l0_coef=1e-4, df_coef=0.0, return_metrics=True)
    loss_orig.backward()

    grads_original = {}
    for name, param in clt_model.named_parameters():
        if param.grad is not None:
            grads_original[name] = param.grad.clone().cpu()

    results = []

    for config in tqdm(configs[1:], desc="Testing gradients"):  # Skip 'none' config
        store = CompressedActivationsStore(config)

        # Quantize and dequantize
        q_in, q_out = store.quantize_activations(test_in.cpu(), test_out.cpu())
        recon_in = q_in.dequantize().to(test_in.dtype).to(device)
        recon_out = q_out.dequantize().to(test_out.dtype).to(device)

        # Compute gradients with compressed activations
        clt_model.zero_grad()
        loss_comp, _ = clt_model(recon_in, recon_out, l0_coef=1e-4, df_coef=0.0, return_metrics=True)
        loss_comp.backward()

        # Compare gradients
        cosine_sims = []
        for name, param in clt_model.named_parameters():
            if param.grad is not None and name in grads_original:
                g_orig = grads_original[name]
                g_comp = param.grad.cpu()

                cosine_sim = torch.nn.functional.cosine_similarity(
                    g_orig.flatten().unsqueeze(0),
                    g_comp.flatten().unsqueeze(0)
                ).item()
                cosine_sims.append(cosine_sim)

        avg_cosine_sim = np.mean(cosine_sims)
        loss_diff_pct = abs(loss_comp.item() - loss_orig.item()) / loss_orig.item() * 100

        config_name = f"{config.quantization}+{config.compression}"
        results.append({
            "Config": config_name,
            "Loss Orig": f"{loss_orig.item():.2f}",
            "Loss Comp": f"{loss_comp.item():.2f}",
            "Loss Diff %": f"{loss_diff_pct:.2f}%",
            "Avg Cos Sim": f"{avg_cosine_sim:.6f}",
        })

    print_table(results)
    return results


def test_training_dynamics(act_in: torch.Tensor, act_out: torch.Tensor, device: str = "cpu", n_steps: int = 50) -> dict:
    """Compare training dynamics with original vs compressed activations."""
    print_header("TRAINING DYNAMICS TEST")

    from clt.clt import CLT

    clt_config = CLTConfig(
        device=device,
        dtype="float32",
        seed=42,
        model_name="gpt2",
        n_layers=12,
        d_in=768,
        d_latent=768 * 4,
        jumprelu_bandwidth=0.001,
        jumprelu_init_threshold=0.001,
        normalize_decoder=False,
        dead_feature_window=250,
        cross_layer_decoders=False,
        context_size=128,
        l0_coefficient=1e-4,
    )

    compression_config = CompressionConfig(
        quantization="int8",
        compression="zstd",
        compression_level=3,
        per_layer_scale=True,
        symmetric=True,
    )
    compression_store = CompressedActivationsStore(compression_config)

    # Prepare compressed activations
    q_in, q_out = compression_store.quantize_activations(act_in, act_out)
    comp_in = q_in.dequantize().to(act_in.dtype)
    comp_out = q_out.dequantize().to(act_out.dtype)

    def run_training(clt_model, act_in_train, act_out_train, n_steps):
        torch.manual_seed(42)
        clt_model._initialize()
        optimizer = torch.optim.Adam(clt_model.parameters(), lr=1e-4)

        losses = []
        for step in range(n_steps):
            # Use different chunks for each step
            start = (step * 1024) % (act_in_train.shape[0] - 1024)
            batch_in = act_in_train[start:start+1024].to(device)
            batch_out = act_out_train[start:start+1024].to(device)

            optimizer.zero_grad()
            loss, _ = clt_model(batch_in, batch_out, l0_coef=1e-4, df_coef=0.0, return_metrics=True)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        return losses

    print(f"Running {n_steps} training steps with original activations...")
    clt_model = CLT(clt_config)
    clt_model.to(device)
    losses_orig = run_training(clt_model, act_in, act_out, n_steps)

    print(f"Running {n_steps} training steps with compressed activations...")
    clt_model = CLT(clt_config)
    clt_model.to(device)
    losses_comp = run_training(clt_model, comp_in, comp_out, n_steps)

    # Compare final losses
    final_orig = np.mean(losses_orig[-10:])
    final_comp = np.mean(losses_comp[-10:])
    diff_pct = abs(final_comp - final_orig) / final_orig * 100

    print(f"\nResults (last 10 steps average):")
    print(f"  Original:   {final_orig:.4f}")
    print(f"  Compressed: {final_comp:.4f}")
    print(f"  Difference: {diff_pct:.2f}%")

    return {
        "original": losses_orig,
        "compressed": losses_comp,
        "final_diff_pct": diff_pct,
    }


def test_storage_estimates():
    """Test storage estimates for large datasets."""
    print_header("STORAGE ESTIMATES (1B tokens, GPT-2)")

    configs = [
        CompressionConfig(quantization="int8", compression="none"),
        CompressionConfig(quantization="int8", compression="zstd"),
        CompressionConfig(quantization="int4", compression="none"),
        CompressionConfig(quantization="int4", compression="zstd"),
    ]

    results = []
    for config in configs:
        est = estimate_compression_savings(
            num_tokens=1_000_000_000,
            num_layers=12,
            d_model=768,
            config=config,
        )
        config_name = f"{config.quantization}+{config.compression}"
        results.append({
            "Config": config_name,
            "Base (GB)": f"{est['base_size_gb']:.1f}",
            "Final (GB)": f"{est['final_size_gb']:.1f}",
            "Ratio": f"{est['compression_ratio']:.1f}x",
            "Savings (GB)": f"{est['savings_gb']:.1f}",
        })

    print_table(results)
    return results


def main():
    parser = argparse.ArgumentParser(description="Test CLT compression functionality")
    parser.add_argument("--quick", action="store_true", help="Run quick tests with smaller data")
    parser.add_argument("--no-training", action="store_true", help="Skip training dynamics test")
    parser.add_argument("--no-gradients", action="store_true", help="Skip gradient error test")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for tests")
    parser.add_argument("--context-size", type=int, default=128, help="Context size for tests")
    args = parser.parse_args()

    print_header("CLT COMPRESSION TEST SUITE")
    print(f"Device: {args.device}")
    print(f"Batch size: {args.batch_size}")
    print(f"Context size: {args.context_size}")

    if args.quick:
        args.batch_size = 8
        print("Running in QUICK mode with reduced data")

    # Set random seeds
    torch.manual_seed(42)
    np.random.seed(42)

    # Generate test activations
    print("\nGenerating synthetic activations...")
    act_in, act_out = generate_synthetic_activations(
        batch_size=args.batch_size,
        context_size=args.context_size,
        device=args.device,
    )
    print(f"  act_in shape:  {act_in.shape}")
    print(f"  act_out shape: {act_out.shape}")

    # Define compression configs to test
    configs = [
        CompressionConfig(quantization="none", compression="none"),
        CompressionConfig(quantization="int8", compression="none", per_layer_scale=True, symmetric=True),
        CompressionConfig(quantization="int8", compression="zstd", per_layer_scale=True, symmetric=True),
        CompressionConfig(quantization="int4", compression="none", per_layer_scale=True, symmetric=True),
        CompressionConfig(quantization="int4", compression="zstd", per_layer_scale=True, symmetric=True),
    ]

    # Run tests
    all_passed = True

    try:
        test_quantization_error(act_in, act_out, configs)
    except Exception as e:
        print(f"FAILED: {e}")
        all_passed = False

    try:
        test_compression_ratio(act_in, act_out, configs)
    except Exception as e:
        print(f"FAILED: {e}")
        all_passed = False

    try:
        test_save_load_cycle(act_in, act_out, configs)
    except Exception as e:
        print(f"FAILED: {e}")
        all_passed = False

    if not args.no_gradients:
        try:
            test_gradient_error(act_in, act_out, configs, device=args.device)
        except Exception as e:
            print(f"FAILED: {e}")
            all_passed = False

    if not args.no_training:
        try:
            n_steps = 20 if args.quick else 50
            test_training_dynamics(act_in, act_out, device=args.device, n_steps=n_steps)
        except Exception as e:
            print(f"FAILED: {e}")
            all_passed = False

    try:
        test_storage_estimates()
    except Exception as e:
        print(f"FAILED: {e}")
        all_passed = False

    # Summary
    print_header("TEST SUMMARY")
    if all_passed:
        print("All tests PASSED")
        return 0
    else:
        print("Some tests FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(main())
