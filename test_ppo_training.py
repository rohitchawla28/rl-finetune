"""
Quick test to verify PPO is actually training.

Run this to check if PPO updates are happening.
"""

# Test: Check if model weights change during PPO training
import torch
from transformers import T5ForConditionalGeneration
from src.ppo import run_ppo

# Load a model and get initial weights
model_path = "google/flan-t5-small"  # or your SFT model
model_before = T5ForConditionalGeneration.from_pretrained(model_path)
weights_before = model_before.lm_head.weight.data.clone()

# Run a very short PPO training (just a few steps)
print("Running short PPO training...")
ppo_dir = run_ppo(
    model_name=model_path,
    out_dir="./test_ppo_check",
    n_train=64,  # Very small - just 1 batch
    batch_size=64,
    epochs=1,
    lr=1e-5,  # Higher LR to see changes
    ppo_epochs=4,
    verbosity="steps",  # Show all steps
)

# Load model after training
model_after = T5ForConditionalGeneration.from_pretrained(ppo_dir)
weights_after = model_after.lm_head.weight.data

# Check if weights changed
weight_diff = (weights_before - weights_after).abs().mean().item()
print(f"\nWeight change (mean absolute difference): {weight_diff:.8f}")

if weight_diff > 1e-6:
    print("✅ PPO training is working! Model weights changed.")
else:
    print("❌ PPO training NOT working! Model weights unchanged.")
    print("   This suggests PPO updates aren't being applied.")

