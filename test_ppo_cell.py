"""
CELL: Test if PPO is Actually Training
=======================================
Copy this into a notebook cell to verify PPO is working.
"""

# Test PPO training - verify model weights change
import torch
from transformers import T5ForConditionalGeneration
from src.ppo import run_ppo
from src.eval_utils import eval_model
from src.experiment_runner import load_eval_dataset, get_dataset_keys

# 1. Load model and capture initial weights
print("="*70)
print("STEP 1: Loading model and capturing initial weights...")
print("="*70)
model_path = CFG["model_name"]  # or use best_dir if you have SFT
model_before = T5ForConditionalGeneration.from_pretrained(model_path)
# Capture weights from a specific layer (lm_head is good to check)
weights_before = model_before.lm_head.weight.data.clone()
print(f"Model loaded: {model_path}")
print(f"Initial weight sample (first 5 values): {weights_before[0, :5].tolist()}")

# 2. Run very short PPO training (just 1-2 batches)
print("\n" + "="*70)
print("STEP 2: Running short PPO training (64 samples, 1 batch)...")
print("="*70)
test_ppo_dir = "./test_ppo_verification"
eval_raw_test = load_eval_dataset(CFG["dataset"], n_eval=100)  # Small eval set
text_key, ref_key = get_dataset_keys(CFG["dataset"])

ppo_dir = run_ppo(
    model_name=model_path,
    out_dir=test_ppo_dir,
    n_train=64,  # Just 1 batch (batch_size=64)
    batch_size=64,
    epochs=1,
    lr=1e-5,  # Higher LR to see changes more clearly
    ppo_epochs=4,
    target_kl=0.1,
    min_len=200,
    max_len=1200,
    max_new_tokens=64,
    min_new_tokens=1,  # Lower minimum to allow shorter generations
    use_instruction=True,  # Use instruction format for flan-t5 models
    verbosity="steps",  # Show all steps for debugging
)

# 3. Load model after training and check weights
print("\n" + "="*70)
print("STEP 3: Checking if model weights changed...")
print("="*70)
model_after = T5ForConditionalGeneration.from_pretrained(ppo_dir)
weights_after = model_after.lm_head.weight.data

# Calculate weight difference
weight_diff = (weights_before - weights_after).abs().mean().item()
max_diff = (weights_before - weights_after).abs().max().item()

print(f"Weight change (mean absolute difference): {weight_diff:.8f}")
print(f"Weight change (max absolute difference): {max_diff:.8f}")
print(f"After weight sample (first 5 values): {weights_after[0, :5].tolist()}")

# 4. Quick evaluation comparison
print("\n" + "="*70)
print("STEP 4: Quick evaluation comparison...")
print("="*70)
scores_before = eval_model(model_path, eval_raw_test, text_key=text_key, ref_key=ref_key)
scores_after = eval_model(ppo_dir, eval_raw_test, text_key=text_key, ref_key=ref_key)

print(f"\nBefore PPO - ROUGE-L: {scores_before['rougeL']:.4f}")
print(f"After PPO  - ROUGE-L: {scores_after['rougeL']:.4f}")
print(f"Difference: {scores_after['rougeL'] - scores_before['rougeL']:+.4f}")

# 5. Conclusion
print("\n" + "="*70)
print("VERIFICATION RESULT")
print("="*70)
if weight_diff > 1e-6:
    print("✅ SUCCESS: PPO training is working!")
    print(f"   Model weights changed by {weight_diff:.8f} (mean)")
    print(f"   This confirms PPO updates are being applied.")
else:
    print("❌ WARNING: Model weights did NOT change significantly")
    print(f"   Weight difference: {weight_diff:.8f}")
    print(f"   This suggests PPO updates may not be working.")
    print(f"   Check training logs and verify PPO step() is being called.")

print("\n" + "="*70)

