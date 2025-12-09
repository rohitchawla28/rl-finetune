# scripts/run_ppo.py
from rl.ppo_trl import run_ppo

if __name__ == "__main__":
    run_ppo(model_name="google/flan-t5-small",
            output_dir="./checkpoints/flan_t5_ppo_cnn_dm",
            batch_size=4, max_new_tokens=64, lr=1e-5, ppo_epochs=1)