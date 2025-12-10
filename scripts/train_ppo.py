# scripts/train_ppo.py
from ppo import run_ppo

def main():
    run_ppo(
        model_name="google/flan-t5-small",     # or "./checkpoints/flan_t5_sft_cnn_dm"
        out_dir="./checkpoints/flan_t5_ppo_cnn_dm",
        n_train=300, steps_per_epoch=25, epochs=1,
        max_new_tokens=64, length_penalty=0.0,
        use_instruction=False,                 # keep False to match SFT inputs
        lr=1e-5, batch_size=4, ppo_epochs=4, target_kl=0.1, seed=42
    )

if __name__ == "__main__":
    main()