# scripts/train_scst.py
from src.scst import run_scst

def main():
    run_scst(
        model_name="google/flan-t5-small",          # or "./checkpoints/flan_t5_sft_cnn_dm"
        out_dir="checkpoints/flan_t5_scst_from_base",
        n_train=1000,
        batch_size=4,
        epochs=1,
        lr=1e-6,
        warmup_ratio=0.1,
        greedy_beams=4,
        greedy_max_new=64,
        sample_max_new=64,
        sample_min_new=8,
        top_k=30,
        top_p=0.8,
        temperature=0.7,
        no_repeat_ngram_size=4,
        max_input_len=512,
        max_target_len=128,
        advantage_normalize=True,
        reward_clip=0.5,
        seed=42,
        debug=True,
    )

if __name__ == "__main__":
    main()