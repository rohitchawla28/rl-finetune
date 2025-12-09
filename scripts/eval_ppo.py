# scripts/eval_ppo.py
from datasets import load_dataset
from transformers import T5Tokenizer, T5ForConditionalGeneration
from metrics import generate_summaries, compute_rouge  # PYTHONPATH=src

def main():
    eval_raw = load_dataset("cnn_dailymail", "3.0.0")["validation"].select(range(100))

    # Base
    base_name = "google/flan-t5-small"
    base_tok = T5Tokenizer.from_pretrained(base_name)
    base_model = T5ForConditionalGeneration.from_pretrained(base_name)
    base_preds, base_refs = generate_summaries(base_model, base_tok, eval_raw)
    base_scores = compute_rouge(base_preds, base_refs)

    # SFT
    sft_dir = "./checkpoints/flan_t5_sft_cnn_dm"
    sft_tok = T5Tokenizer.from_pretrained(sft_dir)
    sft_model = T5ForConditionalGeneration.from_pretrained(sft_dir)
    sft_preds, sft_refs = generate_summaries(sft_model, sft_tok, eval_raw)
    sft_scores = compute_rouge(sft_preds, sft_refs)

    # PPO
    ppo_dir = "./checkpoints/flan_t5_ppo_cnn_dm"
    ppo_tok = T5Tokenizer.from_pretrained(ppo_dir)
    ppo_model = T5ForConditionalGeneration.from_pretrained(ppo_dir)
    ppo_preds, ppo_refs = generate_summaries(ppo_model, ppo_tok, eval_raw)
    ppo_scores = compute_rouge(ppo_preds, ppo_refs)

    print("Base:", base_scores)
    print("SFT :", sft_scores)
    print("PPO :", ppo_scores)

if __name__ == "__main__":
    main()