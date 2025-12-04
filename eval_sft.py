# eval_sft.py
from datasets import load_dataset
from transformers import T5ForConditionalGeneration, T5Tokenizer
from metrics import generate_summaries, compute_rouge


def main():
    ds = load_dataset("cnn_dailymail", "3.0.0")["validation"].select(range(100))

    base_name = "google/flan-t5-small"
    base_tok = T5Tokenizer.from_pretrained(base_name)
    base_model = T5ForConditionalGeneration.from_pretrained(base_name)
    base_preds, base_refs = generate_summaries(base_model, base_tok, ds)
    base_scores = compute_rouge(base_preds, base_refs)
    print("Base FLAN-T5-small ROUGE:", base_scores)

    ckpt_dir = "./checkpoints/flan_t5_sft_cnn_dm"
    sft_tok = T5Tokenizer.from_pretrained(ckpt_dir)
    sft_model = T5ForConditionalGeneration.from_pretrained(ckpt_dir)
    sft_preds, sft_refs = generate_summaries(sft_model, sft_tok, ds)
    sft_scores = compute_rouge(sft_preds, sft_refs)
    print("SFT FLAN-T5-small ROUGE:", sft_scores)


if __name__ == "__main__":
    main()