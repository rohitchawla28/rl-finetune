# scripts/eval_models.py
from datasets import load_dataset
from transformers import T5ForConditionalGeneration, T5Tokenizer
from eval.metrics import generate_summaries, compute_rouge
import torch

def load_and_eval(model_path: str, name: str, ds, device):
    tok = T5Tokenizer.from_pretrained(model_path)
    mdl = T5ForConditionalGeneration.from_pretrained(model_path)
    preds, refs = generate_summaries(mdl, tok, ds, device=device)
    scores = compute_rouge(preds, refs)
    print(f"{name} ROUGE:", scores)

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ds = load_dataset("cnn_dailymail", "3.0.0")["validation"].select(range(100))
    load_and_eval("google/flan-t5-small", "Base FLAN-T5-small", ds, device)
    load_and_eval("./checkpoints/flan_t5_sft_cnn_dm", "SFT FLAN-T5-small", ds, device)
    load_and_eval("./checkpoints/flan_t5_ppo_cnn_dm", "PPO FLAN-T5-small", ds, device)

if __name__ == "__main__":
    main()