# eval_summarizer.py
from datasets import load_dataset
from transformers import T5ForConditionalGeneration, T5Tokenizer
import evaluate
import torch

def generate_summaries(model, tokenizer, articles, max_input_len=512, max_target_len=128, device="cuda"):
    preds = []
    refs = []
    model.to(device)
    model.eval()
    for ex in articles:
        article = ex["article"]
        ref = ex["highlights"]

        inputs = tokenizer(
            article,
            return_tensors="pt",
            truncation=True,
            max_length=max_input_len,
        ).to(device)

        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_length=max_target_len,
                num_beams=4,
            )

        pred = tokenizer.decode(output[0], skip_special_tokens=True)
        preds.append(pred)
        refs.append(ref)
    return preds, refs

def main():
    rouge = evaluate.load("rouge")

    # small val subset
    ds = load_dataset("cnn_dailymail", "3.0.0")["validation"].select(range(100))

    # 1) base model
    base_name = "google/flan-t5-small"
    base_tok = T5Tokenizer.from_pretrained(base_name)
    base_model = T5ForConditionalGeneration.from_pretrained(base_name)
    base_preds, base_refs = generate_summaries(base_model, base_tok, ds)
    base_scores = rouge.compute(predictions=base_preds, references=base_refs)
    print("Base FLAN-T5-small ROUGE:", base_scores)

    # 2) SFT model
    ckpt_dir = "./checkpoints/flan_t5_sft_cnn_dm"
    sft_tok = T5Tokenizer.from_pretrained(ckpt_dir)
    sft_model = T5ForConditionalGeneration.from_pretrained(ckpt_dir)
    sft_preds, sft_refs = generate_summaries(sft_model, sft_tok, ds)
    sft_scores = rouge.compute(predictions=sft_preds, references=sft_refs)
    print("SFT FLAN-T5-small ROUGE:", sft_scores)

if __name__ == "__main__":
    main()