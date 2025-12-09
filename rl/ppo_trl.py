# src/rl/ppo_trl.py
from typing import List, Dict
from datasets import load_dataset
from transformers import T5Tokenizer
import torch, evaluate
from trl import PPOConfig, PPOTrainer, AutoModelForSeq2SeqLMWithValueHead

_rouge = evaluate.load("rouge")

def _build_ppo_data(n_train=300, min_len=200, max_len=1200) -> List[Dict[str, str]]:
    raw = load_dataset("cnn_dailymail", "3.0.0", split="train").select(range(n_train))
    raw = raw.filter(lambda ex: min_len <= len(ex["article"]) <= max_len)
    prefix = "summarize: "
    return [{"query": prefix + ex["article"], "reference": ex["highlights"]} for ex in raw]

def _rougeL_per_example(preds: List[str], refs: List[str]) -> List[torch.Tensor]:
    out = _rouge.compute(predictions=preds, references=refs, rouge_types=["rougeL"], use_aggregator=False)
    return [torch.tensor(s, dtype=torch.float32) for s in out["rougeL"]]

def run_ppo(
    model_name: str = "google/flan-t5-small",
    output_dir: str = "./checkpoints/flan_t5_ppo_cnn_dm",
    batch_size: int = 4,
    max_new_tokens: int = 64,
    lr: float = 1e-5,
    ppo_epochs: int = 1,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = T5Tokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForSeq2SeqLMWithValueHead.from_pretrained(model_name).to(device)
    cfg = PPOConfig(
        model_name=model_name,
        learning_rate=lr,
        batch_size=batch_size,
        mini_batch_size=batch_size,
        gradient_accumulation_steps=1,
        optimize_cuda_cache=True,
        target_kl=0.1,
    )
    trainer = PPOTrainer(config=cfg, model=model, ref_model=None, tokenizer=tok, dataset=None)

    data = _build_ppo_data()
    # simple manual batching
    for epoch in range(ppo_epochs):
        print(f"\n=== PPO Epoch {epoch+1}/{ppo_epochs} ===")
        for i in range(0, len(data), batch_size):
            batch = data[i:i+batch_size]
            queries = [b["query"] for b in batch]
            refs = [b["reference"] for b in batch]

            toks = tok(queries, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
            input_ids = toks["input_ids"]

            responses_ids, responses_txt = [], []
            for q in input_ids:
                out = trainer.generate(input_ids=q.unsqueeze(0), max_new_tokens=max_new_tokens,
                                       do_sample=True, top_k=50, top_p=0.95)
                ids = out[0]
                responses_ids.append(ids)
                responses_txt.append(tok.decode(ids, skip_special_tokens=True))

            rewards = _rougeL_per_example(responses_txt, refs)
            stats = trainer.step(query_tensors=list(input_ids), response_tensors=responses_ids, rewards=rewards)

            if (i // batch_size) % 10 == 0:
                print(f"step {(i//batch_size):04d}  mean_reward={torch.stack(rewards).mean().item():.4f}  "
                      f"ppo_loss={stats['ppo/mean_policy_loss']:.4f}")

    trainer.model.save_pretrained(output_dir)
    tok.save_pretrained(output_dir)
    print(f"Saved PPO checkpoint to {output_dir}")