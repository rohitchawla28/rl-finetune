# src/scst.py
"""
Self-Critical Sequence Training (SCST) for abstractive summarization (T5).

Core idea:
- For each source article x:
  * Generate a greedy (baseline) summary y^g with the current policy.
  * Sample a summary y^s from the current policy.
  * Reward r = ROUGE(y^s, ref) - ROUGE(y^g, ref).
  * Optimize loss L = - E[ r * log p_theta(y^s | x) ]  (advantage = r).
- We compute log p(y^s|x) exactly with teacher forcing over the sampled tokens.

This is a plain PyTorch loop (no TRL dependency).
"""

from typing import List, Dict, Optional, Tuple
import os
import math
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
from transformers import T5ForConditionalGeneration, T5Tokenizer, PreTrainedTokenizerBase, get_linear_schedule_with_warmup
import evaluate

class DictDataset(Dataset):
    """Simple dataset wrapper for list of dictionaries."""
    def __init__(self, data: List[Dict[str, str]]):
        self.data = data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

def _collate_dicts(batch):
    """
    Custom collate function for DataLoader that returns list of dicts as-is.
    PyTorch's default collate tries to collate dicts by key, which doesn't work for our use case.
    """
    # DataLoader passes a list of items from the dataset
    # Just return it as-is
    return batch

# ---------- Data ----------

def build_scst_dataset(
    n_train: int = 1000,
    min_len: int = 200,
    max_len: int = 1200,
    seed: int = 42,
    dataset_name: str = "cnn_dailymail",
) -> List[Dict[str, str]]:
    """
    Returns a list of {"article": <text>, "reference": <gold summary>} from dataset.
    Supports both CNN/DM and XSum.
    """
    if dataset_name == "cnn_dailymail":
        raw = load_dataset("cnn_dailymail", "3.0.0", split="train").shuffle(seed=seed).select(range(n_train))
        def ok(ex):
            # Ensure ex is a dict and has the required keys
            if not isinstance(ex, dict):
                return False
            if "article" not in ex:
                return False
            L = len(ex["article"])
            return (L >= min_len) and (L <= max_len)
        raw = raw.filter(ok)
        # Convert to list to ensure proper iteration
        raw_list = list(raw)
        result = []
        for ex in raw_list:
            if not isinstance(ex, dict):
                raise TypeError(f"Expected dict, got {type(ex)}: {ex}")
            if "article" not in ex or "highlights" not in ex:
                raise KeyError(f"Missing keys in example. Keys: {list(ex.keys()) if isinstance(ex, dict) else 'N/A'}")
            result.append({"article": ex["article"], "reference": ex["highlights"]})
        return result
    elif dataset_name == "xsum":
        raw = load_dataset("xsum", split="train").shuffle(seed=seed).select(range(n_train))
        def ok(ex):
            # Ensure ex is a dict and has the required keys
            if not isinstance(ex, dict):
                return False
            if "document" not in ex:
                return False
            L = len(ex["document"])
            return (L >= min_len) and (L <= max_len)
        raw = raw.filter(ok)
        # Convert to list to ensure proper iteration
        raw_list = list(raw)
        result = []
        for ex in raw_list:
            if not isinstance(ex, dict):
                raise TypeError(f"Expected dict, got {type(ex)}: {ex}")
            if "document" not in ex or "summary" not in ex:
                raise KeyError(f"Missing keys in example. Keys: {list(ex.keys()) if isinstance(ex, dict) else 'N/A'}")
            result.append({"article": ex["document"], "reference": ex["summary"]})
        return result
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

# ---------- Rewards (ROUGE-L per example) ----------

_rouge = evaluate.load("rouge")

def rougeL_list(preds: List[str], refs: List[str]) -> List[float]:
    """
    Return list of ROUGE-L F1 scores, one per example.
    """
    scores = _rouge.compute(
        predictions=preds,
        references=refs,
        rouge_types=["rougeL"],
        use_aggregator=False,
    )["rougeL"]
    return [float(s) for s in scores]

# ---------- Log-prob utilities ----------

def _shift_right(decoder_input_ids: torch.Tensor, start_id: int) -> torch.Tensor:
    """
    For teacher forcing: prepend decoder_start_token_id and remove last token.
    """
    bs = decoder_input_ids.size(0)
    start = torch.full((bs, 1), start_id, dtype=decoder_input_ids.dtype, device=decoder_input_ids.device)
    return torch.cat([start, decoder_input_ids[:, :-1]], dim=1)

@torch.no_grad()
def _generate_texts(
    model: T5ForConditionalGeneration,
    tokenizer: PreTrainedTokenizerBase,
    inputs: List[str],
    device: torch.device,
    *,
    # Greedy (baseline) decode settings
    greedy_num_beams: int = 4,
    greedy_max_new_tokens: int = 64,
    # Sample (exploration) decode settings
    sample_max_new_tokens: int = 64,
    sample_min_new_tokens: int = 8,
    top_k: int = 30,
    top_p: float = 0.8,
    temperature: float = 0.7,
    no_repeat_ngram_size: int = 4,
    max_input_len: int = 512,
) -> Tuple[List[str], List[str]]:
    """
    Returns (greedy_texts, sampled_texts) lists aligned to `inputs`.
    """
    model.eval()

    enc = tokenizer(
        inputs, return_tensors="pt",
        padding=True, truncation=True, max_length=max_input_len
    ).to(device)

    # baseline (greedy / beam)
    greedy_out = model.generate(
        **enc,
        do_sample=False,
        num_beams=greedy_num_beams,
        max_new_tokens=greedy_max_new_tokens,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        use_cache=True,
    )
    greedy_texts = tokenizer.batch_decode(greedy_out, skip_special_tokens=True)

    # sampled
    sampled_out = model.generate(
        **enc,
        do_sample=True,
        top_k=top_k,
        top_p=top_p,
        temperature=temperature,
        no_repeat_ngram_size=no_repeat_ngram_size,
        max_new_tokens=sample_max_new_tokens,
        min_new_tokens=sample_min_new_tokens,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        use_cache=True,
    )
    sampled_texts = tokenizer.batch_decode(sampled_out, skip_special_tokens=True)

    return greedy_texts, sampled_texts

def _seq_logprob(
    model: T5ForConditionalGeneration,
    tokenizer: PreTrainedTokenizerBase,
    src_texts: List[str],
    hyp_texts: List[str],
    device: torch.device,
    max_input_len: int = 512,
    max_target_len: int = 128,
) -> torch.Tensor:
    """
    Compute log p_theta(hyp_text | src_text) per example.
    Implementation: teacher forcing over re-tokenized hyp_texts.
    Returns: tensor of shape [B] with sum of token log-probs.
    """
    model.train()  # ensure grads flow through this pass

    # encode sources
    enc = tokenizer(
        src_texts, return_tensors="pt",
        padding=True, truncation=True, max_length=max_input_len
    ).to(device)
    # encode targets
    tgt = tokenizer(
        hyp_texts, return_tensors="pt",
        padding=True, truncation=True, max_length=max_target_len
    ).to(device)

    # prepare decoder inputs / targets
    decoder_start = model.config.decoder_start_token_id
    if decoder_start is None:
        # T5 usually uses pad as decoder start
        decoder_start = tokenizer.pad_token_id

    # targets (shifted labels)
    labels = tgt["input_ids"]  # [B, T]
    # build decoder inputs
    dec_inp = _shift_right(labels, start_id=decoder_start)

    outputs = model(
        input_ids=enc["input_ids"],
        attention_mask=enc["attention_mask"],
        decoder_input_ids=dec_inp,
        use_cache=False,
    )
    logits = outputs.logits  # [B, T, V]
    log_probs = F.log_softmax(logits, dim=-1)  # [B, T, V]

    # gather per-token log prob of the actual target token
    tgt_lp = torch.gather(log_probs, 2, labels.unsqueeze(-1)).squeeze(-1)  # [B, T]

    # mask out padding positions (use tokenizer.pad_token_id)
    pad_id = tokenizer.pad_token_id
    mask = (labels != pad_id).to(tgt_lp.dtype)  # [B, T]
    seq_logprob = (tgt_lp * mask).sum(dim=1)  # [B]

    return seq_logprob  # sums of token log-probs per example

# ---------- SCST Trainer ----------

def run_scst(
    model_name: str = "google/flan-t5-small",
    out_dir: str = "checkpoints/flan_t5_scst",
    n_train: int = 1000,
    batch_size: int = 4,
    epochs: int = 1,
    lr: float = 1e-6,
    warmup_ratio: float = 0.1,
    # rollout decoding
    greedy_beams: int = 4,
    greedy_max_new: int = 64,
    sample_max_new: int = 64,
    sample_min_new: int = 8,
    top_k: int = 30,
    top_p: float = 0.8,
    temperature: float = 0.7,
    no_repeat_ngram_size: int = 4,
    # lengths
    max_input_len: int = 512,
    max_target_len: int = 128,
    # reward shaping
    advantage_normalize: bool = True,
    reward_clip: Optional[float] = 0.5,  # clip absolute advantage
    # dataset
    dataset_name: str = "cnn_dailymail",
    min_len: int = 200,
    max_len: int = 1200,
    # misc
    seed: int = 42,
    debug: bool = True,
) -> str:
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(out_dir, exist_ok=True)

    tokenizer = T5Tokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = T5ForConditionalGeneration.from_pretrained(model_name).to(device)

    data = build_scst_dataset(
        n_train=n_train, 
        seed=seed,
        dataset_name=dataset_name,
        min_len=min_len,
        max_len=max_len,
    )
    # Wrap data in a proper Dataset class for DataLoader
    dataset = DictDataset(data)
    # Use custom collate function to return list of dicts as-is
    dl = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=_collate_dicts)

    # simple linear warmup + decay on token updates
    total_steps = epochs * math.ceil(len(data) / batch_size)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(warmup_ratio * total_steps),
        num_training_steps=total_steps,
    )

    step_idx = 0
    for ep in range(epochs):
        print(f"\n=== SCST Epoch {ep+1}/{epochs} ===")
        for batch in dl:
            step_idx += 1
            # Debug: Check what batch actually is
            if step_idx == 1 and debug:
                print(f"[SCST DEBUG] Batch type: {type(batch)}")
                print(f"[SCST DEBUG] Batch length: {len(batch) if hasattr(batch, '__len__') else 'N/A'}")
                if isinstance(batch, list) and len(batch) > 0:
                    print(f"[SCST DEBUG] First item type: {type(batch[0])}")
                    if isinstance(batch[0], dict):
                        print(f"[SCST DEBUG] First item keys: {list(batch[0].keys())}")
            
            # Ensure batch is a list of dicts
            if not isinstance(batch, list):
                raise TypeError(f"Expected batch to be a list, got {type(batch)}")
            if len(batch) == 0:
                continue
            if not isinstance(batch[0], dict):
                raise TypeError(f"Expected batch items to be dicts, got {type(batch[0])}. Batch[0] = {batch[0]}")
            
            articles = [ex["article"] for ex in batch]
            refs     = [ex["reference"] for ex in batch]

            # 1) generate baseline (greedy/beam) and sampled summaries
            greedy_txts, sampled_txts = _generate_texts(
                model, tokenizer, articles, device,
                greedy_num_beams=greedy_beams, greedy_max_new_tokens=greedy_max_new,
                sample_max_new_tokens=sample_max_new, sample_min_new_tokens=sample_min_new,
                top_k=top_k, top_p=top_p, temperature=temperature,
                no_repeat_ngram_size=no_repeat_ngram_size, max_input_len=max_input_len,
            )

            # 2) compute rewards per example
            r_g = rougeL_list(greedy_txts, refs)
            r_s = rougeL_list(sampled_txts, refs)
            # advantage = reward(sampled) - reward(greedy)
            adv = torch.tensor([rs - rg for rs, rg in zip(r_s, r_g)], dtype=torch.float32, device=device)

            if advantage_normalize and adv.numel() > 1:
                adv = (adv - adv.mean()) / (adv.std(unbiased=False) + 1e-8)
            if reward_clip is not None:
                adv = torch.clamp(adv, min=-reward_clip, max=reward_clip)

            # 3) compute log p_theta(sampled | x)
            logp = _seq_logprob(
                model, tokenizer,
                src_texts=articles, hyp_texts=sampled_txts,
                device=device,
                max_input_len=max_input_len, max_target_len=max_target_len,
            )  # [B]; sums of token log-probs

            # 4) policy gradient loss = - E[adv * logp]
            loss = -(adv.detach() * logp).mean()

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            if debug and (step_idx % 5 == 0):
                with torch.no_grad():
                    print(
                        f"[SCST][step {step_idx}] "
                        f"loss={loss.item():.4f}  "
                        f"adv_mean={adv.mean().item():.4f}  "
                        f"r_g_mean={float(sum(r_g)/len(r_g)):.4f}  "
                        f"r_s_mean={float(sum(r_s)/len(r_s)):.4f}"
                    )
                    print("--- example 0 ---")
                    print("[QUERY ]", articles[0][:600].replace("\n"," ") + ("..." if len(articles[0])>600 else ""))
                    print("[REF   ]", refs[0][:320].replace("\n"," ") + ("..." if len(refs[0])>320 else ""))
                    print("[GREEDY]", greedy_txts[0][:320].replace("\n"," ") + ("..." if len(greedy_txts[0])>320 else ""))
                    print("[SAMPLED]", sampled_txts[0][:320].replace("\n"," ") + ("..." if len(sampled_txts[0])>320 else ""))

    # save
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    print(f"\nSaved SCST model to {out_dir}")
    return out_dir