# src/ppo.py
"""
PPO for CNN/DailyMail summarization with TRL-generate compatibility.

- Detects PPOTrainer.generate signature (query_tensor vs query_tensors) and adapts
- Filters-by-length FIRST, then shuffle/select
- Uses TRL generate to keep policy/ref aligned
- Safe defaults to limit negative KL and NaNs
"""

from typing import List, Dict, Optional, Literal
import os, math, csv, json, torch, inspect
from datasets import load_dataset
from transformers import T5Tokenizer
from trl import PPOConfig, PPOTrainer, AutoModelForSeq2SeqLMWithValueHead
from tqdm.auto import tqdm

try:
    from rewards import rouge_l_rewards
except ModuleNotFoundError:
    from src.rewards import rouge_l_rewards


def _device_str(device: Optional[str] = None) -> str:
    return device or ("cuda" if torch.cuda.is_available() else "cpu")

def _pick(d: Dict, *keys, default=float("nan")):
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return default

def _safe_isfinite(name: str, t: torch.Tensor):
    if not torch.isfinite(t).all():
        raise ValueError(f"Non-finite values detected in {name}")

def _filter_by_len(ds, min_len: int, max_len: int):
    def ok(ex):
        L = len(ex["article"])
        return (L >= min_len) and (L <= max_len)
    return ds.filter(ok)

def _ppo_generate(ppo_trainer: PPOTrainer, enc: Dict[str, torch.Tensor], **gen_kwargs):
    """
    Compatibility wrapper for TRL's generate across versions:
      - If signature has 'query_tensors': call once with a list (batched).
      - Else (expects 'query_tensor'): loop per sample.
    Returns a list of response tensors (one per example).
    """
    sig = inspect.signature(ppo_trainer.generate)
    am = enc.get("attention_mask", None)

    if "query_tensors" in sig.parameters:
        gens = ppo_trainer.generate(
            query_tensors=[row for row in enc["input_ids"]],
            attention_mask=am,
            return_prompt=False,
            **gen_kwargs,
        )
        # TRL usually returns a single padded tensor [B, T]; normalize to list
        if torch.is_tensor(gens):
            return [row for row in gens]
        return list(gens)

    # Older TRL: per-sample 'query_tensor'
    outs = []
    B = enc["input_ids"].size(0)
    for i in range(B):
        q = enc["input_ids"][i]
        am_i = am[i] if am is not None else None
        g = ppo_trainer.generate(
            query_tensor=q,
            attention_mask=am_i,
            return_prompt=False,
            **gen_kwargs,
        )
        # normalize: g may be tensor or [tensor]
        if isinstance(g, list):
            if len(g) == 0:
                continue
            g = g[0]
        outs.append(g)
    return outs


def run_ppo(
    model_name: str = "google/flan-t5-small",
    out_dir: str = "checkpoints/flan_t5_ppo_from_base",

    pairs: Optional[List[Dict[str, str]]] = None,  # optional prebuilt {query, reference}

    # data controls (used only if pairs is None)
    n_train: int = 300,
    min_len: int = 200,
    max_len: int = 1200,
    use_instruction: bool = False,
    seed: int = 42,

    # schedule
    batch_size: int = 4,
    steps_per_epoch: Optional[int] = None,   # None => ceil(len(data)/bs)
    epochs: int = 1,
    remainder: Literal["skip", "wrap"] = "skip",

    # generation (safe defaults)
    max_new_tokens: int = 56,
    min_new_tokens: int = 8,
    no_repeat_ngram_size: int = 4,
    top_k: int = 30,
    top_p: float = 0.75,
    temperature: float = 0.7,

    # PPO knobs
    lr: float = 1e-6,
    ppo_epochs: int = 2,
    target_kl: float = 0.08,
    cliprange: float = 0.2,
    cliprange_value: float = 0.2,
    max_grad_norm: float = 1.0,
    whiten_rewards: bool = False,
    adap_kl_ctrl: bool = True,
    init_kl_coef: float = 0.4,

    # rewards
    length_penalty: float = 0.0,
    reward_clamp: float = 0.5,
    reward_fn=None,
    reward_kwargs: Optional[dict] = None,

    # logging
    verbosity: Literal["silent", "summary", "steps"] = "summary",
    tqdm_bar: bool = True,
    log_csv: Optional[str] = None,

    # misc
    device: Optional[str] = None,
) -> str:
    device = _device_str(device)
    torch.manual_seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(seed)

    LVL = {"silent": 0, "summary": 1, "steps": 2}
    V = LVL[verbosity]

    # tokenizer + policy/value (fp32 for stability)
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForSeq2SeqLMWithValueHead.from_pretrained(model_name).to(device)

    cfg = PPOConfig(
        model_name=model_name,
        learning_rate=lr,
        batch_size=batch_size,
        mini_batch_size=batch_size,
        gradient_accumulation_steps=1,
        ppo_epochs=ppo_epochs,
        target_kl=target_kl,
        cliprange=cliprange,
        cliprange_value=cliprange_value,
        max_grad_norm=max_grad_norm,
        whiten_rewards=whiten_rewards,
        adap_kl_ctrl=adap_kl_ctrl,
        init_kl_coef=init_kl_coef,
        seed=seed,
        optimize_cuda_cache=True,
    )
    ppo_trainer = PPOTrainer(cfg, model, ref_model=None, tokenizer=tokenizer, dataset=None)

    # data
    if pairs is None:
        raw = load_dataset("cnn_dailymail", "3.0.0", split="train")
        raw = _filter_by_len(raw, min_len=min_len, max_len=max_len)
        survivors = len(raw)
        raw = raw.shuffle(seed=seed)
        take = min(n_train, survivors)
        raw = raw.select(range(take))
        if use_instruction:
            def make_q(a): return f"Summarize the following article.\n\n{a}\n\nSummary:"
        else:
            def make_q(a): return a
        data = [{"query": make_q(ex["article"]), "reference": ex["highlights"]} for ex in raw]
        if V >= 1:
            print(f"[PPO] survivors after filter(min_len={min_len}, max_len={max_len}) = {survivors}")
            print(f"[PPO] taking n_train={take}")
    else:
        data = pairs
        if V >= 1:
            print(f"[PPO] using provided pairs n={len(data)}")

    data_len = len(data)
    auto_steps = math.ceil(data_len / batch_size)
    steps_per_epoch = auto_steps if steps_per_epoch is None else min(steps_per_epoch, auto_steps)
    if V >= 1:
        print(f"[PPO] device={device} n_train={data_len} epochs={epochs} steps/epoch={steps_per_epoch} bs={batch_size}")

    # rewards
    if reward_fn is None:
        reward_fn = lambda preds, refs, **kw: rouge_l_rewards(preds, refs, length_penalty=length_penalty)
    rw_kwargs = reward_kwargs or {}

    # gen kwargs (passed to TRL generate)
    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        min_new_tokens=min_new_tokens,
        do_sample=True,
        top_k=top_k,
        top_p=top_p,
        temperature=temperature,
        no_repeat_ngram_size=no_repeat_ngram_size,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    # CSV logging
    csv_writer = None
    csv_f = None
    if log_csv:
        os.makedirs(os.path.dirname(log_csv) or ".", exist_ok=True)
        csv_f = open(log_csv, "w", newline="")
        csv_writer = csv.DictWriter(csv_f, fieldnames=[
            "epoch","step","mean_reward","kl","policy_loss","value_loss","entropy","kl_coef"
        ])
        csv_writer.writeheader()

    total_steps = epochs * steps_per_epoch
    pbar = tqdm(total=total_steps, disable=(V == 0 or not tqdm_bar), leave=False)

    step_idx = 0
    for ep in range(epochs):
        if V >= 1:
            print(f"\n=== PPO Epoch {ep+1}/{epochs} ===")
        ptr, left = 0, steps_per_epoch

        while ptr < data_len and left > 0:
            batch = data[ptr: ptr + batch_size]
            ptr += batch_size
            left -= 1
            step_idx += 1

            if len(batch) < batch_size:
                if remainder == "skip":
                    if V >= 2:
                        print(f"[PPO] Skipping short batch len={len(batch)}")
                    break
                need = batch_size - len(batch)
                batch = batch + data[:need]

            queries = [ex["query"] for ex in batch]
            refs    = [ex["reference"] for ex in batch]

            # encode
            enc = tokenizer(queries, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
            _safe_isfinite("queries/input_ids", enc["input_ids"])

            # generate via TRL (signature-safe)
            responses = _ppo_generate(ppo_trainer, enc, **gen_kwargs)   # -> list[tensor]
            # sanity check finiteness per response
            for i, r in enumerate(responses):
                _safe_isfinite(f"responses[{i}]", r)

            # decode texts (accepts list[List[int]])
            decode_in = [r.tolist() if torch.is_tensor(r) else r for r in responses]
            response_texts = tokenizer.batch_decode(decode_in, skip_special_tokens=True)

            # rewards
            rewards = reward_fn(response_texts, refs, **rw_kwargs)
            rewards = [torch.clamp(r.float(), -reward_clamp, reward_clamp) for r in rewards]
            rewards = [torch.where(torch.isfinite(r), r, torch.zeros_like(r)) for r in rewards]

            # PPO update
            stats = ppo_trainer.step([q for q in enc["input_ids"]], responses, rewards)
            mean_reward = torch.stack(rewards).mean().item()
            kl = _pick(stats, "objective/kl", "ppo/policy/kl", "kl")
            pol_loss = _pick(stats, "ppo/loss/policy", "train/mean_policy_loss", "ppo/mean_policy_loss")
            val_loss = _pick(stats, "ppo/loss/value", "train/mean_value_loss", "ppo/mean_value_loss")
            ent = _pick(stats, "ppo/policy/entropy", "policy/entropy")
            kl_coef = _pick(stats, "ppo/kl_coef", "train/kl_coef")

            if V >= 2:
                print(f"[step {step_idx}] reward={mean_reward:.4f} kl={kl:.4f} "
                      f"policy_loss={pol_loss:.4f} value_loss={val_loss:.4f} "
                      f"entropy={ent:.4f} kl_coef={kl_coef if isinstance(kl_coef, float) else float('nan'):.4f}")

            if csv_writer is not None:
                csv_writer.writerow({
                    "epoch": ep+1, "step": step_idx,
                    "mean_reward": f"{mean_reward:.6f}",
                    "kl": f"{kl:.6f}" if isinstance(kl, float) else "",
                    "policy_loss": f"{pol_loss:.6f}" if isinstance(pol_loss, float) else "",
                    "value_loss": f"{val_loss:.6f}" if isinstance(val_loss, float) else "",
                    "entropy": f"{ent:.6f}" if isinstance(ent, float) else "",
                    "kl_coef": f"{kl_coef:.6f}" if isinstance(kl_coef, float) else "",
                })

            if pbar is not None:
                pbar.update(1)

    if pbar is not None:
        pbar.close()
    if csv_f is not None:
        csv_f.close()

    # save
    os.makedirs(out_dir, exist_ok=True)
    ppo_trainer.model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    with open(os.path.join(out_dir, "ppo_run_config.json"), "w") as f:
        json.dump({
            "model_name": model_name,
            "n_train": data_len,
            "steps_per_epoch": steps_per_epoch,
            "epochs": epochs,
            "batch_size": batch_size,
            "min_len": min_len,
            "max_len": max_len,
            "gen": dict(
                max_new_tokens=max_new_tokens,
                min_new_tokens=min_new_tokens,
                top_k=top_k,
                top_p=top_p,
                temperature=temperature,
                no_repeat_ngram_size=no_repeat_ngram_size,
            ),
            "ppo": dict(
                lr=lr, ppo_epochs=ppo_epochs, target_kl=target_kl,
                cliprange=cliprange, cliprange_value=cliprange_value,
                max_grad_norm=max_grad_norm, init_kl_coef=init_kl_coef,
                adap_kl_ctrl=adap_kl_ctrl, whiten_rewards=whiten_rewards,
            ),
        }, f, indent=2)

    print(f"\nSaved PPO model to {out_dir}")
    return out_dir