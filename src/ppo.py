# src/ppo.py
"""
Cleaner PPO with verbosity control, progress bar, and optional CSV logging.
Tested with: trl==0.10.1, transformers==4.57.x, torch==2.x
"""

from typing import List, Dict, Optional, Literal
import os, math, csv, json, torch
from datasets import load_dataset
from transformers import T5Tokenizer
from trl import PPOConfig, PPOTrainer, AutoModelForSeq2SeqLMWithValueHead
from tqdm.auto import tqdm

# local or package import for rewards
try:
    from rewards import rouge_l_rewards
except ModuleNotFoundError:
    from src.rewards import rouge_l_rewards


# ----------------------------
# Data
# ----------------------------
def build_ppo_dataset(
    n_train: int = 300,
    min_len: int = 200,
    max_len: int = 1200,
    use_instruction: bool = False,
    seed: int = 42,
) -> List[Dict[str, str]]:
    raw = load_dataset("cnn_dailymail", "3.0.0", split="train")
    raw = raw.shuffle(seed=seed).select(range(n_train))

    def ok(ex):
        L = len(ex["article"])
        return (L >= min_len) and (L <= max_len)

    raw = raw.filter(ok)

    def to_pair(ex):
        article = ex["article"]
        if use_instruction:
            q = f"Summarize the following article.\n\n{article}\n\nSummary:"
        else:
            q = article
        return {"query": q, "reference": ex["highlights"]}

    return [to_pair(ex) for ex in raw]


# ----------------------------
# Utils
# ----------------------------
def _device_str(device: Optional[str] = None) -> str:
    if device is not None:
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"

def _pick(d: Dict, *keys, default=float("nan")):
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return default

def _safe_isfinite(name: str, t: torch.Tensor):
    if not torch.isfinite(t).all():
        raise ValueError(f"Non-finite values detected in {name}")

def _first_n(text: str, n: int = 200) -> str:
    return text if len(text) <= n else text[:n] + " …"

def _kl_coef_from(trainer: PPOTrainer, stats: Dict) -> float:
    val = _pick(stats, "ppo/kl_coef", "train/kl_coef", default=float("nan"))
    if isinstance(val, float) and not math.isnan(val):
        return val
    kl_ctl = getattr(trainer, "kl_ctl", None)
    if kl_ctl is not None:
        for attr in ("value", "kl_coef", "coeff", "coef"):
            v = getattr(kl_ctl, attr, None)
            if isinstance(v, float):
                return v
    return float("nan")


# ----------------------------
# Main
# ----------------------------
def run_ppo(
    model_name: str = "google/flan-t5-small",
    out_dir: str = "checkpoints/flan_t5_ppo_from_base",

    # data / batching
    n_train: int = 300,
    steps_per_epoch: int = 30,
    epochs: int = 1,
    batch_size: int = 4,
    remainder: Literal["skip", "wrap"] = "skip",
    seed: int = 42,

    # generation (safe defaults)
    max_new_tokens: int = 56,
    min_new_tokens: int = 8,
    no_repeat_ngram_size: int = 3,
    top_k: int = 30,
    top_p: float = 0.85,
    temperature: float = 0.8,
    use_instruction: bool = False,

    # PPO knobs
    lr: float = 1e-6,
    ppo_epochs: int = 2,
    target_kl: float = 0.08,
    cliprange: float = 0.2,
    cliprange_value: float = 0.2,
    max_grad_norm: float = 1.0,
    whiten_rewards: bool = False,
    adap_kl_ctrl: bool = True,
    init_kl_coef: float = 0.2,

    # rewards
    length_penalty: float = 0.0,
    reward_clamp: float = 0.5,  # clamp each reward to [-0.5, 0.5]
    reward_fn=None,
    reward_kwargs: Optional[dict] = None,

    # output control
    verbosity: Literal["silent", "summary", "steps", "verbose"] = "summary",
    print_examples: int = 0,          # set >0 only if verbosity="verbose"
    tqdm_bar: bool = True,
    log_csv: Optional[str] = None,     # e.g., "ppo_log.csv"

    # misc
    device: Optional[str] = None,
) -> str:
    """
    Returns: out_dir (where checkpoint is saved)
    """
    device = _device_str(device)
    torch.manual_seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(seed)

    # Verbosity helpers
    LEVELS = {"silent": 0, "summary": 1, "steps": 2, "verbose": 3}
    V = LEVELS[verbosity]

    def vlog(level: str, msg: str):
        if LEVELS[level] <= V:
            print(msg)

    # Tokenizer & policy
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForSeq2SeqLMWithValueHead.from_pretrained(model_name).to(device)

    # Optional bf16 + grad checkpointing for longer runs (kept quiet)
    try:
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            model.to(torch.bfloat16)
        model.gradient_checkpointing_enable()
    except Exception:
        pass

    # TRL config (0.10.1)
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

    ppo_trainer = PPOTrainer(
        cfg,
        model,
        ref_model=None,
        tokenizer=tokenizer,
        dataset=None,
    )

    # Data
    data = build_ppo_dataset(
        n_train=n_train, use_instruction=use_instruction, seed=seed
    )
    if V >= 1:
        print(f"[PPO] device={device} n_train={len(data)} epochs={epochs} steps/epoch={steps_per_epoch} bs={batch_size}")

    # Reward function
    if reward_fn is None:
        reward_fn = lambda preds, refs, **kw: rouge_l_rewards(preds, refs, length_penalty=length_penalty)
    rw_kwargs = reward_kwargs or {}

    # CSV logging
    csv_writer = None
    csv_f = None
    if log_csv:
        os.makedirs(os.path.dirname(log_csv) if os.path.dirname(log_csv) else ".", exist_ok=True)
        csv_f = open(log_csv, "w", newline="")
        csv_writer = csv.DictWriter(csv_f, fieldnames=[
            "epoch","step","mean_reward","kl","policy_loss","value_loss","entropy","kl_coef"
        ])
        csv_writer.writeheader()

    # Progress bar
    total_steps = epochs * steps_per_epoch
    pbar = tqdm(total=total_steps, disable=(V == 0 or not tqdm_bar), leave=False)

    # Aggregates for summary printing
    def agg_reset():
        return {"rewards": [], "kls": [], "pol_losses": [], "val_losses": [], "ents": [], "kl_coefs": []}
    agg = agg_reset()

    step_idx_global = 0

    for ep in range(epochs):
        if V >= 1:
            print(f"\n=== PPO Epoch {ep + 1}/{epochs} ===")

        ptr, left = 0, steps_per_epoch
        while ptr < len(data) and left > 0:
            batch = data[ptr: ptr + batch_size]
            ptr += batch_size
            left -= 1
            step_idx_global += 1

            if len(batch) < batch_size:
                if remainder == "skip":
                    if V >= 2:
                        print(f"[PPO] Skipping short last batch len={len(batch)}.")
                    break
                else:
                    need = batch_size - len(batch)
                    batch = batch + data[:need]

            queries = [ex["query"] for ex in batch]
            refs    = [ex["reference"] for ex in batch]

            # Encode
            enc = tokenizer(
                queries, return_tensors="pt", padding=True, truncation=True, max_length=512
            ).to(device)
            _safe_isfinite("queries/input_ids", enc["input_ids"])

            # Generate
            with torch.no_grad():
                gen = model.generate(
                    **enc,
                    max_new_tokens=max_new_tokens,
                    min_new_tokens=min_new_tokens,
                    do_sample=True,
                    top_k=top_k, top_p=top_p, temperature=temperature,
                    no_repeat_ngram_size=no_repeat_ngram_size,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )

            response_tensors = [g for g in gen]
            response_texts   = tokenizer.batch_decode(gen, skip_special_tokens=True)
            resp_ids = torch.stack(response_tensors, dim=0)
            _safe_isfinite("responses", resp_ids)

            # Rewards
            rewards = reward_fn(response_texts, refs, **rw_kwargs)
            rewards = [torch.clamp(r.float(), -reward_clamp, reward_clamp) for r in rewards]
            rewards = [torch.where(torch.isfinite(r), r, torch.zeros_like(r)) for r in rewards]

            # PPO update (exact batch_size)
            stats = ppo_trainer.step(
                [q for q in enc["input_ids"]],
                response_tensors,
                rewards,
            )

            # Extract stats robustly across TRL variants
            mean_reward = torch.stack(rewards).mean().item()
            kl = _pick(stats, "objective/kl", "ppo/policy/kl", "kl")
            pol_loss = _pick(stats, "ppo/loss/policy", "train/mean_policy_loss", "ppo/mean_policy_loss")
            val_loss = _pick(stats, "ppo/loss/value", "train/mean_value_loss", "ppo/mean_value_loss")
            ent = _pick(stats, "ppo/policy/entropy", "policy/entropy")
            kl_coef = _kl_coef_from(ppo_trainer, stats)

            # Aggregate for summaries
            agg["rewards"].append(mean_reward)
            agg["kls"].append(kl if isinstance(kl, float) else float("nan"))
            agg["pol_losses"].append(pol_loss if isinstance(pol_loss, float) else float("nan"))
            agg["val_losses"].append(val_loss if isinstance(val_loss, float) else float("nan"))
            agg["ents"].append(ent if isinstance(ent, float) else float("nan"))
            agg["kl_coefs"].append(kl_coef if isinstance(kl_coef, float) else float("nan"))

            # Step-level prints
            if V >= 2:
                print(f"[step {step_idx_global}] reward={mean_reward:.4f} kl={kl:.4f} kl_coef={kl_coef:.4f} "
                      f"policy_loss={pol_loss:.4f} value_loss={val_loss:.4f} entropy={ent:.4f}")

            # Optional example dump
            if V >= 3 and print_examples > 0:
                for i in range(min(print_examples, len(batch))):
                    print(f"--- example {i} ---")
                    print("[QUERY ]", _first_n(queries[i], 240))
                    print("[REF   ]", _first_n(refs[i], 200))
                    print("[OUTPUT]", _first_n(response_texts[i], 200))
                    print(f"[REWARD] {rewards[i].item():.4f}")

            # CSV logging
            if csv_writer is not None:
                csv_writer.writerow({
                    "epoch": ep + 1, "step": step_idx_global,
                    "mean_reward": f"{mean_reward:.6f}",
                    "kl": f"{kl:.6f}" if isinstance(kl, float) else "",
                    "policy_loss": f"{pol_loss:.6f}" if isinstance(pol_loss, float) else "",
                    "value_loss": f"{val_loss:.6f}" if isinstance(val_loss, float) else "",
                    "entropy": f"{ent:.6f}" if isinstance(ent, float) else "",
                    "kl_coef": f"{kl_coef:.6f}" if isinstance(kl_coef, float) else "",
                })

            if pbar is not None:
                pbar.update(1)

        # Epoch summary
        if V >= 1:
            def _m(xs): 
                xs = [x for x in xs if isinstance(x, float) and math.isfinite(x)]
                return sum(xs)/len(xs) if xs else float("nan")
            print(f"[epoch {ep+1}] "
                  f"reward_mean={_m(agg['rewards']):.4f} "
                  f"kl={_m(agg['kls']):.4f} "
                  f"policy_loss={_m(agg['pol_losses']):.4f} "
                  f"value_loss={_m(agg['val_losses']):.4f} "
                  f"entropy={_m(agg['ents']):.4f} "
                  f"kl_coef={_m(agg['kl_coefs']):.4f}")
            agg = agg_reset()

    if pbar is not None:
        pbar.close()

    # Save checkpoint + config
    os.makedirs(out_dir, exist_ok=True)
    ppo_trainer.model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    with open(os.path.join(out_dir, "ppo_run_config.json"), "w") as f:
        json.dump({
            "model_name": model_name,
            "n_train": n_train,
            "steps_per_epoch": steps_per_epoch,
            "epochs": epochs,
            "batch_size": batch_size,
            "gen": dict(max_new_tokens=max_new_tokens, min_new_tokens=min_new_tokens,
                        top_k=top_k, top_p=top_p, temperature=temperature,
                        no_repeat_ngram_size=no_repeat_ngram_size),
            "ppo": dict(lr=lr, ppo_epochs=ppo_epochs, target_kl=target_kl,
                        cliprange=cliprange, cliprange_value=cliprange_value,
                        max_grad_norm=max_grad_norm, init_kl_coef=init_kl_coef,
                        adap_kl_ctrl=adap_kl_ctrl, whiten_rewards=whiten_rewards),
        }, f, indent=2)

    if csv_f is not None:
        csv_f.close()

    print(f"\nSaved PPO model to {out_dir}")
    return out_dir