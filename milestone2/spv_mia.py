"""
SPV-MIA: Self-Prompt Calibration Membership Inference Attack
Based on: Fu et al., NeurIPS 2024
Author: Nicole Shantz

Usage:
    # Score validation split and get AUC + TPR@FPR=0.1:
    python spv_mia.py --split validation

    # Score test split and save Kaggle submission CSV:
    python spv_mia.py --split test

    # Reload checkpointed scores (skip re-scoring):
    python spv_mia.py --split validation --reload

    # Enable PVA module (warning: ~27hrs on 15k samples):
    python spv_mia.py --split validation --use_pva
"""

try:
    import unsloth
except ImportError:
    pass

import os
import gc
import argparse
import random
import json

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
)
from sklearn.metrics import roc_auc_score, roc_curve


# 1. CLI Arguments
def parse_args():
    parser = argparse.ArgumentParser(description="SPV-MIA membership inference attack")
    parser.add_argument("--split", choices=["validation", "test"], default="validation",
                        help="Which dataset split to score (default: validation)")
    parser.add_argument("--reload", action="store_true",
                        help="Reload checkpointed scores instead of re-running inference")
    parser.add_argument("--use_pva", action="store_true",
                        help="Enable PVA module (slow: ~27hrs for 15k samples on T4)")
    parser.add_argument("--target_model", default="UBC-SLIME/colx_531_smollm2-360m")
    parser.add_argument("--ref_base_model", default="HuggingFaceTB/SmolLM2-360M")
    parser.add_argument("--ref_model_dir", default="./spv_ref_model")
    parser.add_argument("--dataset", default="UBC-SLIME/colx_531_group_project")
    parser.add_argument("--n_generate", type=int, default=1000,
                        help="Number of self-prompt texts to generate")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Scoring batch size (reduce if OOM)")
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--ckpt_dir", default="./spv_checkpoints")
    parser.add_argument("--output_csv", default="submission_spv_mia.csv")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()



# 2. Setup
def setup(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    os.environ["UNSLOTH_RETURN_LOGITS"] = "1"

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if device == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        _ampere = any(x in gpu_name for x in ["A100", "A10", "L4", "H100", "A6000", "3090", "4090"])
        dtype = torch.bfloat16 if _ampere else torch.float16
        print(f"GPU  : {gpu_name}")
        print(f"Dtype: {'bfloat16' if dtype == torch.bfloat16 else 'float16'}")
    else:
        dtype = torch.float32
        print("Running on CPU")

    # Try to import Unsloth for faster inference
    try:
        from unsloth import FastLanguageModel
        unsloth_available = True
        print("✓ Unsloth available")
    except ImportError:
        unsloth_available = False
        print("  Unsloth not found — using standard HF loader")

    # Flash Attention 2
    flash_attn = False
    if device == "cuda":
        try:
            import flash_attn
            flash_attn = True
            print("✓ Flash Attention 2 available")
        except ImportError:
            print("  Flash Attention 2 not available")

    os.makedirs(args.ckpt_dir, exist_ok=True)

    return device, dtype, unsloth_available, flash_attn



# 3. Model Loading
def load_causal_model(name, device, dtype, unsloth_available, flash_attn, for_inference=True):
    print(f"  Loading {name} ...")

    # AutoTokenizer fails after Unsloth patches transformers because the model's
    # tokenizer_config.json specifies "TokenizersBackend" which isn't registered.
    # Bypass AutoTokenizer entirely and load PreTrainedTokenizerFast directly.
    from transformers import PreTrainedTokenizerFast
    try:
        tok = PreTrainedTokenizerFast.from_pretrained(name)
    except Exception:
        from transformers import LlamaTokenizerFast
        tok = LlamaTokenizerFast.from_pretrained(name)

    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    attn_kwargs = {"attn_implementation": "flash_attention_2"} if flash_attn else {}
    model = AutoModelForCausalLM.from_pretrained(
        name, torch_dtype=dtype, **attn_kwargs).to(device)
    if unsloth_available and device == "cuda" and for_inference:
        from unsloth import FastLanguageModel
        FastLanguageModel.for_inference(model)

    if for_inference:
        model.eval()

    loader = "Unsloth+HF" if (unsloth_available and device == "cuda") else "HF"
    print(f"  ✓ Loaded ({dtype}, {loader})")
    return tok, model



# 4. Self-Prompt Generation (PDC Module)
def generate_self_prompt_texts(target_tok, target_model, seed_texts, device,
                                n_generate, prompt_len=32, max_new_tokens=128,
                                batch_size=8):
    print(f"Generating {n_generate} self-prompt texts ...")
    generated = []
    seeds = random.choices(seed_texts, k=n_generate)

    for i in tqdm(range(0, n_generate, batch_size)):
        batch = seeds[i: i + batch_size]
        prompts = [target_tok.decode(
            target_tok(s, truncation=True, max_length=prompt_len)["input_ids"],
            skip_special_tokens=True) for s in batch]
        inputs = target_tok(
            prompts, return_tensors="pt",
            truncation=True, max_length=prompt_len, padding=True).to(device)
        with torch.no_grad():
            out = target_model.generate(
                **inputs, max_new_tokens=max_new_tokens,
                do_sample=True, temperature=0.9, top_p=0.95,
                pad_token_id=target_tok.eos_token_id)
        for seq in out:
            generated.append(target_tok.decode(seq, skip_special_tokens=True))

    return generated[:n_generate]


def save_reference_model(ref_base_model, ref_model_dir):
    """Save unfine-tuned base model as reference (fine-tuning skipped due to
    Unsloth patch conflicts in the same session as inference)."""
    print("Saving base model as reference ...")
    from transformers import PreTrainedTokenizerFast
    try:
        tok_tmp = PreTrainedTokenizerFast.from_pretrained(ref_base_model)
    except Exception:
        from transformers import LlamaTokenizerFast
        tok_tmp = LlamaTokenizerFast.from_pretrained(ref_base_model)
    if tok_tmp.pad_token is None:
        tok_tmp.pad_token = tok_tmp.eos_token
    model_tmp = AutoModelForCausalLM.from_pretrained(ref_base_model, torch_dtype=torch.float32)
    os.makedirs(ref_model_dir, exist_ok=True)
    model_tmp.save_pretrained(ref_model_dir)
    tok_tmp.save_pretrained(ref_model_dir)
    del model_tmp
    gc.collect()
    print(f"✓ Base model saved as reference → {ref_model_dir}")



# 5. PVA Module (optional, slow)
class SemanticParaphraser:
    """Symmetric paraphrase pairs via T5-base (Algorithm 2, Fu et al. 2024)."""
    T5_NAME = "t5-base"

    def __init__(self, device, dtype, flash_attn, mask_ratio=0.2, n_pairs=5):
        self.mask_ratio = mask_ratio
        self.n_pairs = n_pairs
        self.device = device
        print(f"  Loading T5 ({self.T5_NAME}) ...")
        self.t5_tok = AutoTokenizer.from_pretrained(self.T5_NAME, use_fast=False)
        attn_kwargs = {"attn_implementation": "flash_attention_2"} if flash_attn else {}
        self.t5 = AutoModelForSeq2SeqLM.from_pretrained(
            self.T5_NAME, torch_dtype=dtype, **attn_kwargs).to(device).eval()
        with torch.no_grad():
            self.emb_matrix = self.t5.shared.weight.float().cpu()

    def _nearest(self, emb):
        return int((self.emb_matrix - emb.unsqueeze(0)).norm(dim=1).argmin().item())

    def paraphrase(self, text):
        ids = self.t5_tok.encode(text, truncation=True, max_length=256)
        if len(ids) < 4:
            return []
        pairs = []
        for _ in range(self.n_pairs):
            masked = list(ids)
            mi = [j for j in range(len(masked)) if random.random() < self.mask_ratio]
            if not mi:
                continue
            for j in mi:
                masked[j] = self.t5_tok.mask_token_id or 32099
            enc = self.t5_tok(
                self.t5_tok.decode(masked, skip_special_tokens=False),
                return_tensors="pt", truncation=True, max_length=256).to(self.device)
            with torch.no_grad():
                plus_ids = self.t5.generate(**enc, max_new_tokens=len(ids) + 10)[0].cpu().tolist()
            plus_text = self.t5_tok.decode(plus_ids, skip_special_tokens=True)
            minus_ids = list(ids)
            for j in mi:
                eo = self.emb_matrix[ids[j]].float()
                ep = self.emb_matrix[plus_ids[j] if j < len(plus_ids) else ids[j]].float()
                minus_ids[j] = self._nearest(2.0 * eo - ep)
            pairs.append((plus_text, self.t5_tok.decode(minus_ids, skip_special_tokens=True)))
        return pairs



# 6. Scoring Functions
def score_batch(model, tok, batch_texts, device, max_length=512, use_greybox=True):
    losses, hs_norms = [], []
    for text in batch_texts:
        enc = tok(text, return_tensors="pt", truncation=True,
                  max_length=max_length, padding=False).to(device)
        if enc["input_ids"].shape[1] < 2:
            losses.append(float("nan"))
            hs_norms.append(float("nan"))
            continue
        with torch.no_grad():
            out = model(**enc, labels=enc["input_ids"],
                        output_hidden_states=use_greybox)
        losses.append(out.loss.item())
        if use_greybox:
            am = enc["attention_mask"].float()
            hs = ((out.hidden_states[-1].norm(dim=-1) * am).sum(1)
                  / am.sum(1).clamp(min=1)).float().cpu().item()
            hs_norms.append(hs)
        else:
            hs_norms.append(float("nan"))
    return np.array(losses, dtype=float), np.array(hs_norms, dtype=float)


def get_loss_single(model, tok, text, device, max_length=512):
    enc = tok(text, return_tensors="pt", truncation=True,
              max_length=max_length, padding=False).to(device)
    if enc["input_ids"].shape[1] < 2:
        return float("nan")
    with torch.no_grad():
        return model(**enc, labels=enc["input_ids"]).loss.item()


def pva_score(text, model, tok, paraphraser, device, max_length=512):
    orig = get_loss_single(model, tok, text, device, max_length)
    if np.isnan(orig) or paraphraser is None:
        return float("nan")
    nbr = [(get_loss_single(model, tok, xp, device, max_length) +
            get_loss_single(model, tok, xm, device, max_length)) / 2
           for xp, xm in paraphraser.paraphrase(text)]
    nbr = [v for v in nbr if not np.isnan(v)]
    return float(np.mean(nbr) - orig) if nbr else float("nan")


def z_norm(a):
    return (a - a.mean()) / (a.std() + 1e-8)



# 7. Evaluation
def evaluate(pdc_arr, hs_arr, pva_arr, spv_arr, labels, valid_idx, split, use_greybox, use_pva):
    valid_labels = np.array([labels[i] for i in valid_idx])

    def report(name, scores, lbls):
        auc = roc_auc_score(lbls, scores)
        fpr, tpr, _ = roc_curve(lbls, scores)
        tpr01 = float(tpr[min(np.searchsorted(fpr, 0.1), len(tpr) - 1)])
        print(f"  {name:<42}  AUC={auc:.4f}   TPR@FPR=0.1={tpr01:.4f}")
        return auc, tpr01

    print("=" * 65)
    print(f"  SPV-MIA Results  (split={split})")
    print("=" * 65)
    report("PDC only  (self-prompt LiRA)", pdc_arr, valid_labels)
    if use_greybox and not np.all(np.isnan(hs_arr)):
        report("Grey-box  (hidden-state norm)", z_norm(hs_arr), valid_labels)
    if use_pva:
        report("PVA only  (prob. curvature)", pva_arr, valid_labels)
    report("SPV-MIA   (combined)", spv_arr, valid_labels)
    print("=" * 65)



# 8. Save Submission CSV
def save_submission(valid_ids, spv_arr, pdc_arr, hs_arr, pva_arr, all_ids, output_csv):
    scored_df = pd.DataFrame({"id": valid_ids, "is_member": spv_arr})
    full_df = pd.DataFrame({"id": all_ids})
    submission_df = full_df.merge(scored_df, on="id", how="left").fillna(0.0)
    submission_df["id"] = submission_df["id"].astype(int)

    assert len(submission_df) == len(all_ids), \
        f"Row count mismatch: got {len(submission_df)}, expected {len(all_ids)}"
    assert submission_df["id"].nunique() == len(all_ids), \
        "Duplicate IDs detected."

    submission_df[["id", "is_member"]].to_csv(output_csv, index=False)

    all_scores_csv = output_csv.replace(".csv", "_all_scores.csv")
    pd.DataFrame({
        "id": valid_ids, "pdc": pdc_arr,
        "hs_norm": hs_arr, "pva": pva_arr, "spv": spv_arr,
    }).to_csv(all_scores_csv, index=False)

    print(f"✓ Submission CSV → {output_csv}")
    print(f"  Rows   : {len(submission_df)}")
    print(f"  Scores : [{submission_df['is_member'].min():.4f}, {submission_df['is_member'].max():.4f}]")
    print(f"  All signal scores → {all_scores_csv}")



# 9. Main
def main():
    args = parse_args()
    device, dtype, unsloth_available, flash_attn = setup(args)

    print(f"\nConfiguration:")
    print(f"  split      = {args.split}")
    print(f"  reload     = {args.reload}")
    print(f"  use_pva    = {args.use_pva}")
    print(f"  batch_size = {args.batch_size}")
    print(f"  n_generate = {args.n_generate}")

    # Reload from checkpoint (skip all inference)
  
    if args.reload:
        print(f"\nReloading scores from {args.ckpt_dir}/ ...")
        pdc_arr   = np.load(f"{args.ckpt_dir}/pdc.npy")
        hs_arr    = np.load(f"{args.ckpt_dir}/hs.npy")
        pva_arr   = np.load(f"{args.ckpt_dir}/pva.npy")
        spv_arr   = np.load(f"{args.ckpt_dir}/spv.npy")
        valid_ids = np.load(f"{args.ckpt_dir}/ids.npy").tolist()
        print(f"✓ Reloaded {len(pdc_arr)} scored samples")

        ds      = load_dataset(args.dataset, split=args.split)
        all_ids = list(ds["id"])
        labels  = None
        if "is_member" in ds.column_names:
            raw = ds["is_member"]
            if any(v is not None for v in raw):
                labels = np.array(raw)

        valid_idx = [all_ids.index(i) for i in valid_ids]

        if labels is not None:
            evaluate(pdc_arr, hs_arr, pva_arr, spv_arr, labels,
                     valid_idx, args.split, args.use_pva, args.use_pva)

        save_submission(valid_ids, spv_arr, pdc_arr, hs_arr, pva_arr, all_ids, args.output_csv)
        return

   
    # Step 1: Load target model
    print("\n--- Loading target model ---")
    target_tok, target_model = load_causal_model(
        args.target_model, device, dtype, unsloth_available, flash_attn)


    # Step 2: Generate self-prompt texts + save reference model
    generated_path = f"{args.ckpt_dir}/generated_texts.txt"
    if os.path.exists(generated_path):
        print(f"\nLoading cached self-prompt texts from {generated_path} ...")
        with open(generated_path) as f:
            generated_texts = f.read().split("\n---\n")
        print(f"✓ Loaded {len(generated_texts)} texts")
    else:
        print("\n--- Generating self-prompt texts (PDC) ---")
        train_ds   = load_dataset(args.dataset, split="train")
        seed_texts = list(train_ds["text"])[:5000]
        generated_texts = generate_self_prompt_texts(
            target_tok, target_model, seed_texts, device, args.n_generate)
        with open(generated_path, "w") as f:
            f.write("\n---\n".join(generated_texts))
        print(f"✓ {len(generated_texts)} texts saved → {generated_path}")

    if not os.path.exists(args.ref_model_dir):
        save_reference_model(args.ref_base_model, args.ref_model_dir)

    # Step 3: Load reference model
    print("\n--- Loading reference model ---")
    ref_tok, ref_model = load_causal_model(
        args.ref_model_dir, device, dtype, unsloth_available, flash_attn)


    # Step 4: PVA paraphraser (optional)
    if args.use_pva:
        print("\n--- Loading PVA paraphraser ---")
        paraphraser = SemanticParaphraser(device, dtype, flash_attn,
                                          mask_ratio=0.2, n_pairs=5)
        print("✓ PVA paraphraser ready")
    else:
        paraphraser = None
        print("\nPVA disabled — PDC + grey-box only")

  
    # Step 5: Load dataset split
    print(f"\n--- Loading {args.dataset} ({args.split} split) ---")
    ds      = load_dataset(args.dataset, split=args.split)
    texts   = list(ds["text"])
    all_ids = list(ds["id"])
    labels  = None
    if "is_member" in ds.column_names:
        raw = ds["is_member"]
        if any(v is not None for v in raw):
            labels = np.array(raw)
    print(f"  {len(texts)} samples  |  labels={'available' if labels is not None else 'N/A'}")

    # Step 6: Scoring loop
    t_ls, r_ls, hs_ls, pva_ls, valid_idx = [], [], [], [], []

    print(f"\nScoring {len(texts)} samples ...")
    for start in tqdm(range(0, len(texts), args.batch_size)):
        end   = min(start + args.batch_size, len(texts))
        batch = texts[start:end]

        tl, th = score_batch(target_model, target_tok, batch, device, args.max_length, True)
        rl, _  = score_batch(ref_model,    ref_tok,    batch, device, args.max_length, False)

        for j in range(len(batch)):
            if np.isnan(tl[j]) or np.isnan(rl[j]):
                continue
            pva = pva_score(batch[j], target_model, target_tok, paraphraser,
                            device, args.max_length) if args.use_pva else 0.0
            if args.use_pva and np.isnan(pva):
                continue
            t_ls.append(tl[j])
            r_ls.append(rl[j])
            hs_ls.append(th[j])
            pva_ls.append(pva)
            valid_idx.append(start + j)

    pdc_arr = np.array(r_ls) - np.array(t_ls)
    hs_arr  = np.array(hs_ls)
    pva_arr = np.array(pva_ls)
    spv_arr = z_norm(pdc_arr)
    if not np.all(np.isnan(hs_arr)):
        spv_arr = spv_arr + z_norm(hs_arr)
    if args.use_pva:
        spv_arr = spv_arr + z_norm(pva_arr)
    valid_ids = [all_ids[i] for i in valid_idx]

    # Checkpoint
    for fname, arr in [("pdc", pdc_arr), ("hs", hs_arr), ("pva", pva_arr),
                       ("spv", spv_arr), ("ids", np.array(valid_ids))]:
        np.save(f"{args.ckpt_dir}/{fname}.npy", arr)
    print(f"\n✓ Scored {len(valid_idx)} / {len(texts)} samples  |  checkpointed → {args.ckpt_dir}/")

    # Step 7: Evaluate (validation only)
    if labels is not None:
        evaluate(pdc_arr, hs_arr, pva_arr, spv_arr, labels,
                 valid_idx, args.split, True, args.use_pva)

    # Step 8: Save submission CSV
    save_submission(valid_ids, spv_arr, pdc_arr, hs_arr, pva_arr, all_ids, args.output_csv)


if __name__ == "__main__":
    main()
