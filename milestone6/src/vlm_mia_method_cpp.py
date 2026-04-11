#!/usr/bin/env python3
"""
Membership inference attack for UBC-SLIME/colx_585_vlm.

Pipeline:
1. Load train / validation / test splits
2. Extract pair-level and contrastive features
3. Train a HistGradientBoostingClassifier
4. Evaluate on validation (AUC and TPR@FPR=0.1)
5. Produce a submission CSV with columns: id, is_member

This script is a cleaned .py version of the notebook workflow.
"""

from __future__ import annotations

import argparse
import io
import os
import random
import warnings
import zlib
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from datasets import load_dataset
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, roc_curve
from tqdm import tqdm
from transformers import AutoProcessor, SmolVLMForConditionalGeneration

warnings.filterwarnings("ignore")

MODEL_ID = "UBC-SLIME/colx_585_vlm"
DATASET_ID = "UBC-SLIME/colx585_group_project_data"

FEATURE_COLS_CPP = [
    "pos_loss",
    "worst20_mean_loss",
    "cap_neg_mean",
    "cap_neg_min",
    "cap_neg_std",
    "cap_margin_mean",
    "cap_margin_hard",
    "combined_margin_mean",
    "combined_margin_hard",
    "ratio_mean",
    "ratio_hard",
    "caption_word_len",
    "caption_char_len",
    "caption_zlib_len",
    "zlib_loss",
    "img_neg_mean",
    "img_margin_mean",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run membership inference attack for the VLM task.")
    parser.add_argument("--max-train", type=int, default=None, help="Optional cap for train rows.")
    parser.add_argument("--max-val", type=int, default=None, help="Optional cap for validation rows.")
    parser.add_argument("--max-test", type=int, default=None, help="Optional cap for test rows.")
    parser.add_argument("--k-caption-neg", type=int, default=8, help="Number of caption negatives.")
    parser.add_argument("--k-image-neg", type=int, default=8, help="Number of image negatives.")
    parser.add_argument("--caption-pool-size", type=int, default=30, help="Pool size for hard caption negatives.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--output-dir", type=str, default="outputs_vlm_mia", help="Directory for outputs.")
    parser.add_argument(
        "--ensemble-weight",
        type=float,
        default=0.2,
        help="Weight for handcrafted score when ensembling with classifier output.",
    )
    parser.add_argument(
        "--ensemble-feature",
        type=str,
        default="cap_margin_mean",
        choices=["cap_margin_mean", "combined_margin_hard", "combined_margin_mean"],
        help="Handcrafted feature used in the ensemble.",
    )
    parser.add_argument(
        "--use-ensemble",
        action="store_true",
        help="Use ensemble of classifier score + handcrafted feature for final submission.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_image(example: Dict) -> Image.Image:
    img = example["image"]
    if isinstance(img, dict) and "bytes" in img:
        return Image.open(io.BytesIO(img["bytes"])).convert("RGB")
    return img.convert("RGB")


def split_text_pair(text: str) -> Tuple[str, str]:
    parts = text.split("\n", 1)
    user_text = parts[0]
    assistant_text = parts[1] if len(parts) > 1 else ""
    return user_text, assistant_text


def build_messages(full_text: str) -> List[Dict]:
    user_text, assistant_text = split_text_pair(full_text)
    return [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": user_text},
            ],
        },
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": assistant_text},
            ],
        },
    ]


def compressed_length(text: str) -> int:
    return len(zlib.compress(text.encode("utf-8")))


def caption_stats(text: str) -> Dict[str, int]:
    return {
        "caption_char_len": len(text),
        "caption_word_len": len(text.split()),
        "caption_zlib_len": compressed_length(text),
    }


def text_overlap_score(text_a: str, text_b: str) -> float:
    words_a = set(text_a.lower().split())
    words_b = set(text_b.lower().split())
    if len(words_a) == 0 and len(words_b) == 0:
        return 0.0
    return len(words_a & words_b) / max(1, len(words_a | words_b))


def caption_hardness_score(text_i: str, text_j: str) -> float:
    stats_i = caption_stats(text_i)
    stats_j = caption_stats(text_j)

    word_len_diff = abs(stats_i["caption_word_len"] - stats_j["caption_word_len"])
    char_len_diff = abs(stats_i["caption_char_len"] - stats_j["caption_char_len"])
    zlib_len_diff = abs(stats_i["caption_zlib_len"] - stats_j["caption_zlib_len"])
    overlap = text_overlap_score(text_i, text_j)

    # Higher score = harder negative
    score = (
        overlap
        - 0.01 * word_len_diff
        - 0.001 * char_len_diff
        - 0.01 * zlib_len_diff
    )
    return score


def load_model_and_processor(device: str):
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    processor.image_processor.do_image_splitting = False
    processor.image_processor.size = {"longest_edge": 512}
    processor.image_processor.max_image_size = {"longest_edge": 512}

    dtype = torch.float16 if device == "cuda" else torch.float32

    model = SmolVLMForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=dtype,
        trust_remote_code=True,
    ).to(device)

    model.eval()
    return model, processor


def prepare_inputs(image: Image.Image, full_text: str, processor, device: str):
    messages = build_messages(full_text)
    text = processor.apply_chat_template(messages, tokenize=False)
    inputs = processor(text=text, images=[image], return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    labels = inputs["input_ids"].clone()
    labels[labels == processor.tokenizer.pad_token_id] = -100
    return inputs, labels


@torch.no_grad()
def compute_pair_loss_and_token_stats(
    image: Image.Image,
    full_text: str,
    model,
    processor,
    device: str,
) -> Dict[str, float]:
    inputs, labels = prepare_inputs(image, full_text, processor, device)

    outputs = model(**inputs)
    logits = outputs.logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    vocab_size = logits.size(-1)

    loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
    token_losses = loss_fct(logits.view(-1, vocab_size), shift_labels.view(-1)).view(shift_labels.shape)

    valid_mask = shift_labels != -100
    valid_token_losses = token_losses[valid_mask]

    if valid_token_losses.numel() == 0:
        return {
            "pos_loss": np.nan,
            "token_mean_loss": np.nan,
            "token_max_loss": np.nan,
            "token_p90_loss": np.nan,
            "token_std_loss": np.nan,
            "worst20_mean_loss": np.nan,
        }

    token_losses_np = valid_token_losses.detach().float().cpu().numpy()

    pos_loss = float(token_losses_np.mean())
    token_mean_loss = float(token_losses_np.mean())
    token_max_loss = float(token_losses_np.max())
    token_p90_loss = float(np.percentile(token_losses_np, 90))
    token_std_loss = float(token_losses_np.std())

    k = max(1, int(0.2 * len(token_losses_np)))
    worst20 = np.sort(token_losses_np)[-k:]
    worst20_mean_loss = float(worst20.mean())

    return {
        "pos_loss": pos_loss,
        "token_mean_loss": token_mean_loss,
        "token_max_loss": token_max_loss,
        "token_p90_loss": token_p90_loss,
        "token_std_loss": token_std_loss,
        "worst20_mean_loss": worst20_mean_loss,
    }


def load_split(split_name: str, max_examples: Optional[int] = None) -> List[Dict]:
    ds = load_dataset(DATASET_ID, split=split_name)
    records: List[Dict] = []

    for ex in ds:
        row: Dict[str, object] = {
            "id": ex["id"],
            "image": ex["image"],
            "text": ex["text"],
        }

        if "is_member" in ex:
            row["label"] = ex["is_member"]
        if "type" in ex:
            row["type"] = ex["type"]

        records.append(row)

        if max_examples is not None and len(records) >= max_examples:
            break

    return records


def build_hard_caption_negative_map(
    records: Sequence[Dict],
    k_caption_neg: int = 8,
    pool_size: int = 30,
    seed: int = 42,
) -> Dict[int, List[int]]:
    rng = random.Random(seed)
    n = len(records)
    cap_map: Dict[int, List[int]] = {}

    for i in range(n):
        text_i = records[i]["text"]
        candidates = list(range(n))
        candidates.remove(i)

        if pool_size < len(candidates):
            candidates = rng.sample(candidates, pool_size)

        scored = []
        for j in candidates:
            text_j = records[j]["text"]
            hardness = caption_hardness_score(text_i, text_j)
            scored.append((j, hardness))

        scored.sort(key=lambda x: x[1], reverse=True)
        cap_map[i] = [j for j, _ in scored[:k_caption_neg]]

    return cap_map


def build_random_image_negative_map(n: int, k_image_neg: int, seed: int = 42) -> Dict[int, List[int]]:
    rng = random.Random(seed)
    img_map: Dict[int, List[int]] = {}

    for i in range(n):
        candidates = list(range(n))
        candidates.remove(i)
        if k_image_neg >= len(candidates):
            img_map[i] = candidates
        else:
            img_map[i] = rng.sample(candidates, k_image_neg)

    return img_map


def compute_features_for_split(
    records: List[Dict],
    model,
    processor,
    device: str,
    k_caption_neg: int,
    k_image_neg: int,
    caption_pool_size: int,
    seed: int,
) -> pd.DataFrame:
    print(f"Extracting features with k_caption_neg={k_caption_neg}, k_image_neg={k_image_neg}")
    n = len(records)

    cap_neg_map = build_hard_caption_negative_map(
        records,
        k_caption_neg=k_caption_neg,
        pool_size=caption_pool_size,
        seed=seed,
    )
    img_neg_map = build_random_image_negative_map(
        n=n,
        k_image_neg=k_image_neg,
        seed=seed,
    )

    results: List[Dict[str, object]] = []

    for idx in tqdm(range(n), desc="Extracting features"):
        ex = records[idx]
        try:
            image_i = load_image(ex)
            text_i = ex["text"]

            pos_stats = compute_pair_loss_and_token_stats(image_i, text_i, model, processor, device)
            pos_loss = pos_stats["pos_loss"]

            caption_neg_losses: List[float] = []
            for j in cap_neg_map[idx]:
                text_j = records[j]["text"]
                neg_stats = compute_pair_loss_and_token_stats(image_i, text_j, model, processor, device)
                caption_neg_losses.append(neg_stats["pos_loss"])

            image_neg_losses: List[float] = []
            for j in img_neg_map[idx]:
                image_j = load_image(records[j])
                neg_stats = compute_pair_loss_and_token_stats(image_j, text_i, model, processor, device)
                image_neg_losses.append(neg_stats["pos_loss"])

            caption_neg_losses = [x for x in caption_neg_losses if not np.isnan(x)]
            image_neg_losses = [x for x in image_neg_losses if not np.isnan(x)]
            all_neg = caption_neg_losses + image_neg_losses

            cap_neg_mean = float(np.mean(caption_neg_losses)) if caption_neg_losses else np.nan
            cap_neg_min = float(np.min(caption_neg_losses)) if caption_neg_losses else np.nan
            cap_neg_std = float(np.std(caption_neg_losses)) if caption_neg_losses else np.nan

            img_neg_mean = float(np.mean(image_neg_losses)) if image_neg_losses else np.nan
            img_neg_min = float(np.min(image_neg_losses)) if image_neg_losses else np.nan
            img_neg_std = float(np.std(image_neg_losses)) if image_neg_losses else np.nan

            combined_neg_mean = float(np.mean(all_neg)) if all_neg else np.nan
            combined_neg_min = float(np.min(all_neg)) if all_neg else np.nan

            cap_margin_mean = cap_neg_mean - pos_loss if not np.isnan(cap_neg_mean) else np.nan
            img_margin_mean = img_neg_mean - pos_loss if not np.isnan(img_neg_mean) else np.nan
            combined_margin_mean = combined_neg_mean - pos_loss if not np.isnan(combined_neg_mean) else np.nan

            cap_margin_hard = cap_neg_min - pos_loss if not np.isnan(cap_neg_min) else np.nan
            img_margin_hard = img_neg_min - pos_loss if not np.isnan(img_neg_min) else np.nan
            combined_margin_hard = combined_neg_min - pos_loss if not np.isnan(combined_neg_min) else np.nan

            all_losses = all_neg + [pos_loss]
            if len(all_losses) > 0:
                rank = sum(l <= pos_loss for l in all_losses) / len(all_losses)
                rank_score = 1.0 - rank
            else:
                rank_score = np.nan

            ratio_mean = pos_loss / (combined_neg_mean + 1e-6) if not np.isnan(combined_neg_mean) else np.nan
            ratio_hard = pos_loss / (combined_neg_min + 1e-6) if not np.isnan(combined_neg_min) else np.nan

            cstats = caption_stats(text_i)
            zlib_loss = pos_loss / (cstats["caption_zlib_len"] + 1e-6)

            row: Dict[str, object] = {
                "id": ex["id"],
                "pos_loss": pos_loss,
                "token_mean_loss": pos_stats["token_mean_loss"],
                "token_max_loss": pos_stats["token_max_loss"],
                "token_p90_loss": pos_stats["token_p90_loss"],
                "token_std_loss": pos_stats["token_std_loss"],
                "worst20_mean_loss": pos_stats["worst20_mean_loss"],
                "cap_neg_mean": cap_neg_mean,
                "cap_neg_min": cap_neg_min,
                "cap_neg_std": cap_neg_std,
                "img_neg_mean": img_neg_mean,
                "img_neg_min": img_neg_min,
                "img_neg_std": img_neg_std,
                "combined_neg_mean": combined_neg_mean,
                "combined_neg_min": combined_neg_min,
                "cap_margin_mean": cap_margin_mean,
                "img_margin_mean": img_margin_mean,
                "combined_margin_mean": combined_margin_mean,
                "cap_margin_hard": cap_margin_hard,
                "img_margin_hard": img_margin_hard,
                "combined_margin_hard": combined_margin_hard,
                "rank_score": rank_score,
                "ratio_mean": ratio_mean,
                "ratio_hard": ratio_hard,
                "caption_word_len": cstats["caption_word_len"],
                "caption_char_len": cstats["caption_char_len"],
                "caption_zlib_len": cstats["caption_zlib_len"],
                "zlib_loss": zlib_loss,
            }

            if "label" in ex:
                row["label"] = ex["label"]
            if "type" in ex:
                row["type"] = ex["type"]

            results.append(row)

        except Exception as e:  # pragma: no cover
            print(f"Skipped {ex['id']} due to error: {e}")

    return pd.DataFrame(results)


def clean_feature_frames(frames: Sequence[pd.DataFrame], feature_cols: Sequence[str]) -> None:
    for df in frames:
        df[list(feature_cols)] = df[list(feature_cols)].replace([np.inf, -np.inf], np.nan)
        df[list(feature_cols)] = df[list(feature_cols)].fillna(0)


def tpr_at_fpr(y_true, y_score, target_fpr: float = 0.1) -> float:
    fpr, tpr, _ = roc_curve(y_true, y_score)
    valid = np.where(fpr <= target_fpr)[0]
    if len(valid) == 0:
        return 0.0
    return float(np.max(tpr[valid]))


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    set_seed(args.seed)
    device = get_device()
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    model, processor = load_model_and_processor(device)
    print("Model loaded.")

    train_records = load_split("train", args.max_train)
    val_records = load_split("validation", args.max_val)
    test_records = load_split("test", args.max_test)

    print(f"Train: {len(train_records)}")
    print(f"Val:   {len(val_records)}")
    print(f"Test:  {len(test_records)}")

    df_train = compute_features_for_split(
        train_records,
        model,
        processor,
        device,
        k_caption_neg=args.k_caption_neg,
        k_image_neg=args.k_image_neg,
        caption_pool_size=args.caption_pool_size,
        seed=args.seed,
    )
    df_val = compute_features_for_split(
        val_records,
        model,
        processor,
        device,
        k_caption_neg=args.k_caption_neg,
        k_image_neg=args.k_image_neg,
        caption_pool_size=args.caption_pool_size,
        seed=args.seed,
    )
    df_test = compute_features_for_split(
        test_records,
        model,
        processor,
        device,
        k_caption_neg=args.k_caption_neg,
        k_image_neg=args.k_image_neg,
        caption_pool_size=args.caption_pool_size,
        seed=args.seed,
    )

    df_train.to_csv(os.path.join(args.output_dir, "train_features.csv"), index=False)
    df_val.to_csv(os.path.join(args.output_dir, "val_features.csv"), index=False)
    df_test.to_csv(os.path.join(args.output_dir, "test_features.csv"), index=False)

    clean_feature_frames([df_train, df_val, df_test], FEATURE_COLS_CPP)
    print("Feature cleaning done.")

    X_train = df_train[FEATURE_COLS_CPP]
    y_train = df_train["label"].values
    X_val = df_val[FEATURE_COLS_CPP]
    y_val = df_val["label"].values

    clf = HistGradientBoostingClassifier(
        max_depth=4,
        learning_rate=0.05,
        max_iter=300,
        min_samples_leaf=20,
        random_state=args.seed,
    )
    clf.fit(X_train, y_train)
    print("Classifier trained.")

    df_val["score_cpp"] = clf.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, df_val["score_cpp"])
    tpr01 = tpr_at_fpr(y_val, df_val["score_cpp"], target_fpr=0.1)
    print(f"[Method C++] AUC = {auc:.4f} | TPR@FPR=0.1 = {tpr01:.4f}")

    if args.use_ensemble:
        ens_col = "score_cpp_ens"
        df_val[ens_col] = (
            (1.0 - args.ensemble_weight) * df_val["score_cpp"]
            + args.ensemble_weight * df_val[args.ensemble_feature]
        )
        auc_ens = roc_auc_score(y_val, df_val[ens_col])
        tpr01_ens = tpr_at_fpr(y_val, df_val[ens_col], target_fpr=0.1)
        print(f"[Method C++ Ensemble] AUC = {auc_ens:.4f} | TPR@FPR=0.1 = {tpr01_ens:.4f}")

    df_test["score_cpp"] = clf.predict_proba(df_test[FEATURE_COLS_CPP])[:, 1]
    if args.use_ensemble:
        df_test["is_member"] = (
            (1.0 - args.ensemble_weight) * df_test["score_cpp"]
            + args.ensemble_weight * df_test[args.ensemble_feature]
        )
    else:
        df_test["is_member"] = df_test["score_cpp"]

    submission = df_test[["id", "is_member"]].copy().sort_values("id")
    submission_path = os.path.join(args.output_dir, "submission_method_cpp.csv")
    submission.to_csv(submission_path, index=False)
    print(f"Saved submission to: {submission_path}")
    print(submission.head())


if __name__ == "__main__":
    main()
