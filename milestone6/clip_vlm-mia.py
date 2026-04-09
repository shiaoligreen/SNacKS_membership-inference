"""
CLIP-Based Image-Text Alignment Membership Inference Attack
===========================================================
Method: CLIP cosine similarity between image and caption as membership score.

Intuition:
  - Members (seen_both): genuine image-caption pairs → high CLIP alignment
  - Non-members: include at least one mismatched component → lower alignment

Model:  OpenAI CLIP ViT-B/32 via open_clip
Data:   UBC-SLIME/colx585_group_project_data
Target: UBC-SLIME/colx_585_vlm (SmolVLMForConditionalGeneration)

Reference:
  Radford et al. (2021) "Learning Transferable Visual Models From
  Natural Language Supervision." ICML 2021.

Usage:
    python clip_mia.py --split validation
    python clip_mia.py --split test
    python clip_mia.py --split train
"""

import argparse
import io
import os

import numpy as np
import open_clip
import pandas as pd
import torch
from datasets import load_dataset
from PIL import Image
from sklearn.metrics import roc_auc_score, roc_curve
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DATASET_ID      = "UBC-SLIME/colx585_group_project_data"
CLIP_MODEL      = "ViT-B-32"
CLIP_PRETRAINED = "openai"

CHECKPOINT_DIR  = "./clip_checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
CHECKPOINT_EVERY = 500


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_clip_model(device: str):
    print(f"Loading CLIP {CLIP_MODEL} ({CLIP_PRETRAINED}) ...")
    model, _, preprocess = open_clip.create_model_and_transforms(
        CLIP_MODEL, pretrained=CLIP_PRETRAINED
    )
    tokenizer = open_clip.get_tokenizer(CLIP_MODEL)
    model     = model.to(device).eval()
    print("CLIP model loaded.")
    return model, preprocess, tokenizer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_image(example) -> Image.Image:
    if isinstance(example["image"], dict) and "bytes" in example["image"]:
        return Image.open(io.BytesIO(example["image"]["bytes"])).convert("RGB")
    return example["image"].convert("RGB")


def get_caption(example) -> str:
    """Extract only the assistant (caption) part of the text field."""
    text = example["text"]
    return text.split("\n", 1)[1] if "\n" in text else text


@torch.no_grad()
def clip_alignment_score(
    example,
    model,
    preprocess,
    tokenizer,
    device: str,
) -> float:
    """
    Compute CLIP cosine similarity between image and caption.

    Steps:
      1. Preprocess image and encode with CLIP vision encoder.
      2. Tokenize caption (max 77 tokens) and encode with CLIP text encoder.
      3. L2-normalise both embeddings.
      4. Return cosine similarity as membership score.

    Higher score → image and caption are well-aligned → more likely a member.
    """
    image   = load_image(example)
    caption = get_caption(example)

    # Encode image
    image_tensor   = preprocess(image).unsqueeze(0).to(device)
    image_features = model.encode_image(image_tensor)
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)

    # Encode caption (CLIP max token length = 77)
    text_tokens   = tokenizer([caption[:77]]).to(device)
    text_features = model.encode_text(text_tokens)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    # Cosine similarity
    similarity = (image_features * text_features).sum(dim=-1)
    return float(similarity.item())


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def tpr_at_fpr(y_true, y_score, target_fpr: float = 0.1) -> float:
    fpr, tpr, _ = roc_curve(y_true, y_score)
    valid = np.where(fpr <= target_fpr)[0]
    return float(np.max(tpr[valid])) if len(valid) > 0 else 0.0


def evaluate(df: pd.DataFrame, split_name: str):
    if "label" not in df.columns or df["label"].isna().all():
        print(f"{split_name}: no labels available (test split).")
        return
    auc   = roc_auc_score(df["label"], df["score"])
    tpr01 = tpr_at_fpr(df["label"].values, df["score"].values)
    print(f"[{split_name}]  AUC = {auc:.4f}  |  TPR@FPR=0.1 = {tpr01:.4f}")


def evaluate_per_type(df: pd.DataFrame):
    """Per-type AUC breakdown for report error analysis."""
    if "type" not in df.columns or "label" not in df.columns:
        return
    print("\nPer-type AUC breakdown:")
    for t, group in df.groupby("type"):
        if group["label"].nunique() > 1:
            auc = roc_auc_score(group["label"], group["score"])
            print(f"  {t:<30}  AUC = {auc:.4f}  (n={len(group)})")

    print("\nMean CLIP score per type:")
    for t, group in df.groupby("type"):
        print(f"  {t:<30}  mean = {group['score'].mean():.4f}  "
              f"(n={len(group)})")


# ---------------------------------------------------------------------------
# Scoring loop
# ---------------------------------------------------------------------------

def score_split(
    split_name: str,
    model,
    preprocess,
    tokenizer,
    device: str,
) -> pd.DataFrame:

    ckpt_path = os.path.join(CHECKPOINT_DIR, f"{split_name}_scores.csv")

    # Resume from checkpoint if available
    completed_ids: set = set()
    records = []
    if os.path.exists(ckpt_path):
        ckpt_df = pd.read_csv(ckpt_path)
        records = ckpt_df.to_dict("records")
        completed_ids = set(ckpt_df["id"].tolist())
        print(f"Resuming from checkpoint: {len(completed_ids)} samples "
              f"already scored.")

    ds = load_dataset(DATASET_ID, split=split_name)

    for i, example in enumerate(tqdm(ds, desc=f"CLIP scoring {split_name}")):
        sample_id = example["id"]
        if sample_id in completed_ids:
            continue

        try:
            score = clip_alignment_score(
                example, model, preprocess, tokenizer, device
            )
            record = {"id": sample_id, "score": score}
            if example.get("is_member") is not None:
                record["label"] = example["is_member"]
            if example.get("type") is not None:
                record["type"] = example["type"]
            records.append(record)

        except Exception as e:
            print(f"  Error on sample {sample_id}: {e} — skipping.")
            records.append({"id": sample_id, "score": 0.0})

        # Periodic checkpoint
        if (i + 1) % CHECKPOINT_EVERY == 0:
            pd.DataFrame(records).to_csv(ckpt_path, index=False)
            print(f"  Checkpoint saved at sample {i + 1}.")

    df = pd.DataFrame(records)
    df.to_csv(ckpt_path, index=False)
    print(f"Scoring complete. Results saved to {ckpt_path}")
    return df


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(split="validation", output=None):
    device = get_device()
    print(f"Device: {device}")

    model, preprocess, tokenizer = load_clip_model(device)

    df = score_split(split, model, preprocess, tokenizer, device)

    # Evaluate if labels available
    evaluate(df, split)
    evaluate_per_type(df)

    # Save Kaggle submission CSV
    output_path = output or f"submission_clip_{split}.csv"
    submission  = df[["id", "score"]].rename(columns={"score": "is_member"})
    submission.to_csv(output_path, index=False)
    print(f"\nKaggle submission saved: {output_path}  ({len(submission)} rows)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="CLIP-based image-text alignment MIA"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="validation",
        choices=["train", "validation", "test"],
        help="Dataset split to score (default: validation).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path for Kaggle submission CSV. "
             "Defaults to submission_clip_<split>.csv",
    )
    args = parser.parse_args()
    main(split=args.split, output=args.output)
