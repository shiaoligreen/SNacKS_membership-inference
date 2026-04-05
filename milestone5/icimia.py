# coded with Gemini and used a Kaggle GPU notebook to run this

import io
import torch
import torchvision.transforms as T
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import roc_curve, roc_auc_score
from datasets import load_dataset
from transformers import AutoProcessor, SmolVLMForConditionalGeneration
from tqdm import tqdm

# ==========================================
# 1. Setup and Initialization
# ==========================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_ID = "UBC-SLIME/colx_585_vlm"
DATASET_ID = "UBC-SLIME/colx585_group_project_data"

print(f"Using device: {DEVICE}")
print("Loading model and processor...")
processor = AutoProcessor.from_pretrained(MODEL_ID)
model = SmolVLMForConditionalGeneration.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16 if DEVICE == "cuda" else torch.float32,
).to(DEVICE)
model.eval()

print("Loading Dev and Test datasets...")
dev_ds = load_dataset(DATASET_ID, split="validation")
test_ds = load_dataset(DATASET_ID, split="test")

# Define Image Corruption (Gaussian Blur) - ICIMIA Core
corrupt_image = T.GaussianBlur(kernel_size=9, sigma=(5.0, 5.0))

# ==========================================
# 2. Helper & Method Definitions
# ==========================================
def get_pil_image(image_data):
    if isinstance(image_data, dict) and "bytes" in image_data:
        return Image.open(io.BytesIO(image_data["bytes"])).convert("RGB")
    return image_data.convert("RGB")

def compute_loss(image, caption, user_prompt="Describe this image in detail."):
    """Computes the exact VLM cross-entropy loss for generating the caption given the image."""
    messages = [
        {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": user_prompt}]},
        {"role": "assistant", "content": [{"type": "text", "text": caption}]}
    ]
    text = processor.apply_chat_template(messages, tokenize=False)
    inputs = processor(text=text, images=[image], return_tensors="pt").to(DEVICE)
    
    if DEVICE == "cuda":
        inputs = {k: v.to(torch.bfloat16) if v.dtype == torch.float32 else v for k, v in inputs.items()}
    
    labels = inputs["input_ids"].clone()
    labels[labels == processor.tokenizer.pad_token_id] = -100
    
    with torch.no_grad():
        outputs = model(**inputs, labels=labels)
        return outputs.loss.item()

def calculate_metrics(y_true, scores):
    """Calculates AUC and TPR at FPR=0.1"""
    if len(np.unique(y_true)) < 2:
        return float('nan'), float('nan')
    auc = roc_auc_score(y_true, scores)
    fpr, tpr, _ = roc_curve(y_true, scores)
    valid_tprs = tpr[fpr <= 0.10]
    tpr_at_fpr_10 = valid_tprs[-1] if len(valid_tprs) > 0 else 0.0
    return auc, tpr_at_fpr_10

# ==========================================
# 3. Report Evaluation (Dev Set)
# ==========================================
def evaluate_pair_icimia():
    y_true, baseline_loss_scores, icimia_gap_scores = [], [], []
    
    print(f"\n--- Phase 1: Evaluating Pair-Level ICIMIA on {len(dev_ds)} Dev Samples ---")
    for sample in tqdm(dev_ds, desc="Processing Dev Set"):
        # 1 = Seen Pair, 0 = Unseen/Mismatched Pair
        label = sample.get("is_member", sample.get("label", 0))
        y_true.append(label)
        
        img = get_pil_image(sample["image"])
        img_corr = corrupt_image(img)
        caption = sample.get("text", sample.get("caption", ""))
        
        # Compute Loss on Original and Corrupted Images
        loss_orig = compute_loss(img, caption)
        loss_corr = compute_loss(img_corr, caption)
        
        # Metric 1: Baseline Loss (Negated so higher = member)
        baseline_loss_scores.append(-loss_orig)
        
        # Metric 2: ICIMIA Loss Gap (Loss_Corr - Loss_Orig)
        # Members are ROBUST to corruption, so this gap should be SMALL.
        # We negate it so a smaller gap yields a higher membership score.
        loss_gap = loss_corr - loss_orig
        icimia_gap_scores.append(-loss_gap)

    return np.array(y_true), np.array(baseline_loss_scores), np.array(icimia_gap_scores)

# ==========================================
# 4. Kaggle Submission Generation
# ==========================================
def generate_kaggle_submission():
    print(f"\n--- Phase 2: Generating Kaggle Predictions for {len(test_ds)} Test Samples ---")
    ids, predictions = [], []

    for i, sample in enumerate(tqdm(test_ds, desc="Kaggle Test Set")):
        sample_id = sample.get("id", i + 1)
        ids.append(sample_id)
        
        img = get_pil_image(sample["image"])
        img_corr = corrupt_image(img)
        caption = sample.get("text", sample.get("caption", ""))
        
        # Compute ICIMIA Gap
        loss_orig = compute_loss(img, caption)
        loss_corr = compute_loss(img_corr, caption)
        loss_gap = loss_corr - loss_orig
        
        # Using the ICIMIA Loss Gap as the submission score
        predictions.append(-loss_gap)

    df_submission = pd.DataFrame({"id": ids, "is_member": predictions})
    output_file = "/kaggle/working/submission.csv"
    df_submission.to_csv(output_file, index=False)
    print(f"\nSuccessfully saved Pair ICIMIA submission to {output_file}!")

# ==========================================
# 5. Main Execution
# ==========================================
if __name__ == "__main__":
    y_true, baseline_scores, icimia_scores = evaluate_pair_icimia()
    
    base_auc, base_tpr = calculate_metrics(y_true, baseline_scores)
    icimia_auc, icimia_tpr = calculate_metrics(y_true, icimia_scores)
    
    print("\n" + "="*50)
    print(f"{'Method':<25} | {'AUC':<8} | {'TPR@FPR=0.1'}")
    print("-" * 50)
    if np.isnan(base_auc):
        print("NOTE: Dev set missing ground truth labels for AUC.")
    else:
        print(f"{'Baseline Loss MIA':<25} | {base_auc:.4f}   | {base_tpr:.4f}")
        print(f"{'Pair-Level ICIMIA (Gap)':<25} | {icimia_auc:.4f}   | {icimia_tpr:.4f}")
    print("="*50 + "\n")
    
    generate_kaggle_submission()