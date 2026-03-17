import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, roc_curve
from tqdm.auto import tqdm

def get_token_losses(model, input_ids, attention_mask):
    """Computes token-level cross entropy losses for a batch."""
    with torch.no_grad():
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
    
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = input_ids[:, 1:].contiguous()
    shift_mask = attention_mask[:, 1:].contiguous()

    token_losses = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)), 
        shift_labels.view(-1), 
        reduction='none'
    ).view(shift_labels.size())

    token_losses = token_losses * shift_mask
    return token_losses.cpu().numpy(), shift_mask.cpu().numpy()

def extract_statistical_features(losses, mask, prefix=""):
    """Extracts comprehensive statistical features from token losses."""
    features = {}
    valid_losses = losses[mask == 1]
    
    if len(valid_losses) == 0:
        return {f"{prefix}_{k}": 0.0 for k in ["mean", "std", "max", "p90", "p10", "mink_20"]}
        
    features[f"{prefix}_mean"] = float(np.mean(valid_losses))
    features[f"{prefix}_std"] = float(np.std(valid_losses))
    features[f"{prefix}_max"] = float(np.max(valid_losses))
    features[f"{prefix}_p90"] = float(np.percentile(valid_losses, 90))
    features[f"{prefix}_p10"] = float(np.percentile(valid_losses, 10))
    
    # Min-K% Prob (Top 20% highest losses / least confident)
    k = max(1, int(len(valid_losses) * 0.20))
    worst_losses = np.sort(valid_losses)[-k:]
    features[f"{prefix}_mink_20"] = float(np.mean(worst_losses))
    
    return features

def extract_features_for_batch(texts, tokenizer, models, device, max_length=256):
    """Generates the tabular feature set for the XGBoost classifier."""
    encodings = tokenizer(texts, return_tensors="pt", truncation=True, padding=True, max_length=max_length)
    input_ids = encodings["input_ids"].to(device)
    attention_mask = encodings["attention_mask"].to(device)
    
    batch_features = [{} for _ in range(len(texts))]
    
    # Extract features for each model
    for model_name, model in models.items():
        losses, mask = get_token_losses(model, input_ids, attention_mask)
        
        for i in range(len(texts)):
            stats = extract_statistical_features(losses[i], mask[i], prefix=model_name)
            batch_features[i].update(stats)
            
            # Sequence length feature
            if "seq_len" not in batch_features[i]:
                batch_features[i]["seq_len"] = int(np.sum(mask[i]))
                
    # Calculate cross-model difference features for each sample
    for i in range(len(texts)):
        # Ref vs Target Memorization Delta
        batch_features[i]["delta_mean_ref_lm2"] = batch_features[i]["ref360_mean"] - batch_features[i]["lm2_mean"]
        batch_features[i]["delta_mink_ref_lm2"] = batch_features[i]["ref360_mink_20"] - batch_features[i]["lm2_mink_20"]
        
        # 360m vs 135m Capacity Delta
        batch_features[i]["delta_mean_lm1_lm2"] = batch_features[i]["lm1_mean"] - batch_features[i]["lm2_mean"]

    return batch_features

def build_dataset(hf_dataset, tokenizer, models, device, split="train", max_samples=None, batch_size=8):
    """Iterates through the HuggingFace dataset to build a Pandas DataFrame of features."""
    all_features = []
    labels = []
    ids = []
    
    num_samples = len(hf_dataset) if max_samples is None else min(len(hf_dataset), max_samples)
    
    for start_idx in tqdm(range(0, num_samples, batch_size), desc=f"Processing {split}"):
        batch = hf_dataset[start_idx : start_idx + batch_size]
        texts = batch["text"]
        
        features = extract_features_for_batch(texts, tokenizer, models, device)
        all_features.extend(features)
        
        if "is_member" in batch:
            labels.extend(batch["is_member"])
        if "id" in batch:
            ids.extend(batch["id"])
            
    df = pd.DataFrame(all_features)
    if labels:
        df["is_member"] = labels
    if ids:
        df["id"] = ids
        
    return df

def run_meta_classifier_mia():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load Models
    print("Loading models...")
    tokenizer = AutoTokenizer.from_pretrained("UBC-SLIME/colx_531_smollm2-360m")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    models = {
        "lm1": AutoModelForCausalLM.from_pretrained("UBC-SLIME/colx_531_smollm2-135m", torch_dtype=torch.float16).to(device).eval(),
        "lm2": AutoModelForCausalLM.from_pretrained("UBC-SLIME/colx_531_smollm2-360m", torch_dtype=torch.float16).to(device).eval(),
        "ref360": AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM2-360M", torch_dtype=torch.float16).to(device).eval()
    }

    # 2. Load Dataset
    print("Loading datasets...")
    dataset = load_dataset("UBC-SLIME/colx_531_group_project")
    
    # To save time, we train the meta-classifier on a 5000-sample subset of the train split. 
    train_df = build_dataset(dataset["train"], tokenizer, models, device, split="train", max_samples=5000)
    val_df = build_dataset(dataset["validation"], tokenizer, models, device, split="validation")
    
    feature_cols = [c for c in train_df.columns if c not in ["is_member", "id"]]
    X_train, y_train = train_df[feature_cols], train_df["is_member"]
    X_val, y_val = val_df[feature_cols], val_df["is_member"]

    # 3. Train XGBoost Meta-Classifier
    print("\nTraining XGBoost Meta-Classifier...")
    xgb = XGBClassifier(
        n_estimators=250,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric='auc',
        random_state=42
    )
    xgb.fit(X_train, y_train)

    # 4. Evaluate
    val_probs = xgb.predict_proba(X_val)[:, 1]
    
    roc_auc = roc_auc_score(y_val, val_probs)
    print(f"\nFinal Supervised ROC-AUC: {roc_auc:.4f}")
    
    fpr, tpr, thresholds = roc_curve(y_val, val_probs)
    tpr_at_fpr_01 = tpr[fpr <= 0.1][-1]
    print(f"TPR @ FPR=0.1: {tpr_at_fpr_01:.4f}")
    
    test_df = build_dataset(dataset["test"], tokenizer, models, device, split="test")
    test_probs = xgb.predict_proba(test_df[feature_cols])[:, 1]
    submission = pd.DataFrame({"id": test_df["id"], "score": test_probs})
    submission.to_csv("submission_xgboost_mia.csv", index=False)
    print("Saved Kaggle submission to submission_xgboost_mia.csv")

if __name__ == "__main__":
    # Ensure xgboost is installed: pip install xgboost
    run_meta_classifier_mia()