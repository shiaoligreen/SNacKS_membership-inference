# coded using gemini

import torch
import io
import zlib
import difflib
import gc
import os
import numpy as np
import pandas as pd
import xgboost as xgb
from PIL import Image
from torch.nn import CrossEntropyLoss
from datasets import load_dataset
from transformers import AutoProcessor, SmolVLMForConditionalGeneration, AutoModelForCausalLM, AutoTokenizer
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, roc_curve
from tqdm import tqdm

# ---------------------------------------------------------
# 1. Setup & Multi-Model Loading
# ---------------------------------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# --- Target VLM ---
model_id = "UBC-SLIME/colx_585_vlm"
dataset_id = "UBC-SLIME/colx585_group_project_data"

processor = AutoProcessor.from_pretrained(model_id)
processor.image_processor.do_image_splitting = False
processor.image_processor.size = {"longest_edge": 512}
processor.image_processor.max_image_size = {"longest_edge": 512}

print(f"Loading Target VLM: {model_id}...")
model = SmolVLMForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
    _attn_implementation="sdpa",
    trust_remote_code=True
).to(device)
model.eval()

# --- Reference LLM (For LiRA) ---
ref_model_id = "Qwen/Qwen2.5-0.5B"
print(f"Loading Reference Model: {ref_model_id}...")
ref_tokenizer = AutoTokenizer.from_pretrained(ref_model_id)
ref_model = AutoModelForCausalLM.from_pretrained(
    ref_model_id,
    torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
).to(device)
ref_model.eval()

# ---------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------
def parse_image(img_data):
    if isinstance(img_data, dict):
        if 'bytes' in img_data and img_data['bytes'] is not None:
            return Image.open(io.BytesIO(img_data['bytes'])).convert("RGB")
        elif 'path' in img_data and img_data['path'] is not None:
            return Image.open(img_data['path']).convert("RGB")
        else:
            return Image.new('RGB', (512, 512), (0, 0, 0)) 
    return img_data.convert("RGB")

def get_text_overlap(target_text, generated_text):
    matcher = difflib.SequenceMatcher(None, target_text.lower(), generated_text.lower())
    return matcher.ratio()

def get_reference_loss(text, ref_model, ref_tokenizer, loss_fct):
    inputs = ref_tokenizer(text, return_tensors="pt").to(device)
    labels = inputs["input_ids"].clone()
    
    with torch.no_grad():
        outputs = ref_model(**inputs)
        shift_logits = outputs.logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        token_losses = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        return token_losses.mean().item()

def get_loss_metrics(inputs, prompt_len, model, processor, loss_fct):
    inputs = {k: v.to(device) for k, v in inputs.items()}
    if device == "cuda":
        inputs = {k: v.to(torch.bfloat16) if v.dtype == torch.float32 else v for k, v in inputs.items()}
        
    labels = inputs["input_ids"].clone()
    labels[labels == processor.tokenizer.pad_token_id] = -100
    labels[:, :prompt_len] = -100
    
    with torch.no_grad():
        outputs = model(**inputs)
        shift_logits = outputs.logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        token_losses = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        valid_losses = token_losses[shift_labels.view(-1) != -100]
        
    if len(valid_losses) == 0:
        return 0.0, 0.0, 0.0, 0
        
    avg_loss = valid_losses.mean().item()
    k = max(1, int(0.2 * len(valid_losses)))
    max_loss_tokens = torch.topk(valid_losses, k, largest=True).values.mean().item()
    
    midpoint = len(valid_losses) // 2
    if midpoint > 0:
        trajectory_diff = valid_losses[:midpoint].mean().item() - valid_losses[midpoint:].mean().item()
    else:
        trajectory_diff = 0.0
    
    return avg_loss, max_loss_tokens, trajectory_diff, len(valid_losses)

def get_tpr_at_fpr(y_true, y_score, fpr_target=0.1):
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    valid_idx = np.where(fpr <= fpr_target)[0]
    if len(valid_idx) == 0:
        return 0.0
    idx = valid_idx[-1]
    return tpr[idx]

# ---------------------------------------------------------
# 2. Elite Feature Extraction (With Checkpointing)
# ---------------------------------------------------------
def extract_elite_features(dataset_split, desc="Extracting Features"):
    features = []
    labels = []
    ids = []
    
    user_text = "Describe this image."
    alt_user_text = "What is depicted in this photograph?"
    loss_fct = CrossEntropyLoss(reduction='none') 
    
    for i, example in enumerate(tqdm(dataset_split, desc=desc)):
        sample_id = example['id']
        assistant_text = example['text']
        label = example.get('is_member', None)
        
        real_image = parse_image(example['image'])
        blank_image = Image.new('RGB', (512, 512), (0, 0, 0))
        
        msg_standard = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": user_text}]}, {"role": "assistant", "content": [{"type": "text", "text": assistant_text}]}]
        msg_lower = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": user_text}]}, {"role": "assistant", "content": [{"type": "text", "text": assistant_text.lower()}]}]
        msg_alt = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": alt_user_text}]}, {"role": "assistant", "content": [{"type": "text", "text": assistant_text}]}]
        
        msg_prompt_standard = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": user_text}]}]
        msg_prompt_alt = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": alt_user_text}]}]
        
        text_standard = processor.apply_chat_template(msg_standard, tokenize=False)
        text_lower = processor.apply_chat_template(msg_lower, tokenize=False)
        text_alt = processor.apply_chat_template(msg_alt, tokenize=False)
        
        text_prompt_standard = processor.apply_chat_template(msg_prompt_standard, tokenize=False, add_generation_prompt=True)
        text_prompt_alt = processor.apply_chat_template(msg_prompt_alt, tokenize=False, add_generation_prompt=True)
        
        # 1. Target VLM Base Metrics
        inputs_real = processor(text=text_standard, images=[real_image], return_tensors="pt")
        prompt_inputs_real = processor(text=text_prompt_standard, images=[real_image], return_tensors="pt")
        prompt_len_real = prompt_inputs_real["input_ids"].shape[1]
        
        avg_real, max_loss_real, trajectory_diff, _ = get_loss_metrics(
            inputs_real, prompt_len_real, model, processor, loss_fct
        )
        
        # 2. Reference Model (LiRA)
        ref_loss = get_reference_loss(assistant_text, ref_model, ref_tokenizer, loss_fct)
        
        # 3. Free Generation (ROUGE / Overlap)
        prompt_inputs_gen = {k: v.to(device) for k, v in prompt_inputs_real.items()}
        with torch.no_grad():
            generated_ids = model.generate(
                **prompt_inputs_gen,
                max_new_tokens=len(assistant_text.split()) + 15,
                do_sample=False,
                use_cache=True,
                pad_token_id=processor.tokenizer.pad_token_id
            )
        gen_text_only = processor.batch_decode(
            generated_ids[:, prompt_inputs_gen["input_ids"].shape[1]:], 
            skip_special_tokens=True
        )[0].strip()
        
        generation_overlap = get_text_overlap(assistant_text, gen_text_only)
        
        # 4. Projector Gradient Norm Attack (White-Box)
        inputs_grad = {k: v.to(device) for k, v in inputs_real.items()}
        if device == "cuda":
            inputs_grad = {k: v.to(torch.bfloat16) if v.dtype == torch.float32 else v for k, v in inputs_grad.items()}
            
        labels_grad = inputs_grad["input_ids"].clone()
        labels_grad[labels_grad == processor.tokenizer.pad_token_id] = -100
        labels_grad[:, :prompt_len_real] = -100
        
        model.zero_grad()
        with torch.enable_grad():
            outputs = model(**inputs_grad, labels=labels_grad)
            loss = outputs.loss
            loss.backward()
            
            grad_norm = 0.0
            for name, param in model.named_parameters():
                # Target projector/connector layers for frozen-vision models
                if ("projector" in name or "visual_projection" in name or "connector" in name) and param.grad is not None:
                    grad_norm += param.grad.data.norm(2).item() ** 2
            grad_norm = grad_norm ** 0.5
            
        model.zero_grad() 
        
        # 5. Statistical Relatives (Lower, Blank, Alt)
        inputs_lower = processor(text=text_lower, images=[real_image], return_tensors="pt")
        avg_lower, _, _, _ = get_loss_metrics(inputs_lower, prompt_len_real, model, processor, loss_fct)
        
        inputs_blank = processor(text=text_standard, images=[blank_image], return_tensors="pt")
        prompt_inputs_blank = processor(text=text_prompt_standard, images=[blank_image], return_tensors="pt")
        prompt_len_blank = prompt_inputs_blank["input_ids"].shape[1]
        avg_blank, max_loss_blank, _, _ = get_loss_metrics(inputs_blank, prompt_len_blank, model, processor, loss_fct)

        inputs_alt = processor(text=text_alt, images=[real_image], return_tensors="pt")
        prompt_inputs_alt = processor(text=text_prompt_alt, images=[real_image], return_tensors="pt")
        prompt_len_alt = prompt_inputs_alt["input_ids"].shape[1]
        avg_alt_prompt, _, _, _ = get_loss_metrics(inputs_alt, prompt_len_alt, model, processor, loss_fct)
        
        # 6. Compile Features
        zlib_entropy = len(zlib.compress(assistant_text.encode('utf-8'))) / (len(assistant_text) + 1)
        
        features.append({
            'mink_prob_real': max_loss_real, 
            'trajectory_diff': trajectory_diff,           
            'lira_ratio': avg_real / (ref_loss + 1e-5),   
            'generation_overlap': generation_overlap,     
            'gradient_norm': grad_norm,                   
            'blank_diff': avg_blank - avg_real,
            'blank_mink_diff': max_loss_blank - max_loss_real,
            'prompt_brittle_diff': avg_alt_prompt - avg_real,
            'lower_diff': avg_lower - avg_real,
            'normalized_loss': avg_real / (zlib_entropy + 1e-5),
        })
        
        ids.append(sample_id)
        if label is not None:
            labels.append(label)
            
        # 7. Kaggle Survival Checkpointing (Saves every 500 rows)
        if i % 500 == 0 and i > 0:
            temp_df = pd.DataFrame(features)
            if labels:
                temp_df['is_member'] = labels
            temp_df['id'] = ids
            checkpoint_path = f"/kaggle/working/checkpoint_{desc.replace(' ', '_').lower()}.csv"
            temp_df.to_csv(checkpoint_path, index=False)
            
    if device == "cuda":
        torch.cuda.empty_cache()
        gc.collect()
            
    df = pd.DataFrame(features)
    if labels:
        df['is_member'] = labels
    df['id'] = ids
    
    return df

# ---------------------------------------------------------
# 3. Load Data & Compute Features
# ---------------------------------------------------------
ds_train = load_dataset(dataset_id, split='train')
ds_val = load_dataset(dataset_id, split='validation')
ds_test = load_dataset(dataset_id, split='test')

print("\n--- Extracting Train Features ---")
df_train = extract_elite_features(ds_train, "Train Data")

print("\n--- Extracting Holdout Validation Features ---")
df_val = extract_elite_features(ds_val, "Validation Data")

print("\n--- Extracting Test Features ---")
df_test = extract_elite_features(ds_test, "Test Data")

feature_cols = [
    'mink_prob_real', 'trajectory_diff', 'lira_ratio', 
    'generation_overlap', 'gradient_norm',
    'blank_diff', 'blank_mink_diff', 
    'prompt_brittle_diff', 'lower_diff', 'normalized_loss'
]

X_train_full = df_train[feature_cols].values
y_train_full = df_train['is_member'].values

X_val_holdout = df_val[feature_cols].values
y_val_holdout = df_val['is_member'].values

X_test = df_test[feature_cols].values

# ---------------------------------------------------------
# 4. K-Fold CV & Ensemble Evaluation (Handling Imbalance)
# ---------------------------------------------------------
print("\nInitiating 5-Fold Stratified CV on Train Set...")

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
val_holdout_preds = np.zeros(len(X_val_holdout))
test_preds = np.zeros(len(X_test))

# Calculate Pos Weight for Imbalance
num_positives = y_train_full.sum()
num_negatives = len(y_train_full) - num_positives
pos_weight = num_negatives / (num_positives + 1e-5)

for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_full, y_train_full)):
    print(f"\n--- Training Fold {fold + 1} ---")
    
    X_tr, y_tr = X_train_full[train_idx], y_train_full[train_idx]
    X_va, y_va = X_train_full[val_idx], y_train_full[val_idx]
    
    xgb_model = xgb.XGBClassifier(
        n_estimators=500,        
        learning_rate=0.01,      
        max_depth=2,             
        min_child_weight=15,     
        subsample=0.5,          
        colsample_bytree=0.6,   
        gamma=2.0,               
        reg_alpha=0.5,           
        reg_lambda=1.5,          
        scale_pos_weight=pos_weight,
        max_delta_step=1,             # Stabilizes predictions for class imbalance
        random_state=42 + fold,
        eval_metric='aucpr',          # Optimize for minority class recall/precision
        early_stopping_rounds=50 
    )
    
    xgb_model.fit(
        X_tr, y_tr,
        eval_set=[(X_tr, y_tr), (X_va, y_va)],
        verbose=100
    )
    
    # Predict on the holdout validation set
    val_holdout_preds += xgb_model.predict_proba(X_val_holdout)[:, 1] / skf.n_splits
    
    # Predict on the test set
    test_preds += xgb_model.predict_proba(X_test)[:, 1] / skf.n_splits

# ---------------------------------------------------------
# 5. Final Holdout Evaluation
# ---------------------------------------------------------
final_val_auc = roc_auc_score(y_val_holdout, val_holdout_preds)
final_val_tpr = get_tpr_at_fpr(y_val_holdout, val_holdout_preds, fpr_target=0.1)

print("\n==================================")
print("--- Final Validation Set Results ---")
print(f"AUC Score:      {final_val_auc:.4f}")
print(f"TPR @ FPR=0.1:  {final_val_tpr:.4f}")
print("==================================")

importances = xgb_model.feature_importances_
print("\n--- Feature Importances (Final Fold) ---")
for col, imp in zip(feature_cols, importances):
    print(f"{col}: {imp:.4f}")

# ---------------------------------------------------------
# 6. Kaggle Submission
# ---------------------------------------------------------
submission_df = pd.DataFrame({
    'id': df_test['id'],
    'is_member': test_preds
})

submission_path = "/kaggle/working/submission_final.csv"
submission_df.to_csv(submission_path, index=False)
print(f"\nSubmission saved to {submission_path} successfully!")