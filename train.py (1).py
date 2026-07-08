#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install scikit-image')


# In[2]:


get_ipython().system('pip install "numpy<2" "scikit-learn" "pandas"')


# In[11]:


import os
import glob
import numpy as np
import SimpleITK as sitk
from tqdm.auto import tqdm
import random

# ==========================================
# 1. DIRECTORY CONFIGURATION
# ==========================================
# Pointing directly to the brain folder shown in your screenshot
RAW_BRAIN_DIR = "./Task2/brain"

OUT_DIR = "./Processed_Task2_Clean"
OUT_CBCT = os.path.join(OUT_DIR, "cbct")
OUT_CT = os.path.join(OUT_DIR, "ct")
OUT_MASK = os.path.join(OUT_DIR, "mask")

# --- FAIL-FAST PATH CHECK ---
if not os.path.exists(RAW_BRAIN_DIR):
    raise FileNotFoundError(f"🛑 DIRECTORY NOT FOUND: Could not find '{RAW_BRAIN_DIR}'.")

os.makedirs(OUT_CBCT, exist_ok=True)
os.makedirs(OUT_CT, exist_ok=True)
os.makedirs(OUT_MASK, exist_ok=True)

print(f"✅ Input Directory Found: {RAW_BRAIN_DIR}")
print(f"✅ Initialized Clean Output Directory: {OUT_DIR}")

# ==========================================
# 2. PREPROCESSING LOOP (Nested Structure)
# ==========================================
# Grab all patient subdirectories (e.g., 2BA001, 2BA002)
patient_dirs = sorted([d for d in glob.glob(os.path.join(RAW_BRAIN_DIR, "*")) if os.path.isdir(d)])

if len(patient_dirs) == 0:
    raise FileNotFoundError(f"🛑 NO PATIENTS FOUND inside '{RAW_BRAIN_DIR}'.")

total_slices_saved = 0

for patient_dir in tqdm(patient_dirs, desc="Slicing Brain Volumes"):
    patient_id = os.path.basename(patient_dir) # e.g., "2BA001"
    
    cbct_path = os.path.join(patient_dir, "cbct.nii.gz")
    ct_path = os.path.join(patient_dir, "ct.nii.gz")
    mask_path = os.path.join(patient_dir, "mask.nii.gz")
    
    # Ensure all 3 files exist for this patient
    if not (os.path.exists(cbct_path) and os.path.exists(ct_path) and os.path.exists(mask_path)):
        print(f"Warning: Missing files for patient {patient_id}, skipping...")
        continue
        
    # Load Volumes
    cbct_itk = sitk.ReadImage(cbct_path)
    ct_itk = sitk.ReadImage(ct_path)
    mask_itk = sitk.ReadImage(mask_path)
    
    cbct_vol = sitk.GetArrayFromImage(cbct_itk).astype(np.float32)
    ct_vol = sitk.GetArrayFromImage(ct_itk).astype(np.float32)
    
    # Load the official provided mask
    mask_vol = sitk.GetArrayFromImage(mask_itk).astype(np.uint8)
    
    # ---------------------------------------------------------
    # VOLUME-LEVEL RAW DATA VERIFICATION
    # ---------------------------------------------------------
    mean_diff = np.mean(np.abs(cbct_vol - ct_vol))
    if mean_diff < 1e-6:
        raise ValueError(f"🛑 CRITICAL FAILURE: {patient_id} entire volume appears identical. Mean diff: {mean_diff}")
    
    # Normalization [-1000, 3000] -> [0, 1]
    cbct_norm = np.clip((cbct_vol + 1000.0) / 4000.0, 0.0, 1.0)
    ct_norm = np.clip((ct_vol + 1000.0) / 4000.0, 0.0, 1.0)

    # Slice and Save with Strict Z-Ordering
    num_slices = cbct_norm.shape[0]
    for z in range(num_slices):
        cbct_slice = cbct_norm[z, :, :]
        ct_slice = ct_norm[z, :, :]
        mask_slice = mask_vol[z, :, :]

        # Save with zero-padding (_000.npy)
        np.save(os.path.join(OUT_CBCT, f"{patient_id}_{z:03d}.npy"), cbct_slice)
        np.save(os.path.join(OUT_CT, f"{patient_id}_{z:03d}.npy"), ct_slice)
        np.save(os.path.join(OUT_MASK, f"{patient_id}_{z:03d}.npy"), mask_slice)
        
        total_slices_saved += 1

print(f"✅ Preprocessing Complete! Total physical slices generated: {total_slices_saved}")

# ==========================================
# 3. POST-PREPROCESSING INTEGRITY AUDIT
# ==========================================
print("\n==================================================")
print(" POST-PREPROCESSING INTEGRITY AUDIT")
print("==================================================")

all_cbct_slices = glob.glob(f"{OUT_DIR}/cbct/*.npy")

if len(all_cbct_slices) == 0:
    raise Exception("🛑 FATAL: Slicing finished, but no .npy files were found in the output directory!")

sample_paths = random.sample(all_cbct_slices, min(10, len(all_cbct_slices)))
audit_diffs = []

for cbct_p in sample_paths:
    ct_p = cbct_p.replace('/cbct/', '/ct/')
    
    cbct_arr = np.load(cbct_p)
    ct_arr = np.load(ct_p)
    
    diff = np.mean(np.abs(cbct_arr - ct_arr))
    audit_diffs.append(diff)
    
    slice_name = os.path.basename(cbct_p)
    print(f"Slice: {slice_name} | Mean Abs Diff: {diff:.6f}")

if sum(audit_diffs) == 0.0:
    raise Exception("🛑 FATAL: All random samples returned 0.0 difference. The dataset is STILL corrupted.")
else:
    print("\n🎉 AUDIT PASSED: CBCT and CT arrays contain non-zero differences. You are cleared to train!")


# In[12]:


print(len(set(train_ids) & set(val_ids)))
print(len(set(train_ids) & set(test_ids)))
print(len(set(val_ids) & set(test_ids)))


# In[2]:


import os
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F  # <--- Added for resizing
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm
from sklearn.model_selection import train_test_split

# ==========================================================
# 1. DATASET CODE (UPDATED FOR BRAIN ONLY + SPLIT)
# ==========================================================
# --- A. Split the Patient IDs (BRAIN ONLY) ---

data_dir = "./Processed_Task2_Clean"

# Define the prefix for Brain patients
BRAIN_PREFIX = "2B" 

# Grab all files and extract unique Patient IDs
all_cbct_files = glob.glob(os.path.join(data_dir, "cbct", "*.npy"))
all_unique_ids = list(set([os.path.basename(f).split('_')[0] for f in all_cbct_files]))

# FILTER FOR BRAIN ONLY
brain_patient_ids = [pid for pid in all_unique_ids if pid.startswith(BRAIN_PREFIX)]

print(f"Total Unique Brain Patients Found: {len(brain_patient_ids)}")

# First split: 80% Train, 20% Temporary (Val + Test)
train_ids, temp_ids = train_test_split(brain_patient_ids, test_size=0.20, random_state=42)

# Second split: Divide the 20% Temporary exactly in half (10% Val, 10% Test)
val_ids, test_ids = train_test_split(temp_ids, test_size=0.50, random_state=42)

print(f"Assigned to Training: {len(train_ids)} patients")
print(f"Assigned to Validation: {len(val_ids)} patients")
print(f"Assigned to Testing: {len(test_ids)} patients")

# --- B. The Upgraded Dataset Class (WITH DYNAMIC RESIZING FIX) ---
class UFormerDataset(Dataset):
    def __init__(self, data_dir, valid_ids=None, target_size=(256, 256)):
        super().__init__()
        self.data_dir = data_dir
        self.target_size = target_size
        
        # Grab all files
        all_files = sorted(glob.glob(os.path.join(data_dir, "cbct", "*.npy")))
        
        # FILTERING LOGIC: Only keep slices belonging to the valid patient IDs
        if valid_ids is not None:
            self.cbct_files = [
                f for f in all_files 
                if os.path.basename(f).split('_')[0] in valid_ids
            ]
        else:
            self.cbct_files = all_files
            
    def __len__(self):
        return len(self.cbct_files)

    def __getitem__(self, idx):
        cbct_path = self.cbct_files[idx]
        filename = os.path.basename(cbct_path)
        
        ct_path = os.path.join(self.data_dir, "ct", filename)
        mask_path = os.path.join(self.data_dir, "mask", filename)
        
        cbct = np.load(cbct_path)
        ct = np.load(ct_path)
        mask = np.load(mask_path)
        
        # Convert to Tensors: Shape becomes [1, H, W]
        cbct_tensor = torch.tensor(cbct, dtype=torch.float32).unsqueeze(0)
        ct_tensor = torch.tensor(ct, dtype=torch.float32).unsqueeze(0)
        mask_tensor = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)
        
        # --- DYNAMIC RESIZING TO 256x256 TO FIX DATALOADER CRASH ---
        # F.interpolate requires a batch dimension [B, C, H, W], so we unsqueeze(0) then squeeze(0)
        cbct_tensor = F.interpolate(cbct_tensor.unsqueeze(0), size=self.target_size, mode='bilinear', align_corners=False).squeeze(0)
        ct_tensor = F.interpolate(ct_tensor.unsqueeze(0), size=self.target_size, mode='bilinear', align_corners=False).squeeze(0)
        
        # Masks must use 'nearest' interpolation so they stay strictly 0.0 or 1.0 (no fractional blending)
        mask_tensor = F.interpolate(mask_tensor.unsqueeze(0), size=self.target_size, mode='nearest').squeeze(0)
        
        return cbct_tensor, ct_tensor, mask_tensor

# ==========================================
# 2. MODEL CODE
# ==========================================
class DepthWiseConv(nn.Module):
    def __init__(self, dim, kernel_size=3, padding=1):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=kernel_size, padding=padding, groups=dim)
    def forward(self, x):
        return self.dwconv(x)

class LeFF(nn.Module):
    def __init__(self, dim, seq_len=256, mlp_ratio=4):
        super().__init__()
        hidden_dim = int(dim * mlp_ratio)
        self.linear1 = nn.Linear(dim, hidden_dim)
        self.dwconv = DepthWiseConv(hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, dim)
        self.gelu = nn.GELU()
    def forward(self, x, H, W):
        B, N, C = x.shape
        x = self.linear1(x)
        x = x.transpose(1, 2).view(B, -1, H, W)
        x = self.dwconv(x)
        x = self.gelu(x)
        x = x.flatten(2).transpose(1, 2)
        x = self.linear2(x)
        return x

class WindowAttention(nn.Module):
    def __init__(self, dim, num_heads, win_size=8):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.win_size = win_size
        self.scale = (dim // num_heads) ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)
    def forward(self, x, H, W):
        B, N, C = x.shape
        head_dim = C // self.num_heads
        x = x.view(B, H, W, C)
        win_h, win_w = self.win_size, self.win_size
        num_win_h, num_win_w = H // win_h, W // win_w
        x = x.view(B, num_win_h, win_h, num_win_w, win_w, C)
        windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, win_h * win_w, C)
        total_wins = windows.shape[0]
        qkv = self.qkv(windows).view(total_wins, win_h * win_w, 3, self.num_heads, head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = (attn @ v).permute(0, 2, 1, 3).contiguous().view(total_wins, win_h * win_w, C)
        out = self.proj(out)
        out = out.view(B, num_win_h, num_win_w, win_h, win_w, C)
        out = out.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H * W, C)
        return out

class LeWinBlock(nn.Module):
    def __init__(self, dim, num_heads, win_size=8, mlp_ratio=4):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, num_heads, win_size)
        self.norm2 = nn.LayerNorm(dim)
        self.leff = LeFF(dim, mlp_ratio=mlp_ratio)
    def forward(self, x, H, W):
        x = x + self.attn(self.norm1(x), H, W)
        x = x + self.leff(self.norm2(x), H, W)
        return x

class UFormer(nn.Module):
    def __init__(self, in_chans=1, out_chans=1, embed_dim=32, win_size=8):
        super().__init__()
        self.win_size = win_size
        self.input_proj = nn.Conv2d(in_chans, embed_dim, kernel_size=3, padding=1)
        self.enc_block1 = LeWinBlock(dim=embed_dim, num_heads=1, win_size=win_size)
        self.down1 = nn.Conv2d(embed_dim, embed_dim * 2, kernel_size=4, stride=2, padding=1)
        self.enc_block2 = LeWinBlock(dim=embed_dim * 2, num_heads=2, win_size=win_size)
        self.down2 = nn.Conv2d(embed_dim * 2, embed_dim * 4, kernel_size=4, stride=2, padding=1)
        self.bottleneck = LeWinBlock(dim=embed_dim * 4, num_heads=4, win_size=win_size)
        self.up2 = nn.ConvTranspose2d(embed_dim * 4, embed_dim * 2, kernel_size=2, stride=2)
        self.dec_block2 = LeWinBlock(dim=embed_dim * 2, num_heads=2, win_size=win_size)
        self.up1 = nn.ConvTranspose2d(embed_dim * 2, embed_dim, kernel_size=2, stride=2)
        self.dec_block1 = LeWinBlock(dim=embed_dim, num_heads=1, win_size=win_size)
        self.output_proj = nn.Conv2d(embed_dim, out_chans, kernel_size=3, padding=1)
        
        # --- ADDED: Sigmoid Activation ---
        self.sigmoid = nn.Sigmoid() 

    def forward(self, x):
        x_proj = self.input_proj(x)
        B, C, H, W = x_proj.shape
        x_flat1 = x_proj.flatten(2).transpose(1, 2)
        x_enc1 = self.enc_block1(x_flat1, H, W).transpose(1, 2).view(B, C, H, W)
        x_down1 = self.down1(x_enc1)
        B2, C2, H2, W2 = x_down1.shape
        x_flat2 = x_down1.flatten(2).transpose(1, 2)
        x_enc2 = self.enc_block2(x_flat2, H2, W2).transpose(1, 2).view(B2, C2, H2, W2)
        x_down2 = self.down2(x_enc2)
        B3, C3, H3, W3 = x_down2.shape
        x_bot_flat = x_down2.flatten(2).transpose(1, 2)
        x_bot = self.bottleneck(x_bot_flat, H3, W3).transpose(1, 2).view(B3, C3, H3, W3)
        x_up2 = self.up2(x_bot)
        x_up2 = x_up2 + x_enc2
        x_up2_flat = x_up2.flatten(2).transpose(1, 2)
        x_dec2 = self.dec_block2(x_up2_flat, H2, W2).transpose(1, 2).view(B2, C2, H2, W2)
        x_up1 = self.up1(x_dec2)
        x_up1 = x_up1 + x_enc1
        x_up1_flat = x_up1.flatten(2).transpose(1, 2)
        x_dec1 = self.dec_block1(x_up1_flat, H, W).transpose(1, 2).view(B, C, H, W)
        
        out = self.output_proj(x_dec1)
        
        # --- ADDED: Apply Sigmoid to strictly bind output between 0.0 and 1.0 ---
        out = self.sigmoid(out) 
        
        return out


# In[3]:


# ==========================================
# STRICT PATIENT-LEVEL LEAKAGE AUDIT
# ==========================================
print("==================================================")
print(" 🕵️ DATA LEAKAGE AUDIT")
print("==================================================")

# Convert lists to Python Sets for mathematical intersection
train_set = set(train_ids)
val_set = set(val_ids)
test_set = set(test_ids)

# Calculate intersections
train_val_overlap = train_set.intersection(val_set)
train_test_overlap = train_set.intersection(test_set)
val_test_overlap = val_set.intersection(test_set)

# Print strict numerical results
print(f"Total Training Patients   : {len(train_set)}")
print(f"Total Validation Patients : {len(val_set)}")
print(f"Total Testing Patients    : {len(test_set)}")
print("--------------------------------------------------")
print(f"Overlap (Train ∩ Val)     : {len(train_val_overlap)} patients")
print(f"Overlap (Train ∩ Test)    : {len(train_test_overlap)} patients")
print(f"Overlap (Val ∩ Test)      : {len(val_test_overlap)} patients")
print("==================================================")

# Fail-Fast Assertions
if len(train_val_overlap) > 0:
    print(f"🛑 FATAL LEAKAGE: These patients are in both Train and Val: {train_val_overlap}")
if len(train_test_overlap) > 0:
    print(f"🛑 FATAL LEAKAGE: These patients are in both Train and Test: {train_test_overlap}")
if len(val_test_overlap) > 0:
    print(f"🛑 FATAL LEAKAGE: These patients are in both Val and Test: {val_test_overlap}")

assert len(train_val_overlap) == 0, "Train/Val Leakage Detected!"
assert len(train_test_overlap) == 0, "Train/Test Leakage Detected!"
assert len(val_test_overlap) == 0, "Val/Test Leakage Detected!"

print("\n🎉 AUDIT PASSED: Absolute Zero Data Leakage. Sets are completely disjoint. Training is mathematically safe!")


# In[4]:


import os
import csv
import torch
import numpy as np
from skimage.metrics import structural_similarity as compare_ssim

# ====================================================
# 1. LOSS FUNCTION DEFINITION
# ====================================================
class MaskedCompositeLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = torch.nn.L1Loss(reduction='none')

    def forward(self, pred, target, mask):
        loss = self.l1(pred, target)
        return (loss * mask).sum() / (mask.sum() + 1e-8)

# ====================================================
# 2. CONFIGURATION & DIRECTORIES
# ====================================================
DATA_DIR = "./Processed_Task2_Clean"
CHECKPOINT_DIR = "./checkpoints"
LOG_FILE = "training_log.csv"

NUM_EPOCHS = 60
SAVE_EVERY = 5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ====================================================
# 3. DATALOADERS (HIGH-SPEED MULTI-THREADING APPLIED)
# ====================================================
train_dataset = UFormerDataset(data_dir=DATA_DIR, valid_ids=train_ids)
val_dataset = UFormerDataset(data_dir=DATA_DIR, valid_ids=val_ids)

# CPU threads now feed the A100 GPU constantly
train_loader = torch.utils.data.DataLoader(
    train_dataset, 
    batch_size=8, 
    shuffle=True, 
    num_workers=4, 
    pin_memory=True,
    prefetch_factor=2 
)

val_loader = torch.utils.data.DataLoader(
    val_dataset, 
    batch_size=1, 
    shuffle=False, 
    num_workers=4, 
    pin_memory=True
)

# ====================================================
# 4. INITIALIZATION & EXPLICIT RESUME LOGIC
# ====================================================
model = UFormer().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
criterion = MaskedCompositeLoss()
scaler = torch.cuda.amp.GradScaler()

best_val_mae = float('inf')
start_epoch = 0

RESUME_PATH = os.path.join(CHECKPOINT_DIR, "epoch_15.pth")

if os.path.exists(RESUME_PATH):
    print(f"🔄 Checkpoint found! Extracting states from: {RESUME_PATH}")
    checkpoint = torch.load(RESUME_PATH, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch']
    
    best_val_mae = checkpoint.get('val_mae', 61.32) 
    
    print(f"✅ Successfully loaded weights. Resuming from Epoch {start_epoch + 1}.")
    print(f"✅ Current Best Val MAE to beat: {best_val_mae:.2f} HU")
else:
    raise FileNotFoundError(f"🛑 Could not find '{RESUME_PATH}'. Make sure 'epoch_15.pth' actually exists in your checkpoints folder!")

# ====================================================
# 5. MAIN TRAINING LOOP
# ====================================================
print("\n==================================================")
print(f" 🚀 RESUMING TRAINING (Epoch {start_epoch + 1} to {NUM_EPOCHS})")
print("==================================================")

for epoch in range(start_epoch, NUM_EPOCHS):
    print(f"\n▶ EPOCH {epoch+1}/{NUM_EPOCHS} STARTED...")
    
    # --- TRAINING PHASE ---
    model.train()
    train_epoch_loss = 0.0
    
    for i, (cbct_batch, ct_batch, mask_batch) in enumerate(train_loader):
        cbct_batch = cbct_batch.to(device).float()
        ct_batch = ct_batch.to(device).float()
        mask_batch = mask_batch.to(device).float()
        
        optimizer.zero_grad()
        
        with torch.cuda.amp.autocast():
            outputs = model(cbct_batch)
            loss = criterion(outputs, ct_batch, mask_batch)
            
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        train_epoch_loss += loss.item()
        
        if (i + 1) % 500 == 0 or (i + 1) == len(train_loader):
            print(f"   [Train] Batch {i+1}/{len(train_loader)} | Current Batch Loss: {loss.item():.4f}")
            
    avg_train_loss = train_epoch_loss / len(train_loader)
    
    # --- VALIDATION PHASE ---
    print(f"   [Val] Evaluating validation set...")
    model.eval()
    val_epoch_loss = 0.0
    val_epoch_mae = 0.0
    val_epoch_psnr = 0.0
    val_epoch_ssim = 0.0
    
    with torch.no_grad():
        for i, (cbct_batch, ct_batch, mask_batch) in enumerate(val_loader):
            cbct_batch = cbct_batch.to(device).float()
            ct_batch = ct_batch.to(device).float()
            mask_batch = mask_batch.to(device).float()
            
            with torch.cuda.amp.autocast():
                outputs = model(cbct_batch)
                val_loss = criterion(outputs, ct_batch, mask_batch)
                
            val_epoch_loss += val_loss.item()
            
            outputs_np = outputs.cpu().squeeze(1).numpy()
            targets_np = ct_batch.cpu().squeeze(1).numpy()
            
            if mask_batch.ndim == 4:
                masks_np = mask_batch.cpu().squeeze(1).numpy()
            else:
                masks_np = mask_batch.cpu().numpy()
            
            pred_hu = (outputs_np * 4000.0) - 1000.0
            target_hu = (targets_np * 4000.0) - 1000.0
            
            batch_size = outputs_np.shape[0]
            batch_mae = 0
            batch_psnr = 0
            batch_ssim = 0
            
            for b in range(batch_size):
                p_hu = pred_hu[b]
                t_hu = target_hu[b]
                m = masks_np[b].astype(bool)
                
                if np.sum(m) == 0:
                    continue
                    
                abs_diff = np.abs(p_hu - t_hu)
                mae = (abs_diff * m).sum() / m.sum()
                batch_mae += mae
                
                mse = ((abs_diff ** 2) * m).sum() / m.sum()
                psnr = 10 * np.log10((4000.0 ** 2) / mse) if mse > 0 else 100.0
                batch_psnr += psnr
                
                p_hu_masked = p_hu * m
                t_hu_masked = t_hu * m
                ssim_val = compare_ssim(t_hu_masked, p_hu_masked, data_range=4000.0)
                batch_ssim += ssim_val
                
            val_epoch_mae += (batch_mae / batch_size)
            val_epoch_psnr += (batch_psnr / batch_size)
            val_epoch_ssim += (batch_ssim / batch_size)
            
    avg_val_loss = val_epoch_loss / len(val_loader)
    avg_val_mae = val_epoch_mae / len(val_loader)
    avg_val_psnr = val_epoch_psnr / len(val_loader)
    avg_val_ssim = val_epoch_ssim / len(val_loader)
    
    print(f"📊 Epoch {epoch+1} Results | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val MAE: {avg_val_mae:.2f} HU | Val PSNR: {avg_val_psnr:.2f} dB | Val SSIM: {avg_val_ssim:.4f}")

    # ====================================================
    # 6. LOGGING & CHECKPOINT SAVING
    # ====================================================
    with open(LOG_FILE, mode='a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([epoch+1, avg_train_loss, avg_val_loss, avg_val_mae, avg_val_psnr, avg_val_ssim])

    if avg_val_mae < best_val_mae:
        best_val_mae = avg_val_mae
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_mae': best_val_mae,
        }, os.path.join(CHECKPOINT_DIR, "best_model.pth"))
        print(f"🌟 New Best Model Saved! (MAE: {best_val_mae:.2f} HU)")
        
    if (epoch + 1) % SAVE_EVERY == 0:
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, os.path.join(CHECKPOINT_DIR, f"epoch_{epoch+1}.pth"))
        print(f"💾 Epoch {epoch+1} checkpoint saved.")

print("\n✅ Training Complete. All metrics logged to", LOG_FILE)


# In[5]:


import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as compare_ssim
from tqdm.auto import tqdm

# ====================================================
# 1. SETUP & DIRECTORIES
# ====================================================
DATA_DIR = "./Processed_Task2_Clean"
BEST_MODEL_PATH = "./checkpoints/best_model.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ====================================================
# 2. LOAD THE BEST MODEL
# ====================================================
print("==================================================")
print(" 🔬 INITIATING FINAL TEST EVALUATION")
print("==================================================")

model = UFormer().to(device)

if not os.path.exists(BEST_MODEL_PATH):
    raise FileNotFoundError(f"🛑 Could not find {BEST_MODEL_PATH}.")

checkpoint = torch.load(BEST_MODEL_PATH, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

best_epoch = checkpoint.get('epoch', 'Unknown')
val_mae = checkpoint.get('val_mae', 0.0)
print(f"🌟 Successfully loaded weights from Epoch {best_epoch}.")
print(f"   (Achieved Validation MAE: {val_mae:.2f} HU)")

# ====================================================
# 3. TEST DATALOADER
# ====================================================
# Using the test_ids generated from your initial split
test_dataset = UFormerDataset(data_dir=DATA_DIR, valid_ids=test_ids)
test_loader = torch.utils.data.DataLoader(
    test_dataset, 
    batch_size=1, 
    shuffle=False, 
    num_workers=4, 
    pin_memory=True
)

print(f"📊 Evaluating on {len(test_dataset)} unseen test slices...")

# ====================================================
# 4. EVALUATION LOOP
# ====================================================
test_mae = 0.0
test_psnr = 0.0
test_ssim = 0.0
valid_slices = 0
sample_saved = False

with torch.no_grad():
    for i, (cbct_tensor, ct_tensor, mask_tensor) in enumerate(tqdm(test_loader, desc="Testing", mininterval=2.0)):
        cbct_batch = cbct_tensor.to(device).float()
        ct_batch = ct_tensor.to(device).float()
        mask_batch = mask_tensor.to(device).float()

        with torch.cuda.amp.autocast():
            outputs = model(cbct_batch)

        outputs_np = outputs.cpu().squeeze().numpy()
        targets_np = ct_batch.cpu().squeeze().numpy()
        inputs_np = cbct_batch.cpu().squeeze().numpy()
        masks_np = mask_batch.cpu().squeeze().numpy()

        # Convert back to true Hounsfield Units (-1000 to +3000)
        pred_hu = (outputs_np * 4000.0) - 1000.0
        target_hu = (targets_np * 4000.0) - 1000.0
        input_hu = (inputs_np * 4000.0) - 1000.0

        m = masks_np.astype(bool)

        # Skip completely empty slices
        if np.sum(m) == 0:
            continue

        valid_slices += 1

        # --- Metrics Math ---
        abs_diff = np.abs(pred_hu - target_hu)
        mae = (abs_diff * m).sum() / m.sum()
        test_mae += mae

        mse = ((abs_diff ** 2) * m).sum() / m.sum()
        psnr = 10 * np.log10((4000.0 ** 2) / mse) if mse > 0 else 100.0
        test_psnr += psnr

        p_hu_masked = pred_hu * m
        t_hu_masked = target_hu * m
        ssim_val = compare_ssim(t_hu_masked, p_hu_masked, data_range=4000.0)
        test_ssim += ssim_val

        # --- Save Visual Stitching Sample ---
        # Grabbing a slice that actually has anatomy in it (mask sum > 5000 pixels)
        if not sample_saved and m.sum() > 5000:
            plt.figure(figsize=(18, 6))
            
            plt.subplot(1, 3, 1)
            plt.imshow(input_hu, cmap='gray', vmin=-1000, vmax=1000)
            plt.title("Input CBCT (Scattered)")
            plt.axis('off')

            plt.subplot(1, 3, 2)
            plt.imshow(pred_hu, cmap='gray', vmin=-1000, vmax=1000)
            plt.title(f"Predicted sCT\n(MAE: {mae:.2f} HU)")
            plt.axis('off')

            plt.subplot(1, 3, 3)
            plt.imshow(target_hu, cmap='gray', vmin=-1000, vmax=1000)
            plt.title("Ground Truth CT")
            plt.axis('off')

            plt.tight_layout()
            plt.savefig("final_test_result.png", dpi=300, bbox_inches='tight')
            plt.close()
            sample_saved = True

# ====================================================
# 5. FINAL RESULTS
# ====================================================
final_mae = test_mae / valid_slices
final_psnr = test_psnr / valid_slices
final_ssim = test_ssim / valid_slices

print("\n==================================================")
print(" 🏆 FINAL TEST SET RESULTS")
print("==================================================")
print(f"Total Evaluated Slices : {valid_slices}")
print(f"Final Test MAE         : {final_mae:.2f} HU")
print(f"Final Test PSNR        : {final_psnr:.2f} dB")
print(f"Final Test SSIM        : {final_ssim:.4f}")
print("==================================================")
print("📸 A high-res sample comparison image has been saved as 'final_test_result.png'!")


# In[7]:


import os
import glob
import re
import torch
import numpy as np
import pandas as pd
import SimpleITK as sitk
import torch.nn.functional as F
from tqdm.auto import tqdm
from skimage.metrics import structural_similarity as compare_ssim

# ====================================================
# 1. CONFIGURATION & DIRECTORIES
# ====================================================
DATA_DIR = "./Processed_Task2_Clean"
BEST_MODEL_PATH = "./checkpoints/best_model.pth"
OUTPUT_DIR = "./3D_Reconstructions"
METRICS_CSV = os.path.join(OUTPUT_DIR, "final_patient_metrics.csv")
SUMMARY_TXT = os.path.join(OUTPUT_DIR, "summary_metrics.txt")

# UPDATE THIS: Point this to the folder containing your original Raw CT NIfTI files 
# so the script can extract the exact physical spacing, origin, and direction metadata.
RAW_CT_NIFTI_DIR = "./Raw_Task2/ct" 

os.makedirs(OUTPUT_DIR, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ====================================================
# 2. HELPER FUNCTIONS: SANITY CHECKS & METRICS
# ====================================================
def extract_z_index(filename):
    """Extracts the slice number from filenames like '2B01_slice_045.npy' or '2B01_045.npy'"""
    match = re.search(r'_(\d+)\.npy$', filename)
    if match:
        return int(match.group(1))
    else:
        raise ValueError(f"Could not extract Z-index from filename: {filename}")

def compute_3d_metrics(pred_vol, gt_vol, mask_vol):
    """Computes masked MAE, PSNR, and SSIM for full 3D volumes"""
    m = mask_vol.astype(bool)
    if np.sum(m) == 0:
        return 0.0, 0.0, 0.0
        
    abs_diff = np.abs(pred_vol - gt_vol)
    mae = (abs_diff * m).sum() / m.sum()
    
    mse = ((abs_diff ** 2) * m).sum() / m.sum()
    psnr = 10 * np.log10((4000.0 ** 2) / mse) if mse > 0 else 100.0
    
    pred_masked = pred_vol * m
    gt_masked = gt_vol * m
    
    # 3D SSIM computation
    ssim = compare_ssim(gt_masked, pred_masked, data_range=4000.0, channel_axis=None)
    
    return mae, psnr, ssim

# ====================================================
# 3. INITIALIZATION
# ====================================================
print("==================================================")
print(" INITIATING 3D RECONSTRUCTION PIPELINE")
print("==================================================")

# Ensure test_ids exist in memory (from your previous split)
if 'test_ids' not in locals():
    raise NameError("'test_ids' is not defined. Please run your dataset split cell first.")

# Load Model
model = UFormer().to(device)
if not os.path.exists(BEST_MODEL_PATH):
    raise FileNotFoundError(f"Could not find {BEST_MODEL_PATH}")

checkpoint = torch.load(BEST_MODEL_PATH, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
print(f"Model weights loaded from Epoch {checkpoint.get('epoch', 'Unknown')}.")

patient_metrics = []

# ====================================================
# 4. PATIENT-LEVEL PROCESSING LOOP
# ====================================================
for pid in tqdm(test_ids, desc="Processing Patients"):
    print(f"\nReconstructing Patient: {pid}")
    
    # --- A. Gather and Sort Slices ---
    patient_cbct_files = glob.glob(os.path.join(DATA_DIR, "cbct", f"{pid}_*.npy"))
    
    if not patient_cbct_files:
        print(f"   WARNING: No slices found for {pid}. Skipping.")
        continue
        
    # Sort files physically by their Z-index
    patient_cbct_files.sort(key=lambda x: extract_z_index(x))
    
    # --- SANITY CHECK 1: Missing Slices / Z-Axis Ordering ---
    z_indices = [extract_z_index(f) for f in patient_cbct_files]
    expected_indices = list(range(z_indices[0], z_indices[-1] + 1))
    if z_indices != expected_indices:
        print(f"   SANITY FAIL: Missing slices detected for patient {pid}!")
        print(f"   Found indices: {z_indices}")
        continue
    
    print(f"   [Sanity Pass] Found {len(patient_cbct_files)} slices ordered correctly.")

    pred_volume_list = []
    gt_volume_list = []
    mask_volume_list = []
    
    # --- B. Inference & Resizing Loop ---
    with torch.no_grad():
        for cbct_path in patient_cbct_files:
            filename = os.path.basename(cbct_path)
            ct_path = os.path.join(DATA_DIR, "ct", filename)
            mask_path = os.path.join(DATA_DIR, "mask", filename)
            
            cbct_np = np.load(cbct_path)
            ct_np = np.load(ct_path)
            mask_np = np.load(mask_path)
            
            original_h, original_w = cbct_np.shape
            
            # Prepare tensors and interpolate to 256x256 for the model
            cbct_tensor = torch.tensor(cbct_np, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
            cbct_256 = F.interpolate(cbct_tensor, size=(256, 256), mode='bilinear', align_corners=False)
            
            # Model Inference
            with torch.cuda.amp.autocast():
                pred_256 = model(cbct_256)
                
            # --- SANITY CHECK 2: Shape Mismatches ---
            # Interpolate prediction back to the EXACT original patient anatomy size
            pred_restored = F.interpolate(pred_256, size=(original_h, original_w), mode='bilinear', align_corners=False)
            pred_np = pred_restored.squeeze().cpu().numpy()
            
            if pred_np.shape != ct_np.shape:
                raise ValueError(f"SANITY FAIL: Shape mismatch after restoration! Pred: {pred_np.shape}, GT: {ct_np.shape}")
            
            # Convert to HU
            pred_hu = (pred_np * 4000.0) - 1000.0
            gt_hu = (ct_np * 4000.0) - 1000.0
            
            pred_volume_list.append(pred_hu)
            gt_volume_list.append(gt_hu)
            mask_volume_list.append(mask_np)

    # --- C. 3D Volume Construction ---
    # Stack along the Z-axis (Typically SimpleITK expects [Z, Y, X])
    pred_3d = np.stack(pred_volume_list, axis=0)
    gt_3d = np.stack(gt_volume_list, axis=0)
    mask_3d = np.stack(mask_volume_list, axis=0)
    
    # --- SANITY CHECK 3: Volume Integrity ---
    assert pred_3d.shape == gt_3d.shape == mask_3d.shape, "SANITY FAIL: 3D Stacking dimension mismatch!"
    
    # --- D. Compute 3D Metrics ---
    p_mae, p_psnr, p_ssim = compute_3d_metrics(pred_3d, gt_3d, mask_3d)
    patient_metrics.append({
        'Patient_ID': pid,
        'Slices': len(patient_cbct_files),
        'MAE': p_mae,
        'PSNR': p_psnr,
        'SSIM': p_ssim
    })
    
    print(f"   3D Metrics -> MAE: {p_mae:.2f} | PSNR: {p_psnr:.2f} | SSIM: {p_ssim:.4f}")

    # --- E. NIfTI Saving & Metadata Preservation ---
    # CAST TO FLOAT32 TO FIX SIMPLEITK FLOAT16 INCOMPATIBILITY
    sitk_pred = sitk.GetImageFromArray(pred_3d.astype(np.float32))
    sitk_gt = sitk.GetImageFromArray(gt_3d.astype(np.float32))
    
    # Attempt to copy metadata from the original raw CT file
    # Assuming original file is named something like '2B01.nii.gz' or '2B01_CT.nii.gz'
    original_nifti_path = glob.glob(os.path.join(RAW_CT_NIFTI_DIR, f"*{pid}*.nii*"))
    
    if original_nifti_path:
        # Load raw header
        raw_img = sitk.ReadImage(original_nifti_path[0])
        
        # Apply physical metadata to our new synthetic volumes
        sitk_pred.SetSpacing(raw_img.GetSpacing())
        sitk_pred.SetOrigin(raw_img.GetOrigin())
        sitk_pred.SetDirection(raw_img.GetDirection())
        
        sitk_gt.SetSpacing(raw_img.GetSpacing())
        sitk_gt.SetOrigin(raw_img.GetOrigin())
        sitk_gt.SetDirection(raw_img.GetDirection())
        print(f"   [Sanity Pass] Metadata preserved from: {os.path.basename(original_nifti_path[0])}")
    else:
        print(f"   WARNING: Original NIfTI not found for {pid} in {RAW_CT_NIFTI_DIR}. Using default spacing.")

    # Save to disk
    sitk.WriteImage(sitk_pred, os.path.join(OUTPUT_DIR, f"{pid}_Synthetic_CT.nii.gz"))
    sitk.WriteImage(sitk_gt, os.path.join(OUTPUT_DIR, f"{pid}_Reference_CT.nii.gz"))

# ====================================================
# 5. FINAL AGGREGATION & REPORTING
# ====================================================
df_metrics = pd.DataFrame(patient_metrics)
df_metrics.to_csv(METRICS_CSV, index=False)

avg_mae = df_metrics['MAE'].mean()
avg_psnr = df_metrics['PSNR'].mean()
avg_ssim = df_metrics['SSIM'].mean()

summary_text = f"""==================================================
 3D SYNTHETIC CT RECONSTRUCTION SUMMARY
==================================================
Total Patients Reconstructed : {len(df_metrics)}
Total Slices Processed       : {df_metrics['Slices'].sum()}

--- AVERAGE 3D VOLUME METRICS ---
Mean 3D MAE  : {avg_mae:.2f} HU
Mean 3D PSNR : {avg_psnr:.2f} dB
Mean 3D SSIM : {avg_ssim:.4f}
==================================================
"""

with open(SUMMARY_TXT, "w") as f:
    f.write(summary_text)

print("\n" + summary_text)
print(f"All NIfTI files and metrics successfully saved to: {OUTPUT_DIR}/")


# In[2]:


import pandas as pd
import os

log_file = "training_log.csv"
if os.path.exists(log_file):
    df = pd.read_csv(log_file)
    print("--- LATEST LOG ENTRIES ---")
    print(df.tail(3).to_string(index=False))


# In[ ]:


import os
import csv
import torch
import numpy as np
from tqdm.auto import tqdm
from skimage.metrics import structural_similarity as compare_ssim

# ====================================================
# 1. LOSS FUNCTION DEFINITION
# ====================================================
class MaskedCompositeLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = torch.nn.L1Loss(reduction='none')

    def forward(self, pred, target, mask):
        loss = self.l1(pred, target)
        return (loss * mask).sum() / (mask.sum() + 1e-8)

# ====================================================
# 2. CONFIGURATION & DIRECTORIES
# ====================================================
DATA_DIR = "./Processed_Task2_Clean"
CHECKPOINT_DIR = "./checkpoints"
LOG_FILE = "training_log.csv"

os.makedirs(CHECKPOINT_DIR, exist_ok=True)

NUM_EPOCHS = 60
SAVE_EVERY = 5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ====================================================
# 3. DATALOADER FORCE-REBUILD (FIXED)
# ====================================================
# ✅ Now correctly using UFormerDataset and valid_ids
train_dataset = UFormerDataset(data_dir=DATA_DIR, valid_ids=train_ids)
val_dataset = UFormerDataset(data_dir=DATA_DIR, valid_ids=val_ids)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False)

# ====================================================
# 4. FRESH TRAINING INIT
# ====================================================
model = UFormer().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
criterion = MaskedCompositeLoss()
scaler = torch.cuda.amp.GradScaler()

# ====================================================
# 5. CSV LOGGER INITIALIZATION
# ====================================================
with open(LOG_FILE, mode='w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['epoch', 'train_loss', 'val_loss', 'val_mae', 'val_psnr', 'val_ssim'])

# ====================================================
# 6. MAIN TRAINING LOOP
# ====================================================
best_val_mae = float('inf')

print("==================================================")
print(" 🚀 INITIATING FRESH TRAINING RUN")
print("==================================================")
print(f"[>] Dataset Directory : {DATA_DIR}")
print(f"[>] Logging to        : {LOG_FILE}")
print(f"[>] Checkpoints to    : {CHECKPOINT_DIR}")
print("==================================================\n")

for epoch in range(NUM_EPOCHS):
    # --- TRAINING PHASE ---
    model.train()
    train_epoch_loss = 0.0
    
    train_bar = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{NUM_EPOCHS}] Train")
    for cbct_batch, ct_batch, mask_batch in train_bar:
        cbct_batch = cbct_batch.to(device).float()
        ct_batch = ct_batch.to(device).float()
        mask_batch = mask_batch.to(device).float()
        
        optimizer.zero_grad()
        
        with torch.cuda.amp.autocast():
            outputs = model(cbct_batch)
            loss = criterion(outputs, ct_batch, mask_batch)
            
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        train_epoch_loss += loss.item()
        train_bar.set_postfix({'Loss': f"{loss.item():.4f}"})
        
    avg_train_loss = train_epoch_loss / len(train_loader)
    
    # --- VALIDATION PHASE ---
    model.eval()
    val_epoch_loss = 0.0
    val_epoch_mae = 0.0
    val_epoch_psnr = 0.0
    val_epoch_ssim = 0.0
    
    val_bar = tqdm(val_loader, desc=f"Epoch [{epoch+1}/{NUM_EPOCHS}] Val")
    
    with torch.no_grad():
        for cbct_batch, ct_batch, mask_batch in val_bar:
            cbct_batch = cbct_batch.to(device).float()
            ct_batch = ct_batch.to(device).float()
            mask_batch = mask_batch.to(device).float()
            
            with torch.cuda.amp.autocast():
                outputs = model(cbct_batch)
                val_loss = criterion(outputs, ct_batch, mask_batch)
                
            val_epoch_loss += val_loss.item()
            
            # SAFE METRIC COMPUTATION (Batch Dim Preserved)
            outputs_np = outputs.cpu().squeeze(1).numpy()
            targets_np = ct_batch.cpu().squeeze(1).numpy()
            
            if mask_batch.ndim == 4:
                masks_np = mask_batch.cpu().squeeze(1).numpy()
            else:
                masks_np = mask_batch.cpu().numpy()
            
            # Conversion to Physical HU Space
            pred_hu = (outputs_np * 4000.0) - 1000.0
            target_hu = (targets_np * 4000.0) - 1000.0
            
            batch_size = outputs_np.shape[0]
            batch_mae = 0
            batch_psnr = 0
            batch_ssim = 0
            
            for i in range(batch_size):
                p_hu = pred_hu[i]
                t_hu = target_hu[i]
                m = masks_np[i].astype(bool)
                
                if np.sum(m) == 0:
                    continue
                    
                abs_diff = np.abs(p_hu - t_hu)
                mae = (abs_diff * m).sum() / m.sum()
                batch_mae += mae
                
                mse = ((abs_diff ** 2) * m).sum() / m.sum()
                psnr = 10 * np.log10((4000.0 ** 2) / mse) if mse > 0 else 100.0
                batch_psnr += psnr
                
                p_hu_masked = p_hu * m
                t_hu_masked = t_hu * m
                ssim_val = compare_ssim(t_hu_masked, p_hu_masked, data_range=4000.0)
                batch_ssim += ssim_val
                
            val_epoch_mae += (batch_mae / batch_size)
            val_epoch_psnr += (batch_psnr / batch_size)
            val_epoch_ssim += (batch_ssim / batch_size)
            
    avg_val_loss = val_epoch_loss / len(val_loader)
    avg_val_mae = val_epoch_mae / len(val_loader)
    avg_val_psnr = val_epoch_psnr / len(val_loader)
    avg_val_ssim = val_epoch_ssim / len(val_loader)
    
    print(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val MAE: {avg_val_mae:.2f} HU | Val PSNR: {avg_val_psnr:.2f} dB | Val SSIM: {avg_val_ssim:.4f}")

    # ====================================================
    # 7. LOGGING & CHECKPOINT SAVING
    # ====================================================
    with open(LOG_FILE, mode='a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([epoch+1, avg_train_loss, avg_val_loss, avg_val_mae, avg_val_psnr, avg_val_ssim])

    if avg_val_mae < best_val_mae:
        best_val_mae = avg_val_mae
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_mae': best_val_mae,
        }, os.path.join(CHECKPOINT_DIR, "best_model.pth"))
        print(f"🌟 New Best Model Saved! (MAE: {best_val_mae:.2f} HU)")
        
    if (epoch + 1) % SAVE_EVERY == 0:
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, os.path.join(CHECKPOINT_DIR, f"epoch_{epoch+1}.pth"))
        print(f"💾 Epoch {epoch+1} checkpoint saved.")

print("\n✅ Training Complete. All metrics logged to", LOG_FILE)


# In[2]:


import pandas as pd

# Load the securely saved CSV
df = pd.read_csv("training_log.csv")

# Display the last 5 epochs
print("--- FINAL 5 EPOCHS ---")
print(df.tail(5).to_string(index=False))

# Find the epoch with the best (lowest) Validation MAE
best_epoch = df.loc[df['val_mae'].idxmin()]
print("\n--- BEST OVERALL MODEL ---")
print(f"Epoch    : {best_epoch['epoch']}")
print(f"Val MAE  : {best_epoch['val_mae']:.2f} HU")
print(f"Val PSNR : {best_epoch['val_psnr']:.2f} dB")
print(f"Val SSIM : {best_epoch['val_ssim']:.4f}")


# In[1]:


get_ipython().system('pip install pytorch-msssim')


# In[3]:


import torch
import torch.nn as nn
from pytorch_msssim import ssim

class MaskedCompositeLoss(nn.Module):
    def __init__(self, l1_weight=0.60, ssim_weight=0.25, grad_weight=0.15):
        super().__init__()
        self.l1_weight = l1_weight
        self.ssim_weight = ssim_weight
        self.grad_weight = grad_weight
        self.l1_criterion = nn.L1Loss(reduction='none')

    def forward(self, pred, target, mask):
        # 1. Create the boolean tissue mask
        active_tissue = (mask > 0).float()
        tissue_pixel_count = active_tissue.sum() + 1e-8 

        # 2. Masked L1 Loss
        raw_l1 = self.l1_criterion(pred, target)
        loss_l1 = (raw_l1 * active_tissue).sum() / tissue_pixel_count

        # 3. Masked SSIM Loss
        pred_masked = pred * active_tissue
        target_masked = target * active_tissue
        ssim_val = ssim(pred_masked, target_masked, data_range=1.0, size_average=True)
        loss_ssim = 1.0 - ssim_val

        # 4. Masked Gradient Loss (Edge detection)
        dy_pred = pred[:, :, 1:, :] - pred[:, :, :-1, :]
        dy_target = target[:, :, 1:, :] - target[:, :, :-1, :]
        dx_pred = pred[:, :, :, 1:] - pred[:, :, :, :-1]
        dx_target = target[:, :, :, 1:] - target[:, :, :, :-1]

        mask_dy = active_tissue[:, :, 1:, :]
        mask_dx = active_tissue[:, :, :, 1:]

        grad_loss_y = (torch.abs(dy_pred - dy_target) * mask_dy).sum() / (mask_dy.sum() + 1e-8)
        grad_loss_x = (torch.abs(dx_pred - dx_target) * mask_dx).sum() / (mask_dx.sum() + 1e-8)
        loss_grad = grad_loss_y + grad_loss_x

        # 5. Final Combined Equation
        total_loss = (self.l1_weight * loss_l1) + (self.ssim_weight * loss_ssim) + (self.grad_weight * loss_grad)
        return total_loss


# In[6]:


# ==========================================
# 3. TRAINING LOOP (CRASH-RESISTANT & RESUME SUPPORT)
# ==========================================
import os
import csv
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from datetime import datetime
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from pytorch_msssim import ssim

# --- DIRECTORY SETTINGS ---
DATA_DIR = "./Processed_Task2"
SAVE_DIR = "./checkpoints"  
BATCH_SIZE = 8       
LEARNING_RATE = 2e-4
NUM_EPOCHS = 60  

# --- RESUME SETTINGS ---
# To resume, change None to the path: e.g., "./checkpoints/epoch_015.pth"
RESUME_CHECKPOINT = None  

os.makedirs(SAVE_DIR, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- INITIALIZE FILE PATHS ---
csv_file = os.path.join(SAVE_DIR, "training_log.csv")
metrics_file = os.path.join(SAVE_DIR, "latest_metrics.txt")
heartbeat_file = os.path.join(SAVE_DIR, "heartbeat.txt")

# Initialize CSV with headers if starting fresh
if not RESUME_CHECKPOINT or not os.path.exists(csv_file):
    with open(csv_file, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'train_loss', 'val_loss', 'val_mae', 'val_psnr', 'val_ssim', 'learning_rate'])
        f.flush()
        os.fsync(f.fileno())

# ==========================================================
# 1. INITIALIZE DATALOADERS & MODEL
# ==========================================================
train_dataset = UFormerDataset(data_dir=DATA_DIR, valid_ids=train_ids)
val_dataset = UFormerDataset(data_dir=DATA_DIR, valid_ids=val_ids) 

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True) 

model = UFormer().to(device)
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)

criterion = MaskedCompositeLoss() 
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)

# --- RESUME LOGIC ---
start_epoch = 0
best_val_mae = float('inf')
best_epoch = 0

if RESUME_CHECKPOINT and os.path.exists(RESUME_CHECKPOINT):
    print(f"\n[!] Resuming training from: {RESUME_CHECKPOINT}")
    checkpoint = torch.load(RESUME_CHECKPOINT, map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch']
    
    if 'best_mae' in checkpoint:
        best_val_mae = checkpoint['best_mae']
    if 'best_epoch' in checkpoint:
        best_epoch = checkpoint['best_epoch']
        
    print(f"[!] Successfully restored model and optimizer. Resuming at Epoch {start_epoch + 1}")

print(f"\n--- Starting COMPOSITE Training on {len(train_ids)} Patients | Validating on {len(val_ids)} Patients ---")

# ==========================================================
# 2. MAIN EPOCH LOOP
# ==========================================================
for epoch in range(start_epoch, NUM_EPOCHS):
    
    # -----------------------
    # PHASE 1: TRAINING
    # -----------------------
    model.train()
    train_epoch_loss = 0.0
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [TRAIN]", leave=True)

    for cbct_batch, ct_batch, mask_batch in progress_bar:
        cbct_batch = cbct_batch.to(device)
        ct_batch = ct_batch.to(device)
        mask_batch = mask_batch.to(device) 

        optimizer.zero_grad()
        synthetic_ct = model(cbct_batch)
        
        loss = criterion(synthetic_ct, ct_batch, mask_batch)
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        train_epoch_loss += loss.item()
        progress_bar.set_postfix({"Train Loss": f"{loss.item():.4f}"})

    avg_train_loss = train_epoch_loss / len(train_loader)
    
    # -----------------------
    # PHASE 2: VALIDATION
    # -----------------------
    model.eval()
    val_epoch_loss = 0.0
    val_epoch_mae = 0.0
    val_epoch_psnr = 0.0
    val_epoch_ssim = 0.0
    
    with torch.no_grad():
        for cbct_batch, ct_batch, mask_batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [VAL]", leave=False):
            cbct_batch = cbct_batch.to(device)
            ct_batch = ct_batch.to(device)
            mask_batch = mask_batch.to(device)
            
            synthetic_ct = model(cbct_batch)
            
            # Loss Calculation
            loss = criterion(synthetic_ct, ct_batch, mask_batch)
            val_epoch_loss += loss.item()
            
            # Real-Time Metric Calculation
            active_tissue = (mask_batch > 0).float()
            tissue_pixel_count = active_tissue.sum() + 1e-8
            
            # MAE
            abs_diff_hu = torch.abs(synthetic_ct - ct_batch) * 4000.0
            masked_mae = (abs_diff_hu * active_tissue).sum() / tissue_pixel_count
            val_epoch_mae += masked_mae.item()
            
            # PSNR
            squared_diff_hu = (abs_diff_hu ** 2)
            masked_mse = (squared_diff_hu * active_tissue).sum() / tissue_pixel_count
            if masked_mse.item() > 0:
                batch_psnr = 10 * np.log10((4000.0 ** 2) / masked_mse.item())
            else:
                batch_psnr = 100.0 
            val_epoch_psnr += batch_psnr
            
            # SSIM
            pred_masked = synthetic_ct * active_tissue
            target_masked = ct_batch * active_tissue
            batch_ssim = ssim(pred_masked, target_masked, data_range=1.0, size_average=True)
            val_epoch_ssim += batch_ssim.item()
            
    # Calculate Averages
    avg_val_loss = val_epoch_loss / len(val_loader)
    avg_val_mae = val_epoch_mae / len(val_loader)
    avg_val_psnr = val_epoch_psnr / len(val_loader)
    avg_val_ssim = val_epoch_ssim / len(val_loader)
    current_lr = optimizer.param_groups[0]['lr']
    
    # -----------------------
    # PHASE 3: CHECKPOINT & LOGGING
    # -----------------------
    
    # Track Best Epoch
    is_best = False
    if avg_val_mae < best_val_mae:
        best_val_mae = avg_val_mae
        best_epoch = epoch + 1
        is_best = True

    # 1. Terminal Output
    print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")
    print(f"Train Loss: {avg_train_loss:.4f}")
    print(f"Val Loss: {avg_val_loss:.4f}")
    print(f"Val MAE: {avg_val_mae:.2f} HU")
    print(f"Val PSNR: {avg_val_psnr:.2f} dB")
    print(f"Val SSIM: {avg_val_ssim:.4f}")
    print(f"Best MAE: {best_val_mae:.2f} HU (Epoch {best_epoch})\n")
    
    # 2. Append to CSV Log (with OS buffer flush)
    with open(csv_file, mode='a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([epoch+1, f"{avg_train_loss:.4f}", f"{avg_val_loss:.4f}", 
                         f"{avg_val_mae:.2f}", f"{avg_val_psnr:.2f}", f"{avg_val_ssim:.4f}", f"{current_lr:.6f}"])
        f.flush()
        os.fsync(f.fileno())

    # 3. Update Latest Metrics File (with OS buffer flush)
    with open(metrics_file, mode='w') as f:
        f.write(f"Epoch: {epoch+1}\n")
        f.write(f"Train Loss: {avg_train_loss:.4f}\n")
        f.write(f"Val Loss: {avg_val_loss:.4f}\n")
        f.write(f"Val MAE: {avg_val_mae:.2f} HU\n")
        f.write(f"Val PSNR: {avg_val_psnr:.2f} dB\n")
        f.write(f"Val SSIM: {avg_val_ssim:.4f}\n")
        f.write(f"Best Epoch: {best_epoch}\n")
        f.write(f"Best MAE: {best_val_mae:.2f} HU\n")
        f.flush()
        os.fsync(f.fileno())

    # 4. Update Heartbeat File (with OS buffer flush)
    with open(heartbeat_file, mode='w') as f:
        f.write("Training Alive\n")
        f.write(f"Current Epoch: {epoch+1}\n")
        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.flush()
        os.fsync(f.fileno())

    # 5. Save Standard Epoch Checkpoint
    checkpoint_state = {
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_mae': avg_val_mae,
        'val_psnr': avg_val_psnr,
        'val_ssim': avg_val_ssim,
        'best_mae': best_val_mae,
        'best_epoch': best_epoch
    }
    
    epoch_save_path = os.path.join(SAVE_DIR, f"epoch_{epoch+1:03d}.pth")
    torch.save(checkpoint_state, epoch_save_path)
    
    # 6. Save Best Checkpoint
    if is_best:
        best_save_path = os.path.join(SAVE_DIR, "best_model.pth")
        torch.save(checkpoint_state, best_save_path)
        print(f"*** 🚀 NEW BEST MODEL SAVED! Val MAE dropped to {best_val_mae:.2f} HU ***\n")

    # --- SCHEDULER STEP ---
    scheduler.step(avg_val_loss)

print("🎉 Crash-Resistant Training Complete!")


# In[5]:


get_ipython().system('pip install SimpleITK')
get_ipython().system('pip install scikit-image')


# In[2]:


get_ipython().system('pip install --upgrade "numpy>=1.23.5,<2.0.0"')


# In[5]:


import os
import glob
import torch
import numpy as np
import pandas as pd
import SimpleITK as sitk
from tqdm.auto import tqdm
from skimage.metrics import structural_similarity as compare_ssim

# ==========================================
# 1. DIRECTORY CONFIGURATION
# ==========================================
PROCESSED_DIR = "./Processed_Task2"       
RAW_CT_DIR = "./Raw_Data/Task2/CT"        
CHECKPOINT_PATH = "./checkpoints/best_model.pth"

OUTPUT_BASE_DIR = "./BestModel_3D_TestResults"
NIFTI_OUT_DIR = os.path.join(OUTPUT_BASE_DIR, "NIfTI_Volumes")
METRICS_OUT_DIR = os.path.join(OUTPUT_BASE_DIR, "Metrics")

os.makedirs(NIFTI_OUT_DIR, exist_ok=True)
os.makedirs(METRICS_OUT_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# 2. VALIDATION & PRINT BLOCK
# ==========================================
print("==================================================")
print(" 3D PATIENT-LEVEL EVALUATION PIPELINE INITIALIZED")
print("==================================================")
print(f"[>] Checkpoint Loaded   : {CHECKPOINT_PATH}")
print(f"[>] Number of Patients  : {len(test_ids)} (Test Set ONLY)")
print(f"[>] Output Directory    : {OUTPUT_BASE_DIR}")
print("[!] VERIFIED: Executing True 3D Volumetric Evaluation")
print("==================================================\n")

# ==========================================
# 3. LOAD BEST MODEL
# ==========================================
model = UFormer().to(device)
checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# ==========================================
# 4. 3D RECONSTRUCTION & INFERENCE LOOP
# ==========================================
results = []

with torch.no_grad():
    for patient_id in tqdm(test_ids, desc="Processing Test Patients (3D)"):
        
        # 1. Locate and sort all 2D CBCT slices for this specific patient
        slice_paths = sorted(
            glob.glob(f"{PROCESSED_DIR}/cbct/{patient_id}_*.npy"), 
            key=lambda x: int(x.split('_')[-1].split('.npy')[0])
        )
        
        if not slice_paths:
            print(f"Warning: No slices found for {patient_id}. Skipping.")
            continue
            
        num_slices = len(slice_paths)
        
        # Determine spatial dimensions from the first slice
        sample_slice = np.load(slice_paths[0])
        H, W = sample_slice.shape
        
        # Initialize empty 3D arrays for the reconstructed volumes
        synth_volume_hu = np.zeros((num_slices, H, W), dtype=np.float32)
        gt_volume_hu = np.zeros((num_slices, H, W), dtype=np.float32)
        mask_volume = np.zeros((num_slices, H, W), dtype=bool)

        # 2. Slice-by-Slice Inference & Z-Axis Stacking
        for z, cbct_path in enumerate(slice_paths):
            ct_path = cbct_path.replace('/cbct/', '/ct/')
            
            cbct_np = np.load(cbct_path) 
            ct_np = np.load(ct_path)     
            
            active_tissue = (ct_np > 0)
            
            # --- FIX: .float() added here to match GPU weight precision ---
            cbct_tensor = torch.from_numpy(cbct_np).unsqueeze(0).unsqueeze(0).to(device).float()
            
            synth_tensor = model(cbct_tensor)
            synth_np = synth_tensor.cpu().squeeze().numpy()
            
            # Convert back to strict Physical HU Space [-1000, 3000]
            synth_hu = (synth_np * 4000.0) - 1000.0
            gt_hu = (ct_np * 4000.0) - 1000.0
            
            synth_volume_hu[z, :, :] = synth_hu
            gt_volume_hu[z, :, :] = gt_hu
            mask_volume[z, :, :] = active_tissue

        # Final Sanity Check Print
        if patient_id == test_ids[0]:
            print(f"\n[Sanity Check] Patient ID: {patient_id}")
            print(f"Synthetic Volume Shape : {synth_volume_hu.shape}")
            print(f"Reference Volume Shape : {gt_volume_hu.shape}")
            assert synth_volume_hu.shape == gt_volume_hu.shape, "SHAPE MISMATCH DETECTED"

        # 3. Load Original CT Metadata
        raw_ct_path = os.path.join(RAW_CT_DIR, f"{patient_id}.nii.gz")
        if os.path.exists(raw_ct_path):
            original_ct_itk = sitk.ReadImage(raw_ct_path)
            spacing = original_ct_itk.GetSpacing()
            origin = original_ct_itk.GetOrigin()
            direction = original_ct_itk.GetDirection()
        else:
            spacing, origin, direction = (1.0, 1.0, 1.0), (0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)

        # 4. Save NIfTI Volumes with Preserved Metadata
        synth_itk = sitk.GetImageFromArray(synth_volume_hu)
        synth_itk.SetSpacing(spacing)
        synth_itk.SetOrigin(origin)
        synth_itk.SetDirection(direction)
        
        gt_itk = sitk.GetImageFromArray(gt_volume_hu)
        gt_itk.SetSpacing(spacing)
        gt_itk.SetOrigin(origin)
        gt_itk.SetDirection(direction)

        sitk.WriteImage(synth_itk, os.path.join(NIFTI_OUT_DIR, f"{patient_id}_Synthetic_CT.nii.gz"))
        sitk.WriteImage(gt_itk, os.path.join(NIFTI_OUT_DIR, f"{patient_id}_Reference_CT.nii.gz"))

        # ==========================================
        # 5. EXACT 3D METRIC CALCULATION (MASKED)
        # ==========================================
        tissue_voxels = mask_volume.sum()
        if tissue_voxels == 0:
            continue
            
        # A. 3D MAE (HU)
        abs_diff_3d = np.abs(synth_volume_hu - gt_volume_hu)
        mae_3d = (abs_diff_3d * mask_volume).sum() / tissue_voxels
        
        # B. 3D PSNR
        squared_diff_3d = abs_diff_3d ** 2
        mse_3d = (squared_diff_3d * mask_volume).sum() / tissue_voxels
        psnr_3d = 10 * np.log10((4000.0 ** 2) / mse_3d) if mse_3d > 0 else 100.0
        
        # C. 3D SSIM
        synth_masked = synth_volume_hu * mask_volume
        gt_masked = gt_volume_hu * mask_volume
        
        ssim_3d = compare_ssim(
            gt_masked, 
            synth_masked, 
            data_range=4000.0, 
            channel_axis=None 
        )
        
        results.append({
            'patient_id': patient_id,
            'mae': mae_3d,
            'psnr': psnr_3d,
            'ssim': ssim_3d
        })

# ==========================================
# 6. SAVE FINAL METRICS & SUMMARY
# ==========================================
df_results = pd.DataFrame(results)
csv_out = os.path.join(METRICS_OUT_DIR, "final_patient_metrics.csv")
df_results.to_csv(csv_out, index=False)

avg_mae = df_results['mae'].mean()
avg_psnr = df_results['psnr'].mean()
avg_ssim = df_results['ssim'].mean()

best_patient = df_results.loc[df_results['mae'].idxmin()]['patient_id']
worst_patient = df_results.loc[df_results['mae'].idxmax()]['patient_id']

summary_text = f"""FINAL 3D CLINICAL EVALUATION SUMMARY
====================================
Total Test Patients : {len(df_results)}

--- AVERAGE METRICS ---
Average MAE  : {avg_mae:.2f} HU
Average PSNR : {avg_psnr:.2f} dB
Average SSIM : {avg_ssim:.4f}

--- EXTREMES ---
Best Patient (Lowest Error) : {best_patient}
Worst Patient (Highest Error): {worst_patient}
"""

with open(os.path.join(METRICS_OUT_DIR, "summary_metrics.txt"), "w") as f:
    f.write(summary_text)

print(summary_text)
print(f"✅ 3D Reconstruction and Evaluation Complete. Files saved to: {OUTPUT_BASE_DIR}")


# In[6]:


import numpy as np
import glob

cbct_file = sorted(glob.glob("./Processed_Task2/cbct/*.npy"))[0]
ct_file = cbct_file.replace("/cbct/", "/ct/")

cbct = np.load(cbct_file)
ct = np.load(ct_file)

print("CBCT mean:", cbct.mean())
print("CT mean:", ct.mean())

print("Average difference:",
      np.mean(np.abs(cbct - ct)))


# In[7]:


import os

cbct_files = sorted(os.listdir("./Processed_Task2/cbct"))[:5]
ct_files = sorted(os.listdir("./Processed_Task2/ct"))[:5]

print("CBCT:")
print(cbct_files)

print("\nCT:")
print(ct_files)


# In[8]:


import numpy as np
import glob
import random

cbct_files = glob.glob("./Processed_Task2/cbct/*.npy")

for i in range(5):
    cbct_file = random.choice(cbct_files)
    ct_file = cbct_file.replace("/cbct/", "/ct/")

    cbct = np.load(cbct_file)
    ct = np.load(ct_file)

    diff = np.mean(np.abs(cbct - ct))

    print(os.path.basename(cbct_file))
    print("Difference:", diff)
    print("-"*30)


# In[4]:


# ==========================================
# 4. 3D INFERENCE & STITCHING PIPELINE
# ==========================================
import os
import glob
import torch
import numpy as np
import SimpleITK as sitk
from tqdm.auto import tqdm

DATA_DIR = "./Processed_Task2"
WEIGHTS_PATH = "./NIT_Final_UFormer_Weights/best_uformer.pth"  
OUTPUT_DIR = "./NIT_Unseen_3D_Volumes"

os.makedirs(OUTPUT_DIR, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. Load the "Best" Model Weights from your 7 Epochs
print(f"Loading Model Weights from: {WEIGHTS_PATH}")
model = UFormer().to(device)
model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device))
model.eval()  # CRITICAL: Locks the weights, preventing any further training

# 2. Process strictly the 18 unseen patients in test_ids
print(f"\n--- Starting 3D Reconstruction for {len(test_ids)} Unseen Brain Patients ---")

with torch.no_grad(): # Disables memory-heavy gradient calculations
    for patient_id in tqdm(test_ids, desc="Processing Patients"):
        
        # Grab all 2D CBCT slices for this specific patient, sorted properly
        search_pattern = os.path.join(DATA_DIR, "cbct", f"{patient_id}_*.npy")
        patient_slices = sorted(glob.glob(search_pattern))
        
        if not patient_slices:
            continue
            
        synthetic_volume_list = []
        
        # 3. Predict the Synthetic CT slice by slice
        for slice_path in patient_slices:
            cbct_np = np.load(slice_path)
            
            # Format shape to [Batch=1, Channels=1, Height, Width]
            cbct_tensor = torch.tensor(cbct_np, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
            
            # Push through the AI
            pred_tensor = model(cbct_tensor)
            
            # Strip extra dimensions and move to CPU RAM
            pred_np = pred_tensor.squeeze().cpu().numpy()
            synthetic_volume_list.append(pred_np)
            
        # 4. Stitch 2D slices into a 3D Volume (Depth, Height, Width)
        volume_3d = np.stack(synthetic_volume_list, axis=0)
        
        # 5. Physics Fix: Un-normalize from [0.0 to 1.0] back to [-1000 to 3000 HU]
        volume_3d_hu = (volume_3d * 4000.0) - 1000.0
        
        # 6. Convert to Medical Image and Save
        sitk_image = sitk.GetImageFromArray(volume_3d_hu)
        
        save_path = os.path.join(OUTPUT_DIR, f"{patient_id}_Synthetic_CT.nii.gz")
        sitk.WriteImage(sitk_image, save_path)

print(f"\n✅ SUCCESS: All {len(test_ids)} unseen 3D volumes successfully generated and saved to {OUTPUT_DIR}!")


# In[5]:


# ==========================================
# 5. FINAL 3D EVALUATION METRICS
# ==========================================
import os
import glob
import numpy as np
import SimpleITK as sitk
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
from tqdm.auto import tqdm

DATA_DIR = "./Processed_Task2"
SYNTH_DIR = "./NIT_Unseen_3D_Volumes"

# Lists to store the scores for all 18 patients
all_mae = []
all_psnr = []
all_ssim = []

print(f"\n--- Evaluating {len(test_ids)} Unseen Test Patients ---")

for patient_id in tqdm(test_ids, desc="Calculating Metrics"):
    
    # 1. Load the AI-Generated 3D Volume
    synth_path = os.path.join(SYNTH_DIR, f"{patient_id}_Synthetic_CT.nii.gz")
    synth_img = sitk.ReadImage(synth_path)
    synth_vol = sitk.GetArrayFromImage(synth_img)
    
    # 2. Reconstruct the Ground Truth CT in 3D (to compare against)
    gt_slices_paths = sorted(glob.glob(os.path.join(DATA_DIR, "ct", f"{patient_id}_*.npy")))
    gt_slices = [np.load(p) for p in gt_slices_paths]
    
    # Stack into 3D and apply the same Hounsfield Unit physics fix
    gt_vol_normalized = np.stack(gt_slices, axis=0)
    gt_vol = (gt_vol_normalized * 4000.0) - 1000.0
    
    # 3. Calculate Mean Absolute Error (MAE)
    mae = np.mean(np.abs(gt_vol - synth_vol))
    all_mae.append(mae)
    
    # 4. Calculate PSNR & SSIM
    # We define the data_range as 4000 (from -1000 HU to 3000 HU)
    psnr = compare_psnr(gt_vol, synth_vol, data_range=4000.0)
    all_psnr.append(psnr)
    
    # Calculate SSIM slice-by-slice for optimal accuracy on 3D medical volumes
    patient_ssim = []
    for i in range(gt_vol.shape[0]):
        slice_ssim = compare_ssim(gt_vol[i], synth_vol[i], data_range=4000.0)
        patient_ssim.append(slice_ssim)
    all_ssim.append(np.mean(patient_ssim))

# 5. Print the Final Presentation-Ready Scores
print("\n" + "="*40)
print("🏆 FINAL UNSEEN 3D TEST RESULTS 🏆")
print("="*40)
print(f"Total Patients Tested : {len(test_ids)}")
print(f"Average MAE           : {np.mean(all_mae):.4f} HU")
print(f"Average PSNR          : {np.mean(all_psnr):.4f} dB")
print(f"Average SSIM          : {np.mean(all_ssim):.4f}")
print("="*40)


# In[6]:


# ==========================================
# 6. MASKED CLINICAL 3D EVALUATION
# ==========================================
import os
import glob
import numpy as np
import SimpleITK as sitk
from skimage.metrics import structural_similarity as compare_ssim
from tqdm.auto import tqdm

DATA_DIR = "./Processed_Task2"
SYNTH_DIR = "./NIT_Unseen_3D_Volumes"

# Lists to store the masked scores
all_masked_mae = []
all_masked_psnr = []
all_masked_ssim = []

print(f"\n--- Running Masked Clinical Evaluation on {len(test_ids)} Patients ---")

for patient_id in tqdm(test_ids, desc="Calculating Masked Metrics"):
    
    # 1. Load the AI-Generated 3D Volume
    synth_path = os.path.join(SYNTH_DIR, f"{patient_id}_Synthetic_CT.nii.gz")
    synth_img = sitk.ReadImage(synth_path)
    synth_vol = sitk.GetArrayFromImage(synth_img)
    
    # 2. Reconstruct Ground Truth CT
    gt_slices_paths = sorted(glob.glob(os.path.join(DATA_DIR, "ct", f"{patient_id}_*.npy")))
    gt_vol_normalized = np.stack([np.load(p) for p in gt_slices_paths], axis=0)
    gt_vol = (gt_vol_normalized * 4000.0) - 1000.0
    
    # 3. Reconstruct the Patient MASK in 3D
    mask_slices_paths = sorted(glob.glob(os.path.join(DATA_DIR, "mask", f"{patient_id}_*.npy")))
    mask_vol = np.stack([np.load(p) for p in mask_slices_paths], axis=0)
    
    # Create a boolean array where True = Patient Tissue, False = Background Air
    boolean_mask = mask_vol > 0
    
    # --- CALCULATE STRICTLY ON TISSUE ---
    
    # MAE (Only on masked pixels)
    mae = np.mean(np.abs(gt_vol[boolean_mask] - synth_vol[boolean_mask]))
    all_masked_mae.append(mae)
    
    # PSNR (Only on masked pixels)
    mse = np.mean((gt_vol[boolean_mask] - synth_vol[boolean_mask])**2)
    if mse == 0:
        psnr = 100 # Perfect score fallback
    else:
        psnr = 10 * np.log10((4000.0**2) / mse)
    all_masked_psnr.append(psnr)
    
    # SSIM (Calculated spatially, but averaged only inside the mask)
    patient_ssim = []
    for i in range(gt_vol.shape[0]):
        # Get the full structural similarity map for the slice
        score, ssim_map = compare_ssim(gt_vol[i], synth_vol[i], data_range=4000.0, full=True)
        
        slice_mask = boolean_mask[i]
        # Only append if there is actual tissue in this slice
        if np.any(slice_mask): 
            masked_ssim = np.mean(ssim_map[slice_mask])
            patient_ssim.append(masked_ssim)
            
    all_masked_ssim.append(np.mean(patient_ssim))

# 4. Print the Real Clinical Scores
print("\n" + "="*45)
print("🩺 REAL CLINICAL MASKED 3D RESULTS 🩺")
print("="*45)
print(f"Total Patients Tested : {len(test_ids)}")
print(f"Masked Average MAE    : {np.mean(all_masked_mae):.4f} HU")
print(f"Masked Average PSNR   : {np.mean(all_masked_psnr):.4f} dB")
print(f"Masked Average SSIM   : {np.mean(all_masked_ssim):.4f}")
print("="*45)


# In[7]:


import matplotlib.pyplot as plt
import numpy as np
import os
import glob

# 1. Grab the very first unseen test patient
sample_patient = test_ids[0]

# 2. Find their slices and grab one from the middle of the brain (e.g., slice 50)
cbct_files = sorted(glob.glob(f"./Processed_Task2/cbct/{sample_patient}_*.npy"))
sample_cbct_path = cbct_files[50] 

# 3. Find the exact matching Ground Truth CT file
sample_ct_path = sample_cbct_path.replace("cbct", "ct")

# 4. Load the numpy arrays
cbct_img = np.load(sample_cbct_path)
ct_img = np.load(sample_ct_path)

# 5. Plot them side-by-side
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.title(f"CBCT Input File\n{os.path.basename(sample_cbct_path)}")
plt.imshow(cbct_img, cmap="gray")
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title(f"Ground Truth CT File\n{os.path.basename(sample_ct_path)}")
plt.imshow(ct_img, cmap="gray")
plt.axis('off')

plt.tight_layout()
plt.show()

# 6. The Mathematical Truth
print("\n" + "="*50)
print("🔍 DIAGNOSTIC RESULTS 🔍")
print("="*50)
print(f"Are the two arrays mathematically identical? {np.array_equal(cbct_img, ct_img)}")
print("="*50)


# In[1]:


get_ipython().system('pip install SimpleITK')


# In[2]:





# In[2]:


import os

save_dir = "./saved_models"
print("Files in saved_models:")

for file in os.listdir(save_dir):
    filepath = os.path.join(save_dir, file)
    size_mb = os.path.getsize(filepath) / (1024 * 1024)
    print(f" - {file} (Size: {size_mb:.2f} MB)")


# In[6]:


import os
import random
import torch
import matplotlib.pyplot as plt
import numpy as np

# --- 1. Setup & Load the Model ---
DATA_DIR = "./Processed_Task2"
WEIGHTS_PATH = "./saved_models/uformer_epoch_12.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running inference on: {device}")

# Initialize the blank model architecture
model = UFormer().to(device)

# Load the trained weights from Epoch 12
model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device))

# CRITICAL: Put the model in evaluation mode (turns off dropout and gradients)
model.eval()
print("Model loaded and ready for inference.")

# --- 2. Get a Random Patient Scan ---
# We initialize the dataset to grab a sample
dataset = UFormerDataset(data_dir=DATA_DIR)

# Pick a random image from your 51,000+ slices
idx = random.randint(0, len(dataset) - 1)
cbct_tensor, ct_tensor, mask_tensor = dataset[idx]

# Add a "batch" dimension of 1, because the model expects [B, C, H, W]
cbct_input = cbct_tensor.unsqueeze(0).to(device)

# --- 3. Generate the Synthetic CT ---
print("Generating Synthetic CT...")
# torch.no_grad() saves memory and speeds up inference by not calculating gradients
with torch.no_grad():
    sct_tensor = model(cbct_input)

# --- 4. Prepare Images for Matplotlib ---
# Move tensors back to CPU, remove batch/channel dimensions, and convert to numpy
cbct_img = cbct_tensor.squeeze().cpu().numpy()
ct_img = ct_tensor.squeeze().cpu().numpy()
sct_img = sct_tensor.squeeze().cpu().numpy()

# --- 5. Plot the Results Side-by-Side ---
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Plot 1: Input CBCT (The noisy original)
axes[0].imshow(cbct_img, cmap='gray')
axes[0].set_title('Input: CBCT (Raw)', fontsize=14)
axes[0].axis('off')

# Plot 2: Synthetic CT (The AI's generation)
axes[1].imshow(sct_img, cmap='gray')
axes[1].set_title('Output: AI Synthetic CT (Epoch 12)', fontsize=14, color='blue')
axes[1].axis('off')

# Plot 3: Ground Truth CT (The target goal)
axes[2].imshow(ct_img, cmap='gray')
axes[2].set_title('Target: Ground Truth CT', fontsize=14, color='green')
axes[2].axis('off')

plt.tight_layout()
plt.show()


# In[7]:


import os
import glob
import random
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import Dataset

# ==========================================
# 1. THE BLUEPRINTS (Dataset & Model)
# ==========================================
class UFormerDataset(Dataset):
    def __init__(self, data_dir):
        super().__init__()
        self.data_dir = data_dir
        self.cbct_files = sorted(glob.glob(os.path.join(data_dir, "cbct", "*.npy")))
    def __len__(self):
        return len(self.cbct_files)
    def __getitem__(self, idx):
        cbct_path = self.cbct_files[idx]
        filename = os.path.basename(cbct_path)
        ct_path = os.path.join(self.data_dir, "ct", filename)
        cbct = np.load(cbct_path)
        ct = np.load(ct_path)
        return torch.tensor(cbct, dtype=torch.float32).unsqueeze(0), torch.tensor(ct, dtype=torch.float32).unsqueeze(0)

class DepthWiseConv(nn.Module):
    def __init__(self, dim, kernel_size=3, padding=1):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=kernel_size, padding=padding, groups=dim)
    def forward(self, x):
        return self.dwconv(x)

class LeFF(nn.Module):
    def __init__(self, dim, seq_len=256, mlp_ratio=4):
        super().__init__()
        hidden_dim = int(dim * mlp_ratio)
        self.linear1 = nn.Linear(dim, hidden_dim)
        self.dwconv = DepthWiseConv(hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, dim)
        self.gelu = nn.GELU()
    def forward(self, x, H, W):
        B, N, C = x.shape
        x = self.linear1(x)
        x = x.transpose(1, 2).view(B, -1, H, W)
        x = self.dwconv(x)
        x = self.gelu(x)
        x = x.flatten(2).transpose(1, 2)
        x = self.linear2(x)
        return x

class WindowAttention(nn.Module):
    def __init__(self, dim, num_heads, win_size=8):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.win_size = win_size
        self.scale = (dim // num_heads) ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)
    def forward(self, x, H, W):
        B, N, C = x.shape
        head_dim = C // self.num_heads
        x = x.view(B, H, W, C)
        win_h, win_w = self.win_size, self.win_size
        num_win_h, num_win_w = H // win_h, W // win_w
        x = x.view(B, num_win_h, win_h, num_win_w, win_w, C)
        windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, win_h * win_w, C)
        total_wins = windows.shape[0]
        qkv = self.qkv(windows).view(total_wins, win_h * win_w, 3, self.num_heads, head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = (attn @ v).permute(0, 2, 1, 3).contiguous().view(total_wins, win_h * win_w, C)
        out = self.proj(out)
        out = out.view(B, num_win_h, num_win_w, win_h, win_w, C)
        out = out.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H * W, C)
        return out

class LeWinBlock(nn.Module):
    def __init__(self, dim, num_heads, win_size=8, mlp_ratio=4):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, num_heads, win_size)
        self.norm2 = nn.LayerNorm(dim)
        self.leff = LeFF(dim, mlp_ratio=mlp_ratio)
    def forward(self, x, H, W):
        x = x + self.attn(self.norm1(x), H, W)
        x = x + self.leff(self.norm2(x), H, W)
        return x

class UFormer(nn.Module):
    def __init__(self, in_chans=1, out_chans=1, embed_dim=32, win_size=8):
        super().__init__()
        self.win_size = win_size
        self.input_proj = nn.Conv2d(in_chans, embed_dim, kernel_size=3, padding=1)
        self.enc_block1 = LeWinBlock(dim=embed_dim, num_heads=1, win_size=win_size)
        self.down1 = nn.Conv2d(embed_dim, embed_dim * 2, kernel_size=4, stride=2, padding=1)
        self.enc_block2 = LeWinBlock(dim=embed_dim * 2, num_heads=2, win_size=win_size)
        self.down2 = nn.Conv2d(embed_dim * 2, embed_dim * 4, kernel_size=4, stride=2, padding=1)
        self.bottleneck = LeWinBlock(dim=embed_dim * 4, num_heads=4, win_size=win_size)
        self.up2 = nn.ConvTranspose2d(embed_dim * 4, embed_dim * 2, kernel_size=2, stride=2)
        self.dec_block2 = LeWinBlock(dim=embed_dim * 2, num_heads=2, win_size=win_size)
        self.up1 = nn.ConvTranspose2d(embed_dim * 2, embed_dim, kernel_size=2, stride=2)
        self.dec_block1 = LeWinBlock(dim=embed_dim, num_heads=1, win_size=win_size)
        self.output_proj = nn.Conv2d(embed_dim, out_chans, kernel_size=3, padding=1)
    def forward(self, x):
        x_proj = self.input_proj(x)
        B, C, H, W = x_proj.shape
        x_flat1 = x_proj.flatten(2).transpose(1, 2)
        x_enc1 = self.enc_block1(x_flat1, H, W).transpose(1, 2).view(B, C, H, W)
        x_down1 = self.down1(x_enc1)
        B2, C2, H2, W2 = x_down1.shape
        x_flat2 = x_down1.flatten(2).transpose(1, 2)
        x_enc2 = self.enc_block2(x_flat2, H2, W2).transpose(1, 2).view(B2, C2, H2, W2)
        x_down2 = self.down2(x_enc2)
        B3, C3, H3, W3 = x_down2.shape
        x_bot_flat = x_down2.flatten(2).transpose(1, 2)
        x_bot = self.bottleneck(x_bot_flat, H3, W3).transpose(1, 2).view(B3, C3, H3, W3)
        x_up2 = self.up2(x_bot)
        x_up2 = x_up2 + x_enc2
        x_up2_flat = x_up2.flatten(2).transpose(1, 2)
        x_dec2 = self.dec_block2(x_up2_flat, H2, W2).transpose(1, 2).view(B2, C2, H2, W2)
        x_up1 = self.up1(x_dec2)
        x_up1 = x_up1 + x_enc1
        x_up1_flat = x_up1.flatten(2).transpose(1, 2)
        x_dec1 = self.dec_block1(x_up1_flat, H, W).transpose(1, 2).view(B, C, H, W)
        out = self.output_proj(x_dec1)
        return out

# ==========================================
# 2. INFERENCE & VISUALIZATION
# ==========================================
DATA_DIR = "./Processed_Task2"
WEIGHTS_PATH = "./saved_models/uformer_epoch_12.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running inference on: {device}")

# Build the model and load the weights
model = UFormer().to(device)
model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device))
model.eval()  # Lock the weights
print("Model loaded and ready for inference.")

# Grab a random image
dataset = UFormerDataset(data_dir=DATA_DIR)
idx = random.randint(0, len(dataset) - 1)
cbct_tensor, ct_tensor = dataset[idx]

# Run it through the AI
cbct_input = cbct_tensor.unsqueeze(0).to(device)
with torch.no_grad():
    sct_tensor = model(cbct_input)

# Prep for Matplotlib
cbct_img = cbct_tensor.squeeze().cpu().numpy()
ct_img = ct_tensor.squeeze().cpu().numpy()
sct_img = sct_tensor.squeeze().cpu().numpy()

# Plot them
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

axes[0].imshow(cbct_img, cmap='gray')
axes[0].set_title('Input: CBCT (Raw)', fontsize=14)
axes[0].axis('off')

axes[1].imshow(sct_img, cmap='gray')
axes[1].set_title('Output: AI Synthetic CT (Epoch 12)', fontsize=14, color='blue')
axes[1].axis('off')

axes[2].imshow(ct_img, cmap='gray')
axes[2].set_title('Target: Ground Truth CT', fontsize=14, color='green')
axes[2].axis('off')

plt.tight_layout()
plt.show()


# In[2]:


get_ipython().system('pip install "numpy<2.0"')


# In[4]:


get_ipython().system('pip install scikit-image "numpy<2.0"')


# In[5]:


import torch
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr_metric
from skimage.metrics import structural_similarity as ssim_metric

# --- 1. Compute Voxel-Wise Metrics ---
# Convert your numpy images from the previous cell to float64 for metric calculations
true_ct = ct_img.astype(np.float64)
synth_ct = sct_img.astype(np.float64)

# Calculate Mean Absolute Error (MAE)
mae_value = np.mean(np.abs(true_ct - synth_ct))

# Calculate Peak Signal-to-Noise Ratio (PSNR)
# dynamic_range is the difference between max and min values in your ground truth
data_range = true_ct.max() - true_ct.min()
psnr_value = psnr_metric(true_ct, synth_ct, data_range=data_range)

# Calculate Structural Similarity Index (SSIM)
ssim_value = ssim_metric(true_ct, synth_ct, data_range=data_range)

print("="*40)
print("       QUANTITATIVE ACCURACY REPORT       ")
print("="*40)
print(f"Mean Absolute Error (MAE):      {mae_value:.4f} HU (Lower is better)")
print(f"Peak Signal-to-Noise Ratio (PSNR): {psnr_value:.2f} dB  (Higher is better)")
print(f"Structural Similarity (SSIM):    {ssim_value:.4f}     (Closer to 1 is better)")
print("="*40)

# --- 2. Advanced Clinical Check: HU Line Profile ---
# Choose a horizontal row across the middle of the image (e.g., row 128 of a 256x256 image)
# This usually cuts right through the pelvic bones and soft tissue
row_idx = true_ct.shape[0] // 2

true_profile = true_ct[row_idx, :]
synth_profile = synth_ct[row_idx, :]

plt.figure(figsize=(12, 4))
plt.plot(true_profile, label='Ground Truth CT', linewidth=2)
plt.plot(synth_profile, label='AI Synthetic CT', linestyle='--')
plt.title(f'Hounsfield Unit (HU) Profile Line across Row {row_idx}', fontsize=12)
plt.xlabel('Voxel Position (X-Axis)')
plt.ylabel('Intensity (Hounsfield Units)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()


# In[6]:


import numpy as np
import matplotlib.pyplot as plt

# --- SAFETY CHECK ---
if 'ct_img' not in locals() or 'sct_img' not in locals():
    print("⚠️ WAIT! Python's memory is empty.")
    print("Please run your Inference cell first!")
else:
    # 1. Create Biological Masks (Separate dense tissue from air/background)
    # We will use the mean intensity of the ground truth to draw the boundary
    threshold = true_ct.mean()
    
    # Create binary maps (1 for tissue, 0 for air)
    true_anatomy = (true_ct > threshold).astype(np.float64)
    synth_anatomy = (synth_ct > threshold).astype(np.float64)

    # 2. Calculate Dice Similarity Coefficient (DSC)
    # Formula: 2 * (Overlap Area) / (Total Area 1 + Total Area 2)
    intersection = np.sum(true_anatomy * synth_anatomy)
    dice_score = (2.0 * intersection) / (np.sum(true_anatomy) + np.sum(synth_anatomy) + 1e-8)

    # 3. Print the Clinical Report
    print("="*45)
    print(" 🏥 CLINICAL & ANATOMICAL REPORT ")
    print("="*45)
    print(f"Dice Similarity Coefficient (DSC): {dice_score:.4f}")
    print("-" * 45)
    
    if dice_score > 0.95:
        print("Diagnosis:  Structures align perfectly.")
    elif dice_score > 0.85:
        print("Diagnosis: Great overlap, but minor edge blurring exists.")
    else:
        print("Diagnosis: The shapes are mismatched. Check for spatial shifting.")
        
    # 4. Visualize the Overlap and Errors!
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.imshow(true_anatomy, cmap='gray')
    plt.title('Target: True Anatomy Boundary', fontsize=12)
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.imshow(synth_anatomy, cmap='gray')
    plt.title('AI: Predicted Anatomy Boundary', fontsize=12)
    plt.axis('off')
    
    # Show exactly where the AI made a mistake (Difference map)
    error_map = np.abs(true_anatomy - synth_anatomy)
    plt.subplot(1, 3, 3)
    plt.imshow(error_map, cmap='hot')
    plt.title('Anatomical Errors (Black = Perfect, Bright = Error)', fontsize=12)
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()


# In[1]:


import zipfile
import os

# Since the zip is in the same folder as the notebook, just use the filename!
zip_path = "nii-20260604T054237Z-3-001.zip"
extract_path = "./Unzipped_Nii_Data"

# Extract the files
print("📦 Extracting dataset on the server...")
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_path)
    
print(f"✅ Extracted completely to: {extract_path}")


# In[3]:


get_ipython().system('pip install nibabel tqdm')


# In[2]:


import os
sample_folder = "./Unzipped_Nii_Data/nii/18RT1337"
print(f"Files inside 18RT1337:\n{os.listdir(sample_folder)}")


# In[3]:


import os
import glob
import numpy as np
import nibabel as nib
from tqdm.auto import tqdm

# =========================================================================
# 1. PATH CONFIGURATION
# =========================================================================
RAW_DATA_DIR = "./Unzipped_Nii_Data/nii" 
PROCESSED_TEST_DIR = "./Processed_Test_Data"

# Create clean destination directories for your test slices
os.makedirs(os.path.join(PROCESSED_TEST_DIR, "cbct"), exist_ok=True)
os.makedirs(os.path.join(PROCESSED_TEST_DIR, "ct"), exist_ok=True)

# Find all patient folders (18RT1337, etc.) and sort them alphabetically
patient_folders = sorted([f for f in glob.glob(os.path.join(RAW_DATA_DIR, "*")) if os.path.isdir(f)])
print(f"📊 Found {len(patient_folders)} patient folder(s) for testing.")

# =========================================================================
# 2. INTENSITY NORMALIZATION FUNCTION
# =========================================================================
def normalize_hu(image_array):
    """
    Clips Hounsfield Units to standard radiotherapy ranges [-1000, 1000]
    and scales them to [0, 1] for the neural network.
    """
    min_hu, max_hu = -1000.0, 1000.0
    clipped = np.clip(image_array, min_hu, max_hu)
    normalized = (clipped - min_hu) / (max_hu - min_hu)
    return normalized

# =========================================================================
# 3. Z-AXIS SLICING PIPELINE
# =========================================================================
for folder in patient_folders:
    patient_id = os.path.basename(folder)
    
    # 1. Grab EVERY file in the folder, regardless of name
    all_files = os.listdir(folder)
    
    # 2. Search for 'cbct' and 'fbct' ignoring uppercase/lowercase
    cbct_files = [os.path.join(folder, f) for f in all_files if 'cbct' in f.lower() and '.nii' in f.lower()]
    fbct_files = [os.path.join(folder, f) for f in all_files if 'fbct' in f.lower() and '.nii' in f.lower()]
    
    if not cbct_files or not fbct_files:
        print(f"⚠️ Skipping folder {patient_id}: Could not find paired files.")
        continue
        
    print(f"\n👁️ Processing patient: {patient_id}")
    
    # ... (The rest of your loading and slicing code stays exactly the same!)
    cbct_vol = nib.load(cbct_files[0]).get_fdata()
    fbct_vol = nib.load(fbct_files[0]).get_fdata()
    # ...
    
    # Load 3D matrices into memory
    cbct_vol = nib.load(cbct_files[0]).get_fdata()
    fbct_vol = nib.load(fbct_files[0]).get_fdata()
    
    # Determine number of axial slices along the Z-axis
    num_slices = cbct_vol.shape[2]
    
    # Slice the volume layer-by-layer
    for z in tqdm(range(num_slices), leave=False, desc=f"Slicing {patient_id}"):
        cbct_slice = cbct_vol[:, :, z]
        fbct_slice = fbct_vol[:, :, z]
        
        # Apply normalization and convert to float32 to save server memory
        cbct_norm = normalize_hu(cbct_slice).astype(np.float32)
        fbct_norm = normalize_hu(fbct_slice).astype(np.float32)
        
        # Save pairs using an identical naming structure
        slice_filename = f"{patient_id}_slice_{z:03d}.npy"
        
        np.save(os.path.join(PROCESSED_TEST_DIR, "cbct", slice_filename), cbct_norm)
        np.save(os.path.join(PROCESSED_TEST_DIR, "ct", slice_filename), fbct_norm)

print("\n🎉 Preprocessing complete! Your unseen 2D slices are fully structured inside './Processed_Test_Data'")


# In[2]:


get_ipython().system('pip install "numpy<2.0.0"')


# In[ ]:


import shutil

shutil.make_archive("Processed_Task2", "zip", "Processed_Task2")


# In[2]:


get_ipython().system('pip install scikit-image')


# In[2]:


get_ipython().system('pip install --force-reinstall "numpy<2.0.0" "scikit-image" "matplotlib"')


# In[25]:


import torch
import numpy as np
from tqdm.auto import tqdm
from scipy.ndimage import binary_fill_holes
from skimage.metrics import peak_signal_noise_ratio as psnr_metric
from skimage.metrics import structural_similarity as ssim_metric

# 1. Configuration Setup
TEST_DATA_DIR = "./Processed_Test_Data"
WEIGHTS_PATH = "./saved_models/uformer_epoch_12.pth" 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load Model
model = UFormer().to(device)
model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device))
model.eval() 

# Load Dataset
test_dataset = UFormerDataset(data_dir=TEST_DATA_DIR)
total_slices = len(test_dataset)
print(f"🚀 Starting Full Evaluation on {total_slices} slices...")

# 2. Metric Accumulators
total_mae, total_rmse, total_psnr, total_ssim = 0.0, 0.0, 0.0, 0.0
valid_slices = 0  # Counter for slices that actually contain anatomy

# 3. The Full Evaluation Loop
with torch.no_grad():
    for idx in tqdm(range(total_slices), desc="Evaluating Dataset"):
        
        # Load one slice
        sample = test_dataset[idx]
        cbct_tensor, ct_tensor = sample[:2] if len(sample) >= 2 else sample
        
        # Run inference
        sct_tensor = model(cbct_tensor.unsqueeze(0).to(device))
        
        # Convert to numpy
        true_ct = ct_tensor.squeeze().cpu().numpy().astype(np.float64)
        synth_ct = sct_tensor.squeeze().cpu().numpy().astype(np.float64)
        
        # Create Dynamic Corner Mask
        bg_value = true_ct[0, 0]
        body_mask = (np.abs(true_ct - bg_value) > 0.05).astype(np.float64)
        body_mask = binary_fill_holes(body_mask).astype(np.float64)
        
        # CRITICAL: Skip slices that are 100% background air (e.g., top of the head)
        # If we don't skip these, the MAE math will divide by zero and crash.
        if np.sum(body_mask) < 100: 
            continue
            
        # Apply Mask
        true_ct_masked = true_ct * body_mask
        synth_ct_masked = synth_ct * body_mask
        data_range = true_ct_masked.max() - true_ct_masked.min()
        
        # Calculate single-slice metrics
        mae = np.mean(np.abs((true_ct - synth_ct)[body_mask > 0]))
        rmse = np.sqrt(np.mean(((true_ct - synth_ct)[body_mask > 0]) ** 2))
        psnr = psnr_metric(true_ct_masked, synth_ct_masked, data_range=data_range)
        ssim = ssim_metric(true_ct_masked, synth_ct_masked, data_range=data_range)
        
        # Add to running totals
        total_mae += mae
        total_rmse += rmse
        total_psnr += psnr
        total_ssim += ssim
        valid_slices += 1

# 4. Final Averaging
final_mae = total_mae / valid_slices
final_rmse = total_rmse / valid_slices
final_psnr = total_psnr / valid_slices
final_ssim = total_ssim / valid_slices

print("\n" + "="*50)
print("🏆 FINAL GRAND AVERAGE METRICS (ALL UNSEEN PATIENTS) 🏆")
print("="*50)
print(f"Total Slices Evaluated: {valid_slices} (Ignored {total_slices - valid_slices} empty slices)")
print(f"Global MAE:   {final_mae:.4f}")
print(f"Global RMSE:  {final_rmse:.4f}")
print(f"Global PSNR:  {final_psnr:.2f} dB")
print(f"Global SSIM:  {final_ssim:.4f}")
print("="*50)


# In[27]:


import torch
import numpy as np
from tqdm.auto import tqdm
from scipy.spatial import cKDTree
from scipy.ndimage import binary_erosion, center_of_mass

# ==========================================
# 1. GEOMETRIC EVALUATION FUNCTIONS
# ==========================================
def calculate_dice(true_mask, pred_mask):
    """Calculates the Dice Similarity Coefficient (Volume Overlap)."""
    intersection = np.logical_and(true_mask, pred_mask).sum()
    total_area = true_mask.sum() + pred_mask.sum()
    if total_area == 0: return 1.0 
    return 2.0 * intersection / total_area

def calculate_hd95(true_mask, pred_mask, pixel_spacing=1.0):
    """Calculates the 95th Percentile Hausdorff Distance (Boundary Sharpness)."""
    true_bnd = true_mask ^ binary_erosion(true_mask)
    pred_bnd = pred_mask ^ binary_erosion(pred_mask)
    true_pts, pred_pts = np.argwhere(true_bnd), np.argwhere(pred_bnd)
    if len(true_pts) == 0 or len(pred_pts) == 0: return np.nan
    
    tree_true = cKDTree(true_pts)
    tree_pred = cKDTree(pred_pts)
    dist_to_pred, _ = tree_true.query(pred_pts)
    dist_to_true, _ = tree_pred.query(true_pts)
    return np.percentile(np.concatenate([dist_to_pred, dist_to_true]), 95) * pixel_spacing

def calculate_com_shift(true_mask, pred_mask, pixel_spacing=1.0):
    """Calculates the Center of Mass Shift (Spatial Translation)."""
    com_true = center_of_mass(true_mask)
    com_pred = center_of_mass(pred_mask)
    if np.isnan(com_true).any() or np.isnan(com_pred).any(): return np.nan
    return np.linalg.norm(np.array(com_true) - np.array(com_pred)) * pixel_spacing

# ==========================================
# 2. INITIALIZATION
# ==========================================
TEST_DATA_DIR = "./Processed_Test_Data"
WEIGHTS_PATH = "./saved_models/uformer_epoch_12.pth" 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load Model
model = UFormer().to(device)
model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device))
model.eval() 

# Load Dataset
test_dataset = UFormerDataset(data_dir=TEST_DATA_DIR)
total_slices = len(test_dataset)
print(f"🚀 Running Geometry-Only Evaluation Loop on {total_slices} slices...")

# Metric Accumulators
total_dice, total_hd95, total_com = 0.0, 0.0, 0.0
valid_geom_slices = 0   

# ==========================================
# 3. GEOMETRY-ONLY EVALUATION LOOP
# ==========================================
with torch.no_grad():
    for idx in tqdm(range(total_slices), desc="Analyzing Anatomy Preservation"):
        
        sample = test_dataset[idx]
        cbct_tensor, ct_tensor = sample[:2] if len(sample) >= 2 else sample
        
        # Extract Ground Truth CT first to check if bone exists before running model forward pass
        true_ct = ct_tensor.squeeze().cpu().numpy().astype(np.float64)
        
        # Isolate dense bone structure (Skull)
        bone_threshold = true_ct.max() * 0.8 
        true_bone = true_ct > bone_threshold
        
        # Optimization: Skip slices with negligible bone structures to save forward-pass & KDTree computation time
        if np.sum(true_bone) <= 50: 
            continue
            
        # Run model inference only on slices containing target anatomy
        sct_tensor = model(cbct_tensor.unsqueeze(0).to(device))
        synth_ct = sct_tensor.squeeze().cpu().numpy().astype(np.float64)
        synth_bone = synth_ct > bone_threshold
        
        # Calculate Geometry Metrics
        dice = calculate_dice(true_bone, synth_bone)
        hd95 = calculate_hd95(true_bone, synth_bone, pixel_spacing=1.0)
        com = calculate_com_shift(true_bone, synth_bone, pixel_spacing=1.0)
        
        # Filter out NaN errors
        if not np.isnan(hd95) and not np.isnan(com):
            total_dice += dice
            total_hd95 += hd95
            total_com += com
            valid_geom_slices += 1

# ==========================================
# 4. FINAL REPORTING
# ==========================================
print("\n" + "="*55)
print("GEOMETRIC PRESERVATION REPORT (BONE STRUCTURES) ")
print("="*55)
if valid_geom_slices > 0:
    print(f"Anatomical Slices Evaluated: {valid_geom_slices}")
    print(f"Skipped Empty/Air Slices:   {total_slices - valid_geom_slices}")
    print("-" * 55)
    print(f"Global Dice Overlap Score:   {total_dice / valid_geom_slices:.4f}  (Target: >0.90)")
    print(f"Global HD95 Boundary Error:  {total_hd95 / valid_geom_slices:.2f} mm  (Target: <2.0mm)")
    print(f"Global Center of Mass Shift: {total_com / valid_geom_slices:.2f} mm  (Target: <1.5mm)")
else:
    print("Error: No dense bone anatomy structures identified across the dataset.")
print("="*55)


# In[1]:


get_ipython().system('tar -czvf Processed_Task2.tar.gz Processed_Task2')


# In[ ]:


import os
import torch
import numpy as np
import SimpleITK as sitk
from torch.utils.data import DataLoader
from torchvision.utils import save_image

# ==========================================================
# 1. WAKE UP THE SAVED MODEL
# ==========================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Loading U-Former architecture...")

model = UFormer().to(device)

# Load the weights from your final saved epoch
checkpoint_path = "./saved_models/uformer_epoch_12.pth"
model.load_state_dict(torch.load(checkpoint_path, map_location=device))
model.eval() # CRITICAL: Lock weights for inference
print(f"Successfully loaded weights from {checkpoint_path}")

# ==========================================================
# 2. SETUP THE EXPORT DATA LOADER
# ==========================================================
# We use your existing UFormerDataset pointing to the processed data
export_dataset = UFormerDataset(data_dir="./Processed_Task2")
export_loader = DataLoader(export_dataset, batch_size=1, shuffle=False, num_workers=0)

export_dir = "./Hospital_Clinical_Review_DICOM"
os.makedirs(export_dir, exist_ok=True)
print(f"Creating DICOM export folder at: {export_dir}")

# ==========================================================
# 3. THE CRITICAL MATH: UN-NORMALIZATION
# ==========================================================
MIN_HU = -1000.0
MAX_HU = 3000.0

def unnormalize_to_hu(tensor):
    tensor_01 = (tensor + 1.0) / 2.0
    hu_tensor = tensor_01 * (MAX_HU - MIN_HU) + MIN_HU
    return hu_tensor.cpu().numpy().astype(np.int16)

# ==========================================================
# 4. RUN INFERENCE AND EXPORT
# ==========================================================
with torch.no_grad():
    for batch_idx, (cbct, ct, mask) in enumerate(export_loader):
        
        cbct = cbct.to(device)
        ct = ct.to(device)
        
        synthetic_ct = model(cbct)
        
        slice_folder = os.path.join(export_dir, f"Slice_{batch_idx:04d}")
        os.makedirs(slice_folder, exist_ok=True)
        
        # --- A. SAVE AS PNG ---
        save_image(cbct, os.path.join(slice_folder, "1_CBCT_Input.png"), normalize=True, value_range=(-1, 1))
        save_image(synthetic_ct, os.path.join(slice_folder, "2_Synthetic_CT_Output.png"), normalize=True, value_range=(-1, 1))
        save_image(ct, os.path.join(slice_folder, "3_Real_CT_Target.png"), normalize=True, value_range=(-1, 1))
        
        # --- B. UN-NORMALIZE ---
        cbct_hu = unnormalize_to_hu(cbct.squeeze())
        synth_hu = unnormalize_to_hu(synthetic_ct.squeeze())
        ct_hu = unnormalize_to_hu(ct.squeeze())
        
        # --- C. SAVE AS DICOM ---
        def save_as_dicom(numpy_array_2d, filename):
            img3d = np.expand_dims(numpy_array_2d, axis=0)
            sitk_img = sitk.GetImageFromArray(img3d)
            writer = sitk.ImageFileWriter()
            writer.SetFileName(filename)
            writer.SetImageIO("GDCMImageIO") 
            writer.Execute(sitk_img)

        save_as_dicom(cbct_hu, os.path.join(slice_folder, "1_CBCT_Input.dcm"))
        save_as_dicom(synth_hu, os.path.join(slice_folder, "2_Synthetic_CT_Output.dcm"))
        save_as_dicom(ct_hu, os.path.join(slice_folder, "3_Real_CT_Target.dcm"))
        
        if batch_idx % 100 == 0:
            print(f"Successfully processed and saved {batch_idx} DICOM slices...")

print("\n" + "="*50)
print("✅ CLINICAL DICOM EXPORT COMPLETE!")
print(f"Files ready for hospital handover in: {export_dir}")
print("="*50)


# In[2]:


import shutil

folder_to_zip = "./Hospital_Clinical_Review_DICOM"
output_filename = "Hospital_DICOM_Export"  # It will automatically add the .zip extension

print("Zipping 38,200 files... This might take 3 to 5 minutes. Do not stop the cell!")

# This uses Python's internal tool to build the zip file
shutil.make_archive(output_filename, 'zip', folder_to_zip)

print("\n" + "="*50)
print("✅ ZIP CREATION COMPLETE!")
print("You can now download 'Hospital_DICOM_Export.zip' from the file explorer.")
print("="*50)


# In[3]:


get_ipython().system('pip install SimpleITK')


# In[14]:


import os
import torch
import numpy as np
import SimpleITK as sitk
from torch.utils.data import DataLoader

# ==========================================================
# 1. SETUP THE 3D EXPORT FOLDER
# ==========================================================
export_dir = "./Hospital_3D_Volumes"
os.makedirs(export_dir, exist_ok=True)

print(f"Creating 3D export folder at: {export_dir}")

# Lock the weights for inference
model.eval()

# Re-initialize the loader
export_dataset = UFormerDataset(data_dir="./Processed_Task2")
export_loader = DataLoader(export_dataset, batch_size=1, shuffle=False, num_workers=0)

# ==========================================================
# 2. THE CRITICAL MATH: UN-NORMALIZATION
# ==========================================================
MIN_HU = -1000.0
MAX_HU = 3000.0

def unnormalize_to_hu(tensor):
    # We completely removed the (tensor + 1.0) / 2.0 line.
    # Multiply directly to stretch the [0, 1] data back to [-1000, 3000]
    hu_tensor = tensor * (MAX_HU - MIN_HU) + MIN_HU
    
    # Return as 16-bit integers (The strict DICOM/NIfTI requirement)
    return hu_tensor.cpu().numpy().astype(np.int16)

# ==========================================================
# 3. THE 3D STACKING ENGINE
# ==========================================================
current_patient_id = None
patient_cbct_slices = []
patient_synth_slices = []
patient_ct_slices = []

def save_3d_volume(patient_id, cbct_list, synth_list, ct_list):
    """Takes a list of 2D arrays, stacks them Z-axis, and saves as 3D medical file."""
    cbct_3d = np.stack(cbct_list, axis=0)
    synth_3d = np.stack(synth_list, axis=0)
    ct_3d = np.stack(ct_list, axis=0)

    img_cbct = sitk.GetImageFromArray(cbct_3d)
    img_synth = sitk.GetImageFromArray(synth_3d)
    img_ct = sitk.GetImageFromArray(ct_3d)

    patient_folder = os.path.join(export_dir, patient_id)
    os.makedirs(patient_folder, exist_ok=True)

    sitk.WriteImage(img_cbct, os.path.join(patient_folder, f"{patient_id}_CBCT_Input.nii.gz"))
    sitk.WriteImage(img_synth, os.path.join(patient_folder, f"{patient_id}_Synthetic_CT_Output.nii.gz"))
    sitk.WriteImage(img_ct, os.path.join(patient_folder, f"{patient_id}_Real_CT_Target.nii.gz"))
    
    print(f"✅ Reconstructed and saved full 3D Volume for Patient: {patient_id} ({len(cbct_list)} slices)")

with torch.no_grad():
    # FIXED: Bulletproof unpacking. It just grabs the raw batch.
    for batch_idx, batch_data in enumerate(export_loader):
        
        # Safely extract exactly what we need, ignoring the rest
        cbct = batch_data[0]
        ct = batch_data[1]
        
        # Look up the exact file path to find out who this slice belongs to
        original_filepath = export_dataset.cbct_files[batch_idx]
        filename = os.path.basename(original_filepath) 
        patient_id = filename.split('_')[0] 
        
        # If we hit a NEW patient, save the PREVIOUS patient's 3D stack and clear the RAM
        if current_patient_id is not None and patient_id != current_patient_id:
            save_3d_volume(current_patient_id, patient_cbct_slices, patient_synth_slices, patient_ct_slices)
            patient_cbct_slices = []
            patient_synth_slices = []
            patient_ct_slices = []
            
        current_patient_id = patient_id
        
        # Move to GPU and generate prediction
        cbct_gpu = cbct.to(device)
        synthetic_ct = model(cbct_gpu)
        
        # Un-normalize back to Hounsfield Units (-1000 to +3000)
        cbct_hu = unnormalize_to_hu(cbct.squeeze())
        synth_hu = unnormalize_to_hu(synthetic_ct.squeeze())
        ct_hu = unnormalize_to_hu(ct.squeeze())
        
        # Add the 2D math arrays to the current patient's stack
        patient_cbct_slices.append(cbct_hu)
        patient_synth_slices.append(synth_hu)
        patient_ct_slices.append(ct_hu)

    # --- CRITICAL: Save the very last patient after the loop finishes ---
    if current_patient_id is not None and len(patient_cbct_slices) > 0:
        save_3d_volume(current_patient_id, patient_cbct_slices, patient_synth_slices, patient_ct_slices)

print("\n" + "="*60)
print("🏆 FULL 3D CLINICAL EXPORT COMPLETE!")
print(f"All 3D patient volumes are safely stored in: {export_dir}")
print("="*60)


# In[12]:


import os
import random
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt

# ==========================================================
# 1. PICK A RANDOM PATIENT
# ==========================================================
export_dir = "./Hospital_3D_Volumes"

patient_folders = [f for f in os.listdir(export_dir) if os.path.isdir(os.path.join(export_dir, f))]

if not patient_folders:
    print(f"Error: Could not find any patient folders in {export_dir}")
else:
    random_patient = random.choice(patient_folders)
    patient_path = os.path.join(export_dir, random_patient)
    
    print(f"🎲 Randomly selected Patient: {random_patient}")
    print("Loading 3D volumes and auto-scaling contrast...")

    # ==========================================================
    # 2. LOAD THE 3D .nii.gz FILES
    # ==========================================================
    cbct_file = os.path.join(patient_path, f"{random_patient}_CBCT_Input.nii.gz")
    synth_file = os.path.join(patient_path, f"{random_patient}_Synthetic_CT_Output.nii.gz")
    ct_file = os.path.join(patient_path, f"{random_patient}_Real_CT_Target.nii.gz")

    cbct_3d = sitk.GetArrayFromImage(sitk.ReadImage(cbct_file))
    synth_3d = sitk.GetArrayFromImage(sitk.ReadImage(synth_file))
    ct_3d = sitk.GetArrayFromImage(sitk.ReadImage(ct_file))

    # ==========================================================
    # 3. FIND THE EXACT MIDDLE OF THE 3D VOLUME
    # ==========================================================
    z_mid = cbct_3d.shape[0] // 2  
    y_mid = cbct_3d.shape[1] // 2  
    x_mid = cbct_3d.shape[2] // 2  

    # ==========================================================
    # 4. PLOT WITH DYNAMIC AUTO-SCALING AND COLORBARS
    # ==========================================================
    fig, axes = plt.subplots(3, 3, figsize=(16, 16))
    fig.suptitle(f"Auto-Scaled 3D View: Patient {random_patient}", fontsize=18, fontweight='bold', y=0.95)

    # Helper function to plot without limits and attach a colorbar
    def plot_slice(ax, image_slice, title):
        # We drop vmin and vmax completely! Let matplotlib figure it out.
        im = ax.imshow(image_slice, cmap='gray')
        ax.set_title(title, pad=10, fontsize=12)
        ax.axis('off')
        # Add a colorbar to see the exact Hounsfield Units
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # --- ROW 1: AXIAL (Top-Down) ---
    plot_slice(axes[0, 0], cbct_3d[z_mid, :, :], "CBCT (Input) - Axial")
    plot_slice(axes[0, 1], synth_3d[z_mid, :, :], "Synth CT (AI) - Axial")
    plot_slice(axes[0, 2], ct_3d[z_mid, :, :], "Real CT (Target) - Axial")

    # --- ROW 2: CORONAL (Front-Back) ---
    plot_slice(axes[1, 0], np.flipud(cbct_3d[:, y_mid, :]), "CBCT - Coronal")
    plot_slice(axes[1, 1], np.flipud(synth_3d[:, y_mid, :]), "Synth CT - Coronal")
    plot_slice(axes[1, 2], np.flipud(ct_3d[:, y_mid, :]), "Real CT - Coronal")

    # --- ROW 3: SAGITTAL (Side


# In[13]:


import glob
import numpy as np

# Grab the very first preprocessed CBCT slice
sample_file = glob.glob("./Processed_Task2/cbct/*.npy")[0]
arr = np.load(sample_file)

print(f"Array Shape: {arr.shape}")
print(f"Absolute Minimum Value: {arr.min()}")
print(f"Absolute Maximum Value: {arr.max()}")


# In[15]:


import os
import torch
import numpy as np
import SimpleITK as sitk
from torch.utils.data import DataLoader

# ==========================================================
# 1. SETUP A BRAND NEW FOLDER
# ==========================================================
# Changing the name so it creates a completely separate folder!
export_dir = "./Hospital_3D_Volumes_Fixed"
os.makedirs(export_dir, exist_ok=True)

print(f"📁 Creating brand new 3D export folder at: {export_dir}")

# Lock weights and initialize loader
model.eval()
export_dataset = UFormerDataset(data_dir="./Processed_Task2")
export_loader = DataLoader(export_dataset, batch_size=1, shuffle=False, num_workers=0)

# ==========================================================
# 2. THE CORRECTED MATH [0, 1] -> [-1000, 3000] HU
# ==========================================================
MIN_HU = -1000.0
MAX_HU = 3000.0

def unnormalize_to_hu(tensor):
    # No shift! Directly scales your 0-1 data to Hospital HU
    hu_tensor = tensor * (MAX_HU - MIN_HU) + MIN_HU
    return hu_tensor.cpu().numpy().astype(np.int16)

# ==========================================================
# 3. THE 3D STACKING ENGINE
# ==========================================================
current_patient_id = None
patient_cbct_slices = []
patient_synth_slices = []
patient_ct_slices = []

def save_3d_volume(patient_id, cbct_list, synth_list, ct_list):
    cbct_3d = np.stack(cbct_list, axis=0)
    synth_3d = np.stack(synth_list, axis=0)
    ct_3d = np.stack(ct_list, axis=0)

    img_cbct = sitk.GetImageFromArray(cbct_3d)
    img_synth = sitk.GetImageFromArray(synth_3d)
    img_ct = sitk.GetImageFromArray(ct_3d)

    patient_folder = os.path.join(export_dir, patient_id)
    os.makedirs(patient_folder, exist_ok=True)

    sitk.WriteImage(img_cbct, os.path.join(patient_folder, f"{patient_id}_CBCT_Input.nii.gz"))
    sitk.WriteImage(img_synth, os.path.join(patient_folder, f"{patient_id}_Synthetic_CT_Output.nii.gz"))
    sitk.WriteImage(img_ct, os.path.join(patient_folder, f"{patient_id}_Real_CT_Target.nii.gz"))
    
    print(f"✅ Saved clean 3D Volume for Patient: {patient_id} ({len(cbct_list)} slices)")

with torch.no_grad():
    for batch_idx, batch_data in enumerate(export_loader):
        cbct = batch_data[0]
        ct = batch_data[1]
        
        original_filepath = export_dataset.cbct_files[batch_idx]
        filename = os.path.basename(original_filepath) 
        patient_id = filename.split('_')[0] 
        
        if current_patient_id is not None and patient_id != current_patient_id:
            save_3d_volume(current_patient_id, patient_cbct_slices, patient_synth_slices, patient_ct_slices)
            patient_cbct_slices = []
            patient_synth_slices = []
            patient_ct_slices = []
            
        current_patient_id = patient_id
        
        cbct_gpu = cbct.to(device)
        synthetic_ct = model(cbct_gpu)
        
        cbct_hu = unnormalize_to_hu(cbct.squeeze())
        synth_hu = unnormalize_to_hu(synthetic_ct.squeeze())
        ct_hu = unnormalize_to_hu(ct.squeeze())
        
        patient_cbct_slices.append(cbct_hu)
        patient_synth_slices.append(synth_hu)
        patient_ct_slices.append(ct_hu)

    if current_patient_id is not None and len(patient_cbct_slices) > 0:
        save_3d_volume(current_patient_id, patient_cbct_slices, patient_synth_slices, patient_ct_slices)

print("\n" + "="*60)
print(f"🏆 NEW EXPORT COMPLETE! Files saved to {export_dir}")
print("="*60)


# In[2]:


get_ipython().system('pip install SimpleITK scikit-image numpy')


# In[4]:


get_ipython().system('pip install --upgrade numpy scipy scikit-image')


# In[1]:


import os
import numpy as np
import SimpleITK as sitk
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

export_dir = "./Hospital_3D_Volumes_Fixed"

# Lists to store the scores for all 180 patients
all_mae = []
all_psnr = []
all_ssim = []

print("Starting 3D Clinical Evaluation...")
print("-" * 50)

# Find all patient folders
patient_folders = [f for f in os.listdir(export_dir) if os.path.isdir(os.path.join(export_dir, f))]

for patient_id in patient_folders:
    patient_path = os.path.join(export_dir, patient_id)
    
    # Define exact file paths for the perfectly matched pair
    synth_file = os.path.join(patient_path, f"{patient_id}_Synthetic_CT_Output.nii.gz")
    target_file = os.path.join(patient_path, f"{patient_id}_Real_CT_Target.nii.gz")
    
    if not os.path.exists(synth_file) or not os.path.exists(target_file):
        continue

    # Load the 3D volumes into numpy arrays
    synth_3d = sitk.GetArrayFromImage(sitk.ReadImage(synth_file)).astype(np.float32)
    target_3d = sitk.GetArrayFromImage(sitk.ReadImage(target_file)).astype(np.float32)

    # 1. Calculate Mean Absolute Error (MAE) in Hounsfield Units
    mae = np.mean(np.abs(target_3d - synth_3d))
    
    # 2. Calculate PSNR
    # Data range is 4000 because we scale from -1000 to 3000
    p_score = psnr(target_3d, synth_3d, data_range=4000)
    
    # 3. Calculate SSIM (Structural Similarity)
    # We use a window size of 7 for 3D medical images
    s_score = ssim(target_3d, synth_3d, data_range=4000)
    
    all_mae.append(mae)
    all_psnr.append(p_score)
    all_ssim.append(s_score)

# ==========================================================
# FINAL AVERAGE SCORES FOR THE THESIS
# ==========================================================
# print("\n" + "="*50)
# print("🏆 FINAL 3D METRICS (APPLES-TO-APPLES)")
# print("="*50)
# print(f"Total Patients Evaluated: {len(all_mae)}")
# print(f"Average MAE  : {np.mean(all_mae):.2f} HU")
# print(f"Average PSNR : {np.mean(all_psnr):.2f} dB")
# print(f"Average SSIM : {np.mean(all_ssim):.4f}")
# print("="*50)


# In[ ]:




