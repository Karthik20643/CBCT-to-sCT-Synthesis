# Create a new folder to hold the extracted data safely
get_ipython().system('mkdir -p SynthRAD2025_Task2_Train_Data')

# Unzip quietly (-q) to prevent browser freezing, directing output to the new folder (-d)
get_ipython().system('unzip -q synthRAD2025_Task2_Train.zip -d SynthRAD2025_Task2_Train_Data')

print("Extraction of SynthRAD 2025 dataset is complete!")



import os

# The exact path from your screenshot
hn_dir = 'SynthRAD2025_Task2_Train_Data/synthRAD2025_Task2_Train/Task2/HN'

# Get a sorted list of patient folders
patient_folders = sorted([f for f in os.listdir(hn_dir) if os.path.isdir(os.path.join(hn_dir, f))])

print(f"Total HN Patients Found: {len(patient_folders)}\n")

# Peek inside the first two patient folders
print("--- 3D File Structure Preview ---")
for patient in patient_folders[:2]:
    patient_path = os.path.join(hn_dir, patient)
    files = sorted(os.listdir(patient_path))
    print(f" {patient}")
    for f in files:
        print(f"  └──  {f}")



get_ipython().system('pip install SimpleITK')



import os
import torch
from torch.utils.data import Dataset
import SimpleITK as sitk
import numpy as np

class SynthRAD3D(Dataset):
    def __init__(self, data_dir, patient_folders):
        """
        Custom PyTorch 3D Dataset for SynthRAD 2025 Head & Neck
        """
        self.data_dir = data_dir
        self.patient_folders = patient_folders

    def __len__(self):
        # Tells PyTorch we have 261 patients
        return len(self.patient_folders)

    def __getitem__(self, idx):
        # 1. Get the specific patient folder for this batch
        patient_id = self.patient_folders[idx]
        patient_path = os.path.join(self.data_dir, patient_id)
        
        # 2. Map the exact filenames we found in our reconnaissance
        cbct_path = os.path.join(patient_path, 'cbct.mha')
        ct_path = os.path.join(patient_path, 'ct.mha')
        mask_path = os.path.join(patient_path, 'mask.mha')
        
        # 3. Load the 3D medical images using SimpleITK
        cbct_img = sitk.ReadImage(cbct_path)
        ct_img = sitk.ReadImage(ct_path)
        mask_img = sitk.ReadImage(mask_path)
        
        # 4. Convert the ITK images to standard NumPy arrays
        # sitk returns shapes as (Depth, Height, Width) -> (D, H, W)
        cbct_array = sitk.GetArrayFromImage(cbct_img).astype(np.float32)
        ct_array = sitk.GetArrayFromImage(ct_img).astype(np.float32)
        mask_array = sitk.GetArrayFromImage(mask_img).astype(np.float32)
        
        # --- PREPROCESSING GOES HERE ---
        # (e.g., HU Clipping, Normalization, Mask Application)
        
        # 5. Convert to PyTorch Tensors and add a Channel dimension (C, D, H, W)
        cbct_tensor = torch.from_numpy(cbct_array).unsqueeze(0)
        ct_tensor = torch.from_numpy(ct_array).unsqueeze(0)
        
        return cbct_tensor, ct_tensor

# --- Quick Test ---
if __name__ == "__main__":
    hn_dir = 'SynthRAD2025_Task2_Train_Data/synthRAD2025_Task2_Train/Task2/HN'
    patient_list = sorted([f for f in os.listdir(hn_dir) if os.path.isdir(os.path.join(hn_dir, f))])
    
    # Initialize the dataset
    train_dataset = SynthRAD3D(data_dir=hn_dir, patient_folders=patient_list)
    
    # Load the very first patient to verify the tensor shapes
    sample_cbct, sample_ct = train_dataset[0]
    print(f"Successfully loaded Patient 1!")
    print(f"CBCT Tensor Shape: {sample_cbct.shape}")



def __getitem__(self, idx):
        # 1. Get the specific patient folder
        patient_id = self.patient_folders[idx]
        patient_path = os.path.join(self.data_dir, patient_id)
        
        # 2. Load the 3D medical images using SimpleITK
        cbct_img = sitk.ReadImage(os.path.join(patient_path, 'cbct.mha'))
        ct_img = sitk.ReadImage(os.path.join(patient_path, 'ct.mha'))
        mask_img = sitk.ReadImage(os.path.join(patient_path, 'mask.mha'))
        
        # Convert ITK images to standard NumPy arrays (D, H, W)
        cbct_array = sitk.GetArrayFromImage(cbct_img).astype(np.float32)
        ct_array = sitk.GetArrayFromImage(ct_img).astype(np.float32)
        mask_array = sitk.GetArrayFromImage(mask_img).astype(np.float32)
        
        # ==========================================
        # STEP 1: HU Clipping & Normalization
        # ==========================================
        # Clinical window: -1000 to +1000 HU
        MIN_HU = -1000.0
        MAX_HU = 1000.0
        
        cbct_array = np.clip(cbct_array, MIN_HU, MAX_HU)
        ct_array = np.clip(ct_array, MIN_HU, MAX_HU)
        
        # Normalize to [0.0, 1.0]
        cbct_array = (cbct_array - MIN_HU) / (MAX_HU - MIN_HU)
        ct_array = (ct_array - MIN_HU) / (MAX_HU - MIN_HU)
        
        # ==========================================
        # STEP 2: Background Masking
        # ==========================================
        # Force all pixels outside the patient's body to absolute 0
        cbct_array[mask_array == 0] = 0.0
        ct_array[mask_array == 0] = 0.0
        
        # ==========================================
        # STEP 3: Mask-Aware 3D Patch Cropping
        # ==========================================
        patch_size = 64
        D, H, W = cbct_array.shape
        
        # Find all voxel coordinates where the mask is valid (tissue exists)
        valid_coords = np.argwhere(mask_array > 0)
        
        if len(valid_coords) > 0:
            # Pick a random valid tissue voxel to act as our patch center
            center_idx = np.random.randint(0, len(valid_coords))
            center_z, center_y, center_x = valid_coords[center_idx]
            
            # Calculate boundaries ensuring we don't go outside the image array
            z_start = max(0, min(center_z - patch_size // 2, D - patch_size))
            y_start = max(0, min(center_y - patch_size // 2, H - patch_size))
            x_start = max(0, min(center_x - patch_size // 2, W - patch_size))
        else:
            # Fallback if mask is empty (rare safety catch)
            z_start = np.random.randint(0, D - patch_size)
            y_start = np.random.randint(0, H - patch_size)
            x_start = np.random.randint(0, W - patch_size)
            
        # Crop the 64x64x64 patches
        cbct_patch = cbct_array[z_start : z_start + patch_size, 
                                y_start : y_start + patch_size, 
                                x_start : x_start + patch_size]
                                
        ct_patch = ct_array[z_start : z_start + patch_size, 
                            y_start : y_start + patch_size, 
                            x_start : x_start + patch_size]
        
        # Convert to PyTorch Tensors and add Channel dimension (1, 64, 64, 64)
        cbct_tensor = torch.from_numpy(cbct_patch).unsqueeze(0)
        ct_tensor = torch.from_numpy(ct_patch).unsqueeze(0)
        
        return cbct_tensor, ct_tensor



import torch
import torch.nn as nn

def window_partition_3d(x, window_size):
    """
    Chops a 3D volume into non-overlapping 3D windows.
    Input shape: (Batch, Depth, Height, Width, Channels)
    """
    B, D, H, W, C = x.shape
    x = x.view(B, D // window_size, window_size, 
                  H // window_size, window_size, 
                  W // window_size, window_size, C)
    
    # Rearrange the blocks and flatten the spatial dimensions into a sequence of tokens
    windows = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous()
    windows = windows.view(-1, window_size * window_size * window_size, C)
    return windows

def window_reverse_3d(windows, window_size, D, H, W):
    """
    Stitches the 3D windows back into the full 3D volume.
    """
    B = int(windows.shape[0] / (D * H * W / window_size / window_size / window_size))
    x = windows.view(B, D // window_size, H // window_size, W // window_size, 
                     window_size, window_size, window_size, -1)
    
    x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous()
    x = x.view(B, D, H, W, -1)
    return x


class WindowAttention3D(nn.Module):
    def __init__(self, dim, window_size=8, num_heads=4):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        # The Q, K, V linear projection layer
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        # x comes in as a 1D sequence of the entire 64x64x64 patch: (Batch, Tokens, Channels)
        B, N, C = x.shape
        
        # Calculate the physical 3D dimensions of the current grid (e.g., 64)
        grid_size = int(round(N**(1/3)))
        D = H = W = grid_size
        
        # 1. Un-flatten into a 3D volume so we can chop it up
        x = x.view(B, D, H, W, C)
        
        # 2. Chop into tiny 8x8x8 windows
        # New shape: (Batch * num_windows, 512 tokens, Channels)
        x_windows = window_partition_3d(x, self.window_size)
        
        # 3. Create Queries, Keys, and Values
        B_, N_, C_ = x_windows.shape
        qkv = self.qkv(x_windows).reshape(B_, N_, 3, self.num_heads, C_ // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # 4. The Core Attention Math: (Q * K^T) / sqrt(d)
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        attn = self.softmax(attn)
        
        # 5. Multiply by Values
        x_attn = (attn @ v).transpose(1, 2).reshape(B_, N_, C_)
        
        # 6. Stitch the 8x8x8 windows back into the full 64x64x64 volume
        x_merged = window_reverse_3d(x_attn, self.window_size, D, H, W)
        
        # 7. Flatten back into the sequence format for the next layer
        x_out = x_merged.view(B, D * H * W, C)
        x_out = self.proj(x_out)
        
        return x_out



import torch.nn as nn

class LeWinBlock3D(nn.Module):
    def __init__(self, dim, window_size=8, num_heads=4):
        """
        The core 3D Transformer block for the U-Former.
        Combines Global 3D Window Attention with Local 3D Feature Extraction.
        """
        super().__init__()
        
        # 1. First Normalization + Global Attention
        self.norm1 = nn.LayerNorm(dim)
        self.attention = WindowAttention3D(dim, window_size, num_heads)
        
        # 2. Second Normalization + Local Edge Detection
        self.norm2 = nn.LayerNorm(dim)
        self.leff = LeFF3D(dim=dim, hidden_dim=dim * 4)

    def forward(self, x):
        # x expects shape: (Batch, Tokens, Channels)
        
        # Phase 1: Global context (with skip connection)
        x = x + self.attention(self.norm1(x))
        
        # Phase 2: Local edges (with skip connection)
        x = x + self.leff(self.norm2(x))
        
        return x



import torch.nn as nn

class Downsample3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # A 4x4x4 kernel with stride 2 halves the spatial dimensions (e.g., 64 -> 32)
        # and we tell it to double the feature channels
        self.proj = nn.Conv3d(
            in_channels, 
            out_channels, 
            kernel_size=4, 
            stride=2, 
            padding=1
        )

    def forward(self, x):
        # x comes in as (Batch, Tokens, Channels)
        B, N, C = x.shape
        
        # Calculate current 3D grid size
        grid_size = int(round(N**(1/3)))
        D = H = W = grid_size
        
        # 1. Un-flatten to 3D volume: (Batch, Channels, Depth, Height, Width)
        x = x.transpose(1, 2).view(B, C, D, H, W)
        
        # 2. Shrink the 3D volume
        x = self.proj(x)
        
        # 3. Flatten back to sequence for the next Transformer block
        x = x.flatten(2).transpose(1, 2)
        
        return x



class Upsample3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # ConvTranspose3d with stride 2 doubles the spatial dimensions (e.g., 32 -> 64)
        # and we tell it to halve the feature channels
        self.proj = nn.ConvTranspose3d(
            in_channels, 
            out_channels, 
            kernel_size=2, 
            stride=2
        )

    def forward(self, x):
        # x comes in as (Batch, Tokens, Channels)
        B, N, C = x.shape
        
        grid_size = int(round(N**(1/3)))
        D = H = W = grid_size
        
        # 1. Un-flatten to 3D volume
        x = x.transpose(1, 2).view(B, C, D, H, W)
        
        # 2. Expand the 3D volume
        x = self.proj(x)
        
        # 3. Flatten back to sequence
        x = x.flatten(2).transpose(1, 2)
        
        return x



import torch
import torch.nn as nn

class UFormer3D(nn.Module):
    def __init__(self, in_ch=1, out_ch=1, base_dim=32):
        super().__init__()
        
        # 0. Initial 3D Projection: Maps 1 grayscale channel (CBCT) to 32 feature channels
        self.input_proj = nn.Conv3d(in_ch, base_dim, kernel_size=3, padding=1)

        # ==========================================
        # THE ENCODER (Extracts features, shrinks spatial size)
        # ==========================================
        self.enc1 = LeWinBlock3D(dim=base_dim, num_heads=1)
        self.down1 = Downsample3D(in_channels=base_dim, out_channels=base_dim * 2)

        self.enc2 = LeWinBlock3D(dim=base_dim * 2, num_heads=2)
        self.down2 = Downsample3D(in_channels=base_dim * 2, out_channels=base_dim * 4)

        # ==========================================
        # THE BOTTLENECK (Deepest features, pure global context)
        # ==========================================
        self.bottleneck = LeWinBlock3D(dim=base_dim * 4, num_heads=4)

        # ==========================================
        # THE DECODER (Rebuilds the synthetic CT)
        # ==========================================
        self.up1 = Upsample3D(in_channels=base_dim * 4, out_channels=base_dim * 2)
        self.dec1 = LeWinBlock3D(dim=base_dim * 4, num_heads=2)

        self.up2 = Upsample3D(in_channels=base_dim * 2, out_channels=base_dim)
        self.dec2 = LeWinBlock3D(dim=base_dim * 2, num_heads=1)

        # Final 3D Projection back to 1 grayscale synthetic CT channel
        self.output_proj = nn.Conv3d(base_dim * 2, out_ch, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.input_proj(x)
        B, C, D, H, W = x.shape
        
        x = x.flatten(2).transpose(1, 2)

        # --- ENCODER PASS ---
        skip1 = self.enc1(x)         
        x = self.down1(skip1)

        skip2 = self.enc2(x)         
        x = self.down2(skip2)

        # --- BOTTLENECK PASS ---
        x = self.bottleneck(x)

        # --- DECODER PASS (With Skip Connections) ---
        x = self.up1(x)
        x = torch.cat([x, skip2], dim=-1)  
        x = self.dec1(x)

        x = self.up2(x)
        x = torch.cat([x, skip1], dim=-1)
        x = self.dec2(x)

        # --- OUTPUT PASS ---
        x = x.transpose(1, 2).view(B, -1, D, H, W)
        x = self.output_proj(x)
        
        return x



import torch
import torch.nn as nn

class TissueWeightedLoss3D(nn.Module):
    def __init__(self, air_weight=0.1, tissue_weight=1.0, bone_weight=3.0):
        super().__init__()
        # The penalty multipliers (The "Three Sticks")
        self.air_weight = air_weight
        self.tissue_weight = tissue_weight
        self.bone_weight = bone_weight
        
        # Standard baseline error
        self.l1_criterion = nn.L1Loss(reduction='none')

    def get_3d_edges(self, x):
        # The "Magic Glasses" (3D Sobel Filter approximation)
        # Calculates the sharpness of the borders along Depth, Height, and Width
        dz = torch.abs(x[:, :, 1:, :, :] - x[:, :, :-1, :, :])
        dy = torch.abs(x[:, :, :, 1:, :] - x[:, :, :, :-1, :])
        dx = torch.abs(x[:, :, :, :, 1:] - x[:, :, :, :, :-1])
        return dx, dy, dz

    def forward(self, pred, target):
        # Step 1: Calculate the raw, unweighted error
        raw_error = self.l1_criterion(pred, target)

        # Step 2: Find the anatomy based on Hounsfield Units!
        air_mask = (target < -500)
        soft_tissue_mask = (target >= -500) & (target <= 300)
        bone_tumor_mask = (target > 300)

        # Step 3: Apply the specific weights to the specific regions
        weights = torch.zeros_like(target)
        weights[air_mask] = self.air_weight
        weights[soft_tissue_mask] = self.tissue_weight
        weights[bone_tumor_mask] = self.bone_weight

        # Step 4: Calculate the Tissue-Aware Loss
        weighted_tissue_loss = torch.mean(raw_error * weights)

        # Step 5: Calculate the Edge Sharpness Loss
        pred_dx, pred_dy, pred_dz = self.get_3d_edges(pred)
        targ_dx, targ_dy, targ_dz = self.get_3d_edges(target)
        
        edge_loss = (torch.mean(torch.abs(pred_dx - targ_dx)) + 
                     torch.mean(torch.abs(pred_dy - targ_dy)) + 
                     torch.mean(torch.abs(pred_dz - targ_dz))) / 3.0

        # Step 6: The Final Grade (80% Tissue Accuracy, 20% Edge Sharpness)
        total_loss = (0.8 * weighted_tissue_loss) + (0.2 * edge_loss)
        
        return total_loss

import torch.optim as optim
from torch.utils.data import DataLoader

# ==========================================
# 1. SETUP THE ENVIRONMENT
# ==========================================
# Pointing directly to the Task2 folder in your current Jupyter directory
data_path = "./Task2"

print("Waking up the Data Robot...")
# Setting target_size to 128x128x128 to prevent GPU Out of Memory errors
train_dataset = SynthRadDataset3D(root_dir=data_path, target_size=(128, 128, 128))
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)

print("Waking up the 3D U-Former Student...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UFormer3D().to(device)

print("Hiring the Smart Teacher (Anatomy-Aware Loss)...")
criterion = TissueWeightedLoss3D().to(device)

# The Engine that updates the U-Former's brain
optimizer = optim.AdamW(model.parameters(), lr=1e-4)

# ==========================================
# 2. THE MASTER TRAINING LOOP
# ==========================================
epochs = 50  # Number of full sweeps through the dataset

print(f"Starting Training on Device: {device}")
print("-" * 30)

for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    
    for batch_idx, (cbct, ct) in enumerate(train_loader):
        # Move the massive 3D volumes to the GPU
        cbct = cbct.to(device)
        ct = ct.to(device)
        
        # Zero out the old gradients
        optimizer.zero_grad()
        
        # Forward Pass: Student guesses the synthetic CT
        synthetic_ct = model(cbct)
        
        # Teacher Grades the Student using the 3D Anatomy Loss
        loss = criterion(synthetic_ct, ct)
        
        # Backward Pass: The Student learns from the mistakes
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        print(f"Epoch [{epoch+1}/{epochs}] | Patient [{batch_idx+1}/{len(train_loader)}] | Loss: {loss.item():.4f}")
        
    # Print average loss for the epoch
    avg_loss = running_loss / len(train_loader)
    print(f"===> Epoch {epoch+1} Complete | Average Loss: {avg_loss:.4f}")
    
    # Save the model's brain periodically so you don't lose progress!
    if (epoch + 1) % 5 == 0:
        torch.save(model.state_dict(), f"uformer3d_epoch_{epoch+1}.pth")
        print(f"Model saved at epoch {epoch+1}!")





