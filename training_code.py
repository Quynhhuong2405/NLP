"""
==============================================================================
PHẦN 3: TRAINING PROCESS (Có Early Stopping)
==============================================================================
Tuân thủ yêu cầu đồ án:
- Early Stopping sau 3 epoch không cải thiện
- Teacher Forcing ratio = 0.5
- Gradient Clipping
- Checkpoint best model
==============================================================================
"""

import time
import math
import torch
import torch.nn as nn
from tqdm import tqdm

# ==============================================================================
# 1. CẤU HÌNH HYPERPARAMETERS
# ==============================================================================
N_EPOCHS = 20           # Số epoch tối đa
CLIP = 1.0              # Gradient clipping
LEARNING_RATE = 0.001   # Learning rate
PATIENCE = 3            # Early Stopping: dừng sau N epoch không cải thiện
TEACHER_FORCING_RATIO = 0.5  # Tỷ lệ Teacher Forcing

# ==============================================================================
# 2. HÀM EPOCH_TIME (Helper)
# ==============================================================================
def epoch_time(start_time, end_time):
    """Tính thời gian chạy 1 epoch (phút, giây)."""
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


# ==============================================================================
# 3. HÀM TRAIN (1 Epoch)
# ==============================================================================
def train(model, iterator, optimizer, criterion, clip, device, teacher_forcing_ratio=0.5):
    """
    Huấn luyện model trong 1 epoch.
    
    Args:
        model: Mô hình Seq2Seq
        iterator: DataLoader train
        optimizer: Adam optimizer
        criterion: CrossEntropyLoss
        clip: Gradient clipping value
        device: 'cuda' hoặc 'cpu'
        teacher_forcing_ratio: Tỷ lệ sử dụng Teacher Forcing (0.5)
        
    Returns:
        epoch_loss: Loss trung bình của epoch
    """
    model.train()
    epoch_loss = 0
    
    progress_bar = tqdm(iterator, desc="Training", leave=False)
    
    for batch in progress_bar:
        # ===== 1. UNPACK BATCH =====
        # collate_fn trả về: (src, trg, src_len)
        src, trg, src_len = batch
        
        # Chuyển src, trg lên device
        src = src.to(device)         # [src_len, batch_size]
        trg = trg.to(device)         # [trg_len, batch_size]
        # ⚠️ src_len PHẢI nằm trên CPU cho pack_padded_sequence
        # KHÔNG gọi src_len.to(device)!
        
        # ===== 2. FORWARD PASS =====
        optimizer.zero_grad()
        
        # Forward với teacher_forcing_ratio
        # output shape: [trg_len, batch_size, output_dim]
        output = model(src, src_len, trg, teacher_forcing_ratio)
        
        # ===== 3. TÍNH LOSS =====
        """
        📌 LOGIC SLICING (QUAN TRỌNG):
        - output[0] là zeros tensor (do loop bắt đầu từ t=1)
        - trg[0] là <sos> token
        - Phải bỏ cả hai trước khi tính loss
        
        Sau khi slice:
        - output: [trg_len-1, batch_size, output_dim]
        - trg:    [trg_len-1, batch_size]
        """
        output_dim = output.shape[-1]
        
        # Bỏ timestep đầu tiên
        output = output[1:]   # [trg_len-1, batch_size, output_dim]
        trg = trg[1:]         # [trg_len-1, batch_size]
        
        # Reshape về 2D cho CrossEntropyLoss
        output = output.reshape(-1, output_dim)  # [(trg_len-1)*batch_size, output_dim]
        trg = trg.reshape(-1)                    # [(trg_len-1)*batch_size]
        
        # Tính loss
        loss = criterion(output, trg)
        
        # ===== 4. BACKWARD PASS =====
        loss.backward()
        
        # Clip gradient để tránh exploding gradient
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        
        # Update weights
        optimizer.step()
        
        epoch_loss += loss.item()
        progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    return epoch_loss / len(iterator)


# ==============================================================================
# 4. HÀM EVALUATE
# ==============================================================================
def evaluate(model, iterator, criterion, device):
    """
    Đánh giá model trên tập validation/test.
    
    Args:
        model: Mô hình Seq2Seq
        iterator: DataLoader val/test
        criterion: CrossEntropyLoss
        device: 'cuda' hoặc 'cpu'
        
    Returns:
        epoch_loss: Loss trung bình
    """
    model.eval()
    epoch_loss = 0
    
    with torch.no_grad():
        for batch in tqdm(iterator, desc="Evaluating", leave=False):
            # Unpack batch
            src, trg, src_len = batch
            
            src = src.to(device)
            trg = trg.to(device)
            # src_len giữ nguyên trên CPU
            
            # Forward với teacher_forcing_ratio = 0 (không dùng ground truth)
            output = model(src, src_len, trg, teacher_forcing_ratio=0)
            
            # Tính loss (y hệt hàm train)
            output_dim = output.shape[-1]
            
            output = output[1:]
            trg = trg[1:]
            
            output = output.reshape(-1, output_dim)
            trg = trg.reshape(-1)
            
            loss = criterion(output, trg)
            epoch_loss += loss.item()
    
    return epoch_loss / len(iterator)


# ==============================================================================
# 5. KHỞI TẠO OPTIMIZER & CRITERION
# ==============================================================================

# Optimizer: Adam với lr = 0.001
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Loss function: CrossEntropyLoss với ignore_index để bỏ qua PAD token
criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

# Áp dụng weight initialization
model.apply(init_weights)

# Đếm số tham số
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print("=" * 60)
print(" CẤU HÌNH HUẤN LUYỆN")
print("=" * 60)
print(f"Device:              {device}")
print(f"Total Parameters:    {total_params:,}")
print(f"Epochs:              {N_EPOCHS}")
print(f"Learning Rate:       {LEARNING_RATE}")
print(f"Gradient Clip:       {CLIP}")
print(f"Teacher Forcing:     {TEACHER_FORCING_RATIO}")
print(f"Early Stopping:      Patience = {PATIENCE}")
print(f"Batch Size:          {BATCH_SIZE}")
print("=" * 60)


# ==============================================================================
# 6. VÒNG LẶP HUẤN LUYỆN CHÍNH (VỚI EARLY STOPPING)
# ==============================================================================

# Biến theo dõi
best_valid_loss = float('inf')
epochs_without_improvement = 0
training_history = {
    'train_loss': [],
    'valid_loss': [],
    'train_ppl': [],
    'valid_ppl': []
}

print("\n" + "=" * 60)
print(" BẮT ĐẦU HUẤN LUYỆN")
print("=" * 60 + "\n")

for epoch in range(N_EPOCHS):
    
    start_time = time.time()
    
    # ===== TRAIN =====
    train_loss = train(
        model, train_loader, optimizer, criterion, 
        CLIP, device, TEACHER_FORCING_RATIO
    )
    
    # ===== EVALUATE =====
    valid_loss = evaluate(model, valid_loader, criterion, device)
    
    end_time = time.time()
    epoch_mins, epoch_secs = epoch_time(start_time, end_time)
    
    # ===== TÍNH PERPLEXITY =====
    train_ppl = math.exp(train_loss)
    valid_ppl = math.exp(valid_loss)
    
    # Lưu history
    training_history['train_loss'].append(train_loss)
    training_history['valid_loss'].append(valid_loss)
    training_history['train_ppl'].append(train_ppl)
    training_history['valid_ppl'].append(valid_ppl)
    
    # ===== CHECKPOINTING & EARLY STOPPING =====
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        epochs_without_improvement = 0
        
        # Lưu best model
        torch.save(model.state_dict(), 'best_model.pth')
        save_status = "✅ Model saved!"
    else:
        epochs_without_improvement += 1
        save_status = f"⚠️ No improvement ({epochs_without_improvement}/{PATIENCE})"
    
    # ===== IN KẾT QUẢ =====
    print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {train_ppl:7.3f}')
    print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {valid_ppl:7.3f}')
    print(f'\t{save_status}')
    print("-" * 60)
    
    # ===== KIỂM TRA EARLY STOPPING =====
    if epochs_without_improvement >= PATIENCE:
        print("\n" + "=" * 60)
        print(f"⛔ EARLY STOPPING: Val loss không giảm sau {PATIENCE} epochs")
        print("=" * 60)
        break


# ==============================================================================
# 7. TỔNG KẾT HUẤN LUYỆN
# ==============================================================================
print("\n" + "=" * 60)
print(" HUẤN LUYỆN HOÀN TẤT!")
print("=" * 60)
print(f"Epochs đã chạy:      {epoch + 1}")
print(f"Best Validation Loss: {best_valid_loss:.3f}")
print(f"Best Validation PPL:  {math.exp(best_valid_loss):.3f}")
print(f"Model đã lưu tại:     'best_model.pth'")
print("=" * 60)


# ==============================================================================
# 8. ĐÁNH GIÁ TRÊN TẬP TEST
# ==============================================================================

# Load best model
model.load_state_dict(torch.load('best_model.pth', map_location=device))

# Đánh giá trên test
test_loss = evaluate(model, test_loader, criterion, device)

print("\n" + "=" * 60)
print(" KẾT QUẢ TRÊN TẬP TEST")
print("=" * 60)
print(f'Test Loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):7.3f}')
print("=" * 60)


# ==============================================================================
# 9. VẼ BIỂU ĐỒ TRAINING HISTORY (Optional)
# ==============================================================================
"""
# Uncomment để vẽ biểu đồ

import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot Loss
axes[0].plot(training_history['train_loss'], label='Train Loss', marker='o')
axes[0].plot(training_history['valid_loss'], label='Valid Loss', marker='s')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].set_title('Training & Validation Loss')
axes[0].legend()
axes[0].grid(True)

# Plot Perplexity
axes[1].plot(training_history['train_ppl'], label='Train PPL', marker='o')
axes[1].plot(training_history['valid_ppl'], label='Valid PPL', marker='s')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Perplexity')
axes[1].set_title('Training & Validation Perplexity')
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.savefig('training_history.png', dpi=150)
plt.show()

print("✅ Đã lưu biểu đồ tại 'training_history.png'")
"""
