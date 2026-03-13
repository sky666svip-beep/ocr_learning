import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms

from utils.data_generator import SemanticTextDataset
from models.v4_transformer_joint import V4JointModel

# --- 双轨字典构建 ---
dummy_ds = SemanticTextDataset(num_samples=1)
CHAR_LIST = dummy_ds.char_list

# Attention branch dictionary (带有 SOS, EOS, PAD, 共 15 + 3 = 18 类)
PAD_TOKEN = 0
SOS_TOKEN = 1
EOS_TOKEN = 2
char2id_attn = {c: i + 3 for i, c in enumerate(CHAR_LIST)}
id2char_attn = {i + 3: c for i, c in enumerate(CHAR_LIST)}
id2char_attn[PAD_TOKEN] = '<PAD>'
id2char_attn[SOS_TOKEN] = '<SOS>'
id2char_attn[EOS_TOKEN] = '<EOS>'
VOCAB_SIZE_ATTN = len(char2id_attn) + 3

# CTC branch dictionary (带有 Blank = 0, 共 15 + 1 = 16 类)
char2id_ctc = {c: i + 1 for i, c in enumerate(CHAR_LIST)}
id2char_ctc = {i + 1: c for i, c in enumerate(CHAR_LIST)}
id2char_ctc[0] = '<Blank>'
VOCAB_SIZE_CTC = len(char2id_ctc) + 1

def text_to_tensor_attn(text, max_len=15):
    seq = [SOS_TOKEN]
    for c in text:
        if c in char2id_attn:
            seq.append(char2id_attn[c])
    seq.append(EOS_TOKEN)
    if len(seq) < max_len:
        seq.extend([PAD_TOKEN] * (max_len - len(seq)))
    else:
        seq = seq[:max_len]
        seq[-1] = EOS_TOKEN
    return torch.tensor(seq, dtype=torch.long)

def collate_fn_joint(batch):
    images, texts = zip(*batch)
    images = torch.stack(images, 0)
    
    attn_targets = []
    ctc_targets = []
    ctc_target_lengths = []
    
    for text in texts:
        # Attention target
        attn_targets.append(text_to_tensor_attn(text))
        
        # CTC target
        target_ctc = [char2id_ctc[c] for c in text if c in char2id_ctc]
        ctc_targets.extend(target_ctc)
        ctc_target_lengths.append(len(target_ctc))
        
    attn_targets = torch.stack(attn_targets, 0)
    ctc_targets = torch.tensor(ctc_targets, dtype=torch.long)
    ctc_target_lengths = torch.tensor(ctc_target_lengths, dtype=torch.long)
    
    return images, texts, attn_targets, ctc_targets, ctc_target_lengths

def train_v4_joint_model(epochs=10, batch_size=32, learning_rate=0.001, save_path="models/best_v4_joint.pth",
                         progress_callback=None, metric_callback=None, lambda_ctc=0.2, nhead=2, num_layers=1, use_stn=False):
    
    train_transform = transforms.Compose([
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: 1.0 - x), 
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: 1.0 - x),
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_samples = 4000
    test_samples = 500
    train_dataset = SemanticTextDataset(num_samples=train_samples, transform=train_transform)
    test_dataset = SemanticTextDataset(num_samples=test_samples, transform=test_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn_joint)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn_joint)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    enable_amp = torch.cuda.is_available()
    scaler = torch.cuda.amp.GradScaler(enabled=enable_amp)
    
    model = V4JointModel(output_dim_attn=VOCAB_SIZE_ATTN, 
                         output_dim_ctc=VOCAB_SIZE_CTC,
                         nhead=nhead,
                         num_layers=num_layers,
                         use_stn=use_stn).to(device)
    
    # 双重损失函数
    criterion_ctc = nn.CTCLoss(blank=0, reduction='mean', zero_infinity=True)
    criterion_attn = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN)
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    best_loss = float('inf')
    total_steps = epochs * len(train_loader)
    current_step = 0
    
    base_teacher_forcing_ratio = 0.8
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        # 退火
        tf_ratio = max(0.2, base_teacher_forcing_ratio - epoch * 0.1)
        
        for i, (images, texts, attn_targets, ctc_targets, ctc_target_lengths) in enumerate(train_loader):
            images = images.to(device)
            attn_targets = attn_targets.to(device)
            ctc_targets = ctc_targets.to(device)
            
            # Forward
            with torch.cuda.amp.autocast(enabled=enable_amp):
                ctc_outputs, outputs_attn, _, _ = model(images, attn_targets, teacher_forcing_ratio=tf_ratio)
            
            # --- 1. CTC Loss 计算 ---
            # ctc_outputs: [B, 32, VOCAB_SIZE_CTC] -> [32, B, VOCAB_SIZE_CTC]
            # 显式退出 autocast (转为 float32)
            ctc_outputs_t = ctc_outputs.transpose(0, 1).float()
            log_probs = torch.nn.functional.log_softmax(ctc_outputs_t, dim=2)
            input_lengths = torch.full(size=(images.size(0),), fill_value=32, dtype=torch.long)
            loss_ctc = criterion_ctc(log_probs, ctc_targets, input_lengths, ctc_target_lengths)
            
            # --- 2. Attention CE Loss 计算 ---
            # 显式退出 autocast (转为 float32)
            outputs_flatten = outputs_attn.view(-1, VOCAB_SIZE_ATTN).float()
            targets_shifted = torch.cat([attn_targets[:, 1:], torch.full((attn_targets.size(0), 1), PAD_TOKEN, dtype=torch.long, device=device)], dim=1)
            targets_flatten = targets_shifted.view(-1)
            loss_attn = criterion_attn(outputs_flatten, targets_flatten)
            
            # --- 3. 联合优化 ---
            loss = lambda_ctc * loss_ctc + (1.0 - lambda_ctc) * loss_attn
            
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            scaler.step(optimizer)
            scaler.update()
            
            running_loss += loss.item()
            current_step += 1
            
            if current_step % 10 == 0 and progress_callback is not None:
                progress_callback(current_step, total_steps)
                
        epoch_loss = running_loss / len(train_loader)
                
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, texts, attn_targets, ctc_targets, ctc_target_lengths in test_loader:
                images = images.to(device)
                attn_targets = attn_targets.to(device)
                ctc_targets = ctc_targets.to(device)
                
                with torch.cuda.amp.autocast(enabled=enable_amp):
                    ctc_outputs, outputs_attn, _, _ = model(images, trg=attn_targets, teacher_forcing_ratio=0.0)
                
                ctc_outputs_t = ctc_outputs.transpose(0, 1).float()
                log_probs = torch.nn.functional.log_softmax(ctc_outputs_t, dim=2)
                input_lengths = torch.full(size=(images.size(0),), fill_value=32, dtype=torch.long)
                v_loss_ctc = criterion_ctc(log_probs, ctc_targets, input_lengths, ctc_target_lengths)
                
                outputs_flatten = outputs_attn.view(-1, VOCAB_SIZE_ATTN).float()
                targets_shifted = torch.cat([attn_targets[:, 1:], torch.full((attn_targets.size(0), 1), PAD_TOKEN, dtype=torch.long, device=device)], dim=1)
                targets_flatten = targets_shifted.view(-1)
                v_loss_attn = criterion_attn(outputs_flatten, targets_flatten)
                
                val_loss += (lambda_ctc * v_loss_ctc + (1.0 - lambda_ctc) * v_loss_attn).item()
                
        val_loss /= len(test_loader)
        
        if metric_callback is not None:
            metric_callback(epoch + 1, epoch_loss, val_loss)
            
        print(f"Epoch [{epoch+1}/{epochs}] Train Loss: {epoch_loss:.4f} Val Loss: {val_loss:.4f} | TF Ratio: {tf_ratio:.2f}")
        
        if val_loss < best_loss:
            best_loss = val_loss
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(model.state_dict(), save_path)

    return best_loss

if __name__ == "__main__":
    train_v4_joint_model(epochs=2)
