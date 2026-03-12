import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms

from utils.data_generator import SemanticTextDataset
from models.seq2seq_attn import Seq2Seq

# 字典构建: 0-9 (10个), -, :, A, B, C (5个), PAD, SOS, EOS (3个) => 18类
# 约定 ID: PAD=0, SOS=1, EOS=2. 实际的 char 按顺序从 3 开始偏移
PAD_TOKEN = 0
SOS_TOKEN = 1
EOS_TOKEN = 2

# 从 Dataset 中获取字符表并建立双向映射
dummy_ds = SemanticTextDataset(num_samples=1)
CHAR_LIST = dummy_ds.char_list
char2id = {c: i + 3 for i, c in enumerate(CHAR_LIST)}
id2char = {i + 3: c for i, c in enumerate(CHAR_LIST)}
id2char[PAD_TOKEN] = '<PAD>'
id2char[SOS_TOKEN] = '<SOS>'
id2char[EOS_TOKEN] = '<EOS>'
VOCAB_SIZE = len(char2id) + 3

def text_to_tensor(text, max_len=15):
    """
    将原生 "123-A" 转换为 [1(SOS), 4, 5, 6, 13, 16(A), 2(EOS), 0(PAD)...]
    返回固定长度 Max_len, 用于 CrossEntropy 计算
    """
    seq = [SOS_TOKEN]
    for c in text:
        if c in char2id:
            seq.append(char2id[c])
    seq.append(EOS_TOKEN)
    
    # 填补 PAD
    if len(seq) < max_len:
        seq.extend([PAD_TOKEN] * (max_len - len(seq)))
    else:
        # 如果超出截断并确保最后是 EOS
        seq = seq[:max_len]
        seq[-1] = EOS_TOKEN
        
    return torch.tensor(seq, dtype=torch.long)

def collate_fn_seq2seq(batch):
    images, texts = zip(*batch)
    images = torch.stack(images, 0)
    
    targets = []
    for text in texts:
        targets.append(text_to_tensor(text))
        
    targets = torch.stack(targets, 0)
    return images, targets, texts

def train_seq2seq_model(epochs=10, batch_size=32, learning_rate=0.001, save_path="models/best_v3_seq2seq.pth",
                        progress_callback=None, metric_callback=None):
    
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
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn_seq2seq)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn_seq2seq)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = Seq2Seq(vocab_size=VOCAB_SIZE).to(device)
    
    # 注意力解码通常用 CrossEntropy，忽略 PAD token 的 loss 贡献
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    best_loss = float('inf')
    total_steps = epochs * len(train_loader)
    current_step = 0
    max_len = 15
    
    # 退火策略：随 epoch 逐渐下降 teacher_forcing 依赖，让模型学会自己走路
    base_teacher_forcing_ratio = 0.8
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        # 退火
        tf_ratio = max(0.2, base_teacher_forcing_ratio - epoch * 0.1)
        
        for i, (images, targets, _) in enumerate(train_loader):
            images = images.to(device)
            targets = targets.to(device) # [B, T=15]
            
            # Forward
            outputs, _, _ = model(images, targets, teacher_forcing_ratio=tf_ratio, max_len=max_len)
            
            # outputs shape: [B, max_len, Vocab_size]
            # [B, T, V] -> [B*T, V]
            outputs_flatten = outputs.view(-1, VOCAB_SIZE)
            
            # 注意：我们的 Target 里第一位是 SOS，模型是从接到 SOS 开始预测之后的一个字。
            # 模型 outputs 的第 0 步结果对应的应该是 target 的第 1 步 (即第一个真实字符)
            # targets: [B, T] -> target_for_loss: 取 [:, 1:] 然后最后补 PAD
            targets_shifted = torch.cat([targets[:, 1:], torch.full((targets.size(0), 1), PAD_TOKEN, dtype=torch.long, device=device)], dim=1)
            targets_flatten = targets_shifted.view(-1)
            
            loss = criterion(outputs_flatten, targets_flatten)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
            
            running_loss += loss.item()
            current_step += 1
            
            if current_step % 10 == 0 and progress_callback is not None:
                progress_callback(current_step, total_steps)
                
        epoch_loss = running_loss / len(train_loader)
                
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, targets, _ in test_loader:
                images = images.to(device)
                targets = targets.to(device)
                outputs, _, _ = model(images, targets=None, teacher_forcing_ratio=0.0, max_len=max_len)
                
                outputs_flatten = outputs.view(-1, VOCAB_SIZE)
                targets_shifted = torch.cat([targets[:, 1:], torch.full((targets.size(0), 1), PAD_TOKEN, dtype=torch.long, device=device)], dim=1)
                targets_flatten = targets_shifted.view(-1)
                
                loss = criterion(outputs_flatten, targets_flatten)
                val_loss += loss.item()
                
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
    train_seq2seq_model(epochs=2)
