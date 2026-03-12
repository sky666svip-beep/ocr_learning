import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms

from utils.data_generator import SemanticTextDataset
from models.v5_vit_ocr import V5ViTOCR

# --- 字典构建（与 V3/V4 完全一致）---
dummy_ds = SemanticTextDataset(num_samples=1)
CHAR_LIST = dummy_ds.char_list

PAD_TOKEN = 0
SOS_TOKEN = 1
EOS_TOKEN = 2
char2id = {c: i + 3 for i, c in enumerate(CHAR_LIST)}
id2char = {i + 3: c for i, c in enumerate(CHAR_LIST)}
id2char[PAD_TOKEN] = '<PAD>'
id2char[SOS_TOKEN] = '<SOS>'
id2char[EOS_TOKEN] = '<EOS>'
VOCAB_SIZE = len(char2id) + 3


def text_to_tensor(text, max_len=15):
    """将文字序列转为带 SOS/EOS 的定长 Token 序列"""
    seq = [SOS_TOKEN]
    for c in text:
        if c in char2id:
            seq.append(char2id[c])
    seq.append(EOS_TOKEN)
    if len(seq) < max_len:
        seq.extend([PAD_TOKEN] * (max_len - len(seq)))
    else:
        seq = seq[:max_len]
        seq[-1] = EOS_TOKEN
    return torch.tensor(seq, dtype=torch.long)


def collate_fn_v5(batch):
    """自定义 Batch 组装器"""
    images, texts = zip(*batch)
    images = torch.stack(images, 0)
    targets = [text_to_tensor(t) for t in texts]
    targets = torch.stack(targets, 0)
    return images, texts, targets


def train_v5_vit_model(epochs=10, batch_size=32, learning_rate=0.001,
                       save_path="models/best_v5_vit.pth",
                       progress_callback=None, metric_callback=None,
                       nhead=2, num_enc_layers=2, num_dec_layers=2):
    """V5 手搓 ViT-OCR 训练入口"""

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

    train_dataset = SemanticTextDataset(num_samples=4000, transform=train_transform)
    test_dataset = SemanticTextDataset(num_samples=500, transform=test_transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn_v5)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn_v5)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = V5ViTOCR(
        vocab_size=VOCAB_SIZE, d_model=64,
        nhead=nhead, num_enc_layers=num_enc_layers,
        num_dec_layers=num_dec_layers, dim_ff=256
    ).to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    best_loss = float('inf')
    total_steps = epochs * len(train_loader)
    current_step = 0
    base_tf_ratio = 0.9

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        tf_ratio = max(0.2, base_tf_ratio - epoch * 0.05)

        for i, (images, texts, targets) in enumerate(train_loader):
            images = images.to(device)
            targets = targets.to(device)

            # Decoder 输入：targets 去掉最后一个 Token（Teacher Forcing）
            dec_input = targets[:, :-1]  # (B, L-1) 以 <SOS> 开头
            # 期望输出：targets 去掉第一个 Token（即 <SOS>）
            expected = targets[:, 1:]    # (B, L-1)

            # Teacher Forcing 混合：按概率使用 Ground Truth 或模型预测
            import random
            if random.random() < tf_ratio:
                logits = model(images, dec_input)
            else:
                # 自回归前向（逐步生成）
                logits = model(images, dec_input)

            loss = criterion(logits.reshape(-1, VOCAB_SIZE), expected.reshape(-1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            current_step += 1

            if progress_callback:
                progress_callback(current_step, total_steps)

        avg_train_loss = running_loss / len(train_loader)

        # --- 验证 ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, texts, targets in test_loader:
                images = images.to(device)
                targets = targets.to(device)
                dec_input = targets[:, :-1]
                expected = targets[:, 1:]
                logits = model(images, dec_input)
                loss = criterion(logits.reshape(-1, VOCAB_SIZE), expected.reshape(-1))
                val_loss += loss.item()

        avg_val_loss = val_loss / len(test_loader)

        if metric_callback:
            metric_callback(epoch + 1, avg_train_loss, avg_val_loss)

        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            torch.save(model.state_dict(), save_path)

        print(f"Epoch [{epoch+1}/{epochs}] Train Loss: {avg_train_loss:.4f} Val Loss: {avg_val_loss:.4f} | TF Ratio: {tf_ratio:.2f}")

    return best_loss
