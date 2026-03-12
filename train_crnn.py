import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms

from utils.data_generator import SyntheticTextDataset
from models.crnn_ctc import CRNN

def collate_fn_crnn(batch):
    """
    处理变长文本标签为 CTCLoss 要求的格式。
    CTCLoss 需求:
    - targets: 一维 tensor 包含所有 batch 中连成片段的标签。
    - target_lengths: 记录每一个 batch item 的 target 真实长度。
    """
    images, texts = zip(*batch)
    images = torch.stack(images, 0)
    
    targets = []
    target_lengths = []
    for text in texts:
        # 标签是 0-9 字符串，由于类别 0 是 Blank，我们的标签需要后移一位。
        # 即 "0" -> class 1，"1" -> class 2 ... "9" -> class 10
        target = [int(c) + 1 for c in text]
        targets.extend(target)
        target_lengths.append(len(target))
        
    targets = torch.tensor(targets, dtype=torch.long)
    target_lengths = torch.tensor(target_lengths, dtype=torch.long)
    
    return images, targets, target_lengths, texts

def train_crnn_model(epochs=5, batch_size=32, learning_rate=0.001, save_path="models/best_crnn.pth",
                     progress_callback=None, metric_callback=None):
    
    # 与 V1 相似的 Invert 变换
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
    train_dataset = SyntheticTextDataset(num_samples=train_samples, transform=train_transform)
    test_dataset = SyntheticTextDataset(num_samples=test_samples, transform=test_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn_crnn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn_crnn)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 模型输出 11 类: index 0 (Blank), index 1-10 (数字 0-9)
    model = CRNN(num_classes=11).to(device)
    
    # 定义 CTC Loss, blank=0 指定分类 0 为空白符
    criterion = nn.CTCLoss(blank=0, reduction='mean', zero_infinity=True)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    best_loss = float('inf')
    total_steps = epochs * len(train_loader)
    current_step = 0
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        for i, (images, targets, target_lengths, _) in enumerate(train_loader):
            images = images.to(device)
            targets = targets.to(device)
            
            # Predict shape: [T, B, num_classes] => [32, Batch, 11]
            outputs = model(images)
            
            # log_softmax 是 CTCLoss 的输入要求
            log_probs = torch.nn.functional.log_softmax(outputs, dim=2)
            
            # 准备输入长度矩阵
            batch_size_current = images.size(0)
            # CNN-RNN 确定的序列输出长度
            input_lengths = torch.full(size=(batch_size_current,), fill_value=32, dtype=torch.long)
            
            loss = criterion(log_probs, targets, input_lengths, target_lengths)
            
            optimizer.zero_grad()
            loss.backward()
            
            # 防止梯度爆炸，RNN 训练常做 Clip
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
            for images, targets, target_lengths, _ in test_loader:
                images = images.to(device)
                targets = targets.to(device)
                outputs = model(images)
                log_probs = torch.nn.functional.log_softmax(outputs, dim=2)
                
                bs = images.size(0)
                input_lengths = torch.full(size=(bs,), fill_value=32, dtype=torch.long)
                loss = criterion(log_probs, targets, input_lengths, target_lengths)
                val_loss += loss.item()
                
        val_loss /= len(test_loader)
        
        if metric_callback is not None:
            # 此处我们通过展现 Val Loss 来代替 Acc (CTC Exact Match Acc 需要做解码计算，这里为求简用 Loss 代替效果)
            metric_callback(epoch + 1, epoch_loss, val_loss)
        
        if val_loss < best_loss:
            best_loss = val_loss
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(model.state_dict(), save_path)

    return best_loss

if __name__ == "__main__":
    train_crnn_model(epochs=1)
