import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms

from utils.data_generator import SyntheticDigitDataset
from models.cnn_classifier import SimpleDigitCNN

def train_model(epochs=5, batch_size=64, learning_rate=0.001, save_path="models/best_model.pth",
                progress_callback=None, metric_callback=None):
    """
    修改后的训练循环，增加两个回调钩子暴露给 Streamlit UI:
    progress_callback(current, total): 用于更新总体训练进度条。
    metric_callback(epoch, train_loss, val_acc): 用于实时绘制 metrics 折线图。
    """
    train_transform = transforms.Compose([
        transforms.RandomAffine(degrees=5, translate=(0.05, 0.05), scale=(0.95, 1.05)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: 1.0 - x), 
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: 1.0 - x),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # 这里为了在 UI 上快速演示，我们将合成数据量适当缩小以保证极速反馈
    train_samples = 5000
    test_samples = 1000
    train_dataset = SyntheticDigitDataset(num_samples=train_samples, transform=train_transform)
    test_dataset = SyntheticDigitDataset(num_samples=test_samples, transform=test_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = SimpleDigitCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    best_acc = 0.0
    total_steps = epochs * len(train_loader)
    current_step = 0
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            current_step += 1
            
            # 每 10 步更新一下底层进度条
            if current_step % 10 == 0 and progress_callback is not None:
                progress_callback(current_step, total_steps)
                
        # 每一个 epoch 计算平均 loss
        epoch_loss = running_loss / len(train_loader)
                
        # Validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
        val_acc = 100 * correct / total
        
        # 传递绘制数据 (Epoch, 归一化的训练 Loss，验证组准确率)
        if metric_callback is not None:
            metric_callback(epoch + 1, epoch_loss, val_acc)
        else:
            print(f"Epoch [{epoch+1}/{epochs}] Validation Accuracy: {val_acc:.2f}%")
        
        # 保存最佳权重
        if val_acc > best_acc:
            best_acc = val_acc
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(model.state_dict(), save_path)

    return best_acc

if __name__ == "__main__":
    train_model(epochs=3)
