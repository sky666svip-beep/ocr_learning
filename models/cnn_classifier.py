import torch
import torch.nn as nn

class SimpleDigitCNN(nn.Module):
    """
    轻量级卷积网络结构，用于对独立的 32x32 灰度字符图像进行 0-9 十分类。
    使用类似于 LeNet-5 的两层卷积架构。
    输入要求: (Batch, 1, 32, 32)
    """
    def __init__(self, num_classes=10):
        super(SimpleDigitCNN, self).__init__()
        
        # 卷积提取层
        self.features = nn.Sequential(
            # in: 1 x 32 x 32  => out: 16 x 32 x 32
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            # 池化后: out: 16 x 16 x 16
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # in: 16 x 16 x 16 => out: 32 x 16 x 16
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            # 池化后: out: 32 x 8 x 8
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # 全连接分类层
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 8 * 8, 128),
            nn.ReLU(),
            nn.Dropout(p=0.5), # 防止针对合成字体过拟合
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

if __name__ == '__main__':
    # 网络形状及流向断言验证
    model = SimpleDigitCNN()
    dummy_input = torch.randn(1, 1, 32, 32)
    output = model(dummy_input)
    print(f"Model output shape: {output.shape}") # Should be (1, 10)
