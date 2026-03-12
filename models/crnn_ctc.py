import torch
import torch.nn as nn
import torch.nn.functional as F

class BidirectionalLSTM(nn.Module):
    def __init__(self, nIn, nHidden, nOut):
        super(BidirectionalLSTM, self).__init__()
        self.rnn = nn.LSTM(nIn, nHidden, bidirectional=True)
        self.embedding = nn.Linear(nHidden * 2, nOut)

    def forward(self, input):
        # input shape: [T, b, nIn]
        recurrent, _ = self.rnn(input)
        T, b, h = recurrent.size()
        t_rec = recurrent.view(T * b, h)
        output = self.embedding(t_rec)  # [T * b, nOut]
        output = output.view(T, b, -1)
        return output

class CRNN(nn.Module):
    """
    针对当前合成定轨(32x256)特化过的轻量 CRNN 结构
    输入要求: [B, 1, 32, 256] 灰度图
    输出返回: [T, B, 11] 即时间步序列上对 11 类概率
    """
    def __init__(self, img_channel=1, hidden_size=256, num_classes=11):
        super(CRNN, self).__init__()
        
        # 1. CNN Feature Extraction
        # 我们希望把 H:32 降维成 H:1，而 W:256 降维成序列 T:32 (即每经过一个MaxPool缩减一半，但维持一定宽度特征)
        # 这意味着我们将得到 T=32 的序列步度。
        self.cnn = nn.Sequential(
            # in: 1 x 32 x 256
            nn.Conv2d(img_channel, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2), # out: 64 x 16 x 128
            
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2), # out: 128 x 8 x 64
            
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            # 自定义池化：H 轴缩小一半，W 轴缩小一半
            nn.MaxPool2d(kernel_size=2, stride=2), # out: 256 x 4 x 32
            
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            # 自定义池化：这波狠狠压扁 H 轴到底，但不缩放 W(Time)
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)), # out: 256 x 2 x 32
            
            # 再抽干最后一层H
            nn.Conv2d(256, 256, kernel_size=2, stride=1, padding=0)
            # out shape 最终应该近似 [B, 256, 1, 31] 之类的，方便拉平展开成 RNN 序列特征
            # 为了确保固定输出长度，我们将在此使用自适应池化将其按需求规整到 T=32
        )
        # 我们直接使用 AdaptiveAvgPool 强制得到 H=1, W=32 的输出特征
        self.pool = nn.AdaptiveAvgPool2d((1, 32))
        
        # 2. RNN Sequence Modeling
        # 特征提取后的通道数是 256
        self.rnn = nn.Sequential(
            BidirectionalLSTM(256, hidden_size, hidden_size),
            BidirectionalLSTM(hidden_size, hidden_size, num_classes)
        )

    def forward(self, input):
        # conv features
        conv = self.cnn(input)
        conv = self.pool(conv)
        # 此时 conv 形状: [batch, channel(256), h(1), w(32)]
        
        b, c, h, w = conv.size()
        
        # 为了喂入 RNN 我们需要移除高度维度并且变换为 [Time, Batch, Channel] 结构
        conv = conv.squeeze(2) # [b, c, w] 
        conv = conv.permute(2, 0, 1)  # [w(Time=32), b, c(256)]
        
        # rnn features
        output = self.rnn(conv)   # output: [T(32), b, num_classes(11)]
        return output

if __name__ == '__main__':
    # 架构自测检定
    model = CRNN(num_classes=11)
    dummy_input = torch.randn(2, 1, 32, 256)
    output = model(dummy_input)
    print("Dummy Forward Shape (T, Batch, Num_Classes):", output.shape) # Expected: 32, 2, 11
