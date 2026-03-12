import torch
import torch.nn as nn
import torch.nn.functional as F

class EncoderCNN(nn.Module):
    """
    V3 Encoder CNN: 延续自定义轻量架构。
    输入大小必须兼容 Batch x 1 x 32 x 256
    由于我们需要在 Attention 层进行空间维度的权衡矩阵匹配，
    我们这里不像 CRNN 一样把高度强行压扁为 1。我们保留一定的空间结构，例如将其折叠为 H=4, W=32
    """
    def __init__(self, img_channel=1):
        super(EncoderCNN, self).__init__()
        self.cnn = nn.Sequential(
            # in: 1 x 32 x 256
            nn.Conv2d(img_channel, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2), # out: 64 x 16 x 128
            
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2), # out: 128 x 8 x 64
            
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2), # out: 256 x 4 x 32 
            
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True)
        )
        # 不再继续缩小，保留 4x32 的空间视野

    def forward(self, images):
        # images: [B, C, H, W]
        features = self.cnn(images) # [B, 256, 4, 32]
        
        # 将空间维度展平作为 Attention 可以扫描的“像素像素序列”长度 L
        B, C, H, W = features.size()
        # [B, C, H*W] -> [B, L, C]
        features_flat = features.view(B, C, -1).permute(0, 2, 1) 
        return features, features_flat


class AttentionDecoder(nn.Module):
    """
    基于 Bahdanau (Additive) Attention 机制的 LSTM 解码器
    """
    def __init__(self, vocab_size, embed_size=128, hidden_size=256, encoder_dim=256, max_len=15):
        super(AttentionDecoder, self).__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.encoder_dim = encoder_dim
        
        self.embedding = nn.Embedding(vocab_size, embed_size)
        
        # Bahdanau Attention layers
        self.W1 = nn.Linear(encoder_dim, hidden_size)
        self.W2 = nn.Linear(hidden_size, hidden_size) # query (LSTM hidden state)
        self.v = nn.Linear(hidden_size, 1)
        
        # Decoder 接收前一个 Token 的 Embedding + 这一步找出的特征上下文向量 Context
        self.lstm_cell = nn.LSTMCell(embed_size + encoder_dim, hidden_size)
        
        self.fc_out = nn.Linear(hidden_size, vocab_size)

    def forward(self, decoder_input, hidden_state, cell_state, encoder_features):
        """
        decoder_input: [B] => 前一步推测出的词 (或者 Teacher Forcing 强制传入的答案)
        hidden_state: [B, hidden_size]
        cell_state: [B, hidden_size]
        encoder_features: [B, L(128), encoder_dim(256)]
        返回: 预测概率分布, 更新后的 hidden, cell, 以及 Attention 热力权重
        """
        B = encoder_features.size(0)
        
        # 计算 Attention Scores
        # hidden_state: [B, hidden_size] -> [B, 1, hidden_size]
        hidden_expanded = hidden_state.unsqueeze(1) 
        
        # score = v * tanh(W1(enc) + W2(dec_hid)) -> [B, L, 1]
        energy = torch.tanh(self.W1(encoder_features) + self.W2(hidden_expanded))
        attention_scores = self.v(energy).squeeze(2) # [B, L]
        
        # Attention Weights [B, L]
        alpha = F.softmax(attention_scores, dim=1) 
        
        # 计算 Context Vector
        # alpha.unsqueeze(1): [B, 1, L]
        # bmm with encoder_features [B, L, C]: [B, 1, C]
        context_vector = torch.bmm(alpha.unsqueeze(1), encoder_features).squeeze(1) # [B, C]
        
        # Embedding current input [B, embed_size]
        embedded = self.embedding(decoder_input) 
        
        # Concat context & embedding
        lstm_input = torch.cat((embedded, context_vector), dim=1)
        
        # LSTM Step
        hidden_state, cell_state = self.lstm_cell(lstm_input, (hidden_state, cell_state))
        
        # Prediction
        prediction = self.fc_out(hidden_state)
        
        # 返回 alpha 方便绘制动图 (Attention Map)
        return prediction, hidden_state, cell_state, alpha

class Seq2Seq(nn.Module):
    def __init__(self, vocab_size, embed_size=128, hidden_size=256, encoder_dim=256):
        super(Seq2Seq, self).__init__()
        self.encoder = EncoderCNN()
        self.decoder = AttentionDecoder(vocab_size, embed_size, hidden_size, encoder_dim)
        
    def forward(self, images, targets=None, teacher_forcing_ratio=0.5, max_len=15):
        """
        images: [B, 1, H, W]
        targets: [B, T] 这是为了 Teacher forcing
        """
        B = images.size(0)
        # Encoder
        feat_map, enc_features = self.encoder(images) 
        
        # 初始化 Decoder 隐藏状态 (可以直接全 0，或取特征的均值)
        hidden_state = torch.zeros(B, self.decoder.hidden_size).to(images.device)
        cell_state = torch.zeros(B, self.decoder.hidden_size).to(images.device)
        
        # SOS Token 获取 （在外部我们要约定 index 0 是 PAD，1 是 SOS，2 是 EOS）
        SOS_TOKEN = 1
        decoder_input = torch.full((B,), SOS_TOKEN, dtype=torch.long, device=images.device)
        
        outputs = []
        attention_maps = []
        
        # 这里进行自回归 Step by Step
        for t in range(max_len):
            prediction, hidden_state, cell_state, alpha = self.decoder(
                decoder_input, hidden_state, cell_state, enc_features
            )
            
            outputs.append(prediction)
            attention_maps.append(alpha) # 收集视角热力图
            
            # 是否执行 Teacher Forcing
            if targets is not None and torch.rand(1).item() < teacher_forcing_ratio:
                # 使用真实 Label 引导下一步
                # 注意 targets 也是 [B, T]，但需排除掉开头，看调用方式
                decoder_input = targets[:, t]
            else:
                # 模型自推断
                decoder_input = prediction.argmax(1) 
                
        # outputs shape: [B, max_len, vocab_size]
        outputs = torch.stack(outputs, dim=1)
        # attention_maps: [B, max_len, L(128)]
        attention_maps = torch.stack(attention_maps, dim=1)
        
        return outputs, attention_maps, feat_map
