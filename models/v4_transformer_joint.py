import torch
import torch.nn as nn
import math

class EncoderCNN(nn.Module):
    """
    沿用 V3 的轻量级 CNN 特征提取器
    输入: (B, 1, 32, 256)
    输出: (B, C, H, W) -> 默认 (B, 64, 4, 32)
    """
    def __init__(self):
        super(EncoderCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2) # 16x128
        
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2) # 8x64
        
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(2, 2) # 4x32
        
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool1(self.relu(self.conv1(x)))
        x = self.pool2(self.relu(self.conv2(x)))
        x = self.pool3(self.relu(self.conv3(x)))
        return x

class PositionalEncoding(nn.Module):
    """
    标准的 Transformer 位置编码
    """
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1) # (max_len, 1, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: (T, B, d_model)
        x = x + self.pe[:x.size(0), :]
        return x

class SpatialTransformerNetwork(nn.Module):
    """
    STN: 学习仿射变换矩阵，对扭曲、透视变形的输入图像进行校正。
    """
    def __init__(self):
        super(SpatialTransformerNetwork, self).__init__()
        
        # Spatial transformer localization-network
        self.localization = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(8, 10, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True)
        )
        
        # Regressor for the 3 * 2 affine matrix
        self.fc_loc = nn.Sequential(
            nn.Linear(10 * 4 * 60, 32), # 假设输入是 32x256，经过两次 pool 大约是这个尺寸
            nn.ReLU(True),
            nn.Linear(32, 3 * 2)
        )
        
        # Initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    def forward(self, x):
        xs = self.localization(x)
        xs = xs.view(-1, 10 * xs.shape[2] * xs.shape[3]) # 动态适应尺寸
        # 如果尺寸不对，我们退回恒等变换
        if xs.shape[1] != self.fc_loc[0].in_features:
            theta = torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float, device=x.device).view(-1, 2, 3)
            theta = theta.repeat(x.size(0), 1, 1)
        else:
            theta = self.fc_loc(xs)
            theta = theta.view(-1, 2, 3)

        grid = torch.nn.functional.affine_grid(theta, x.size(), align_corners=True)
        x = torch.nn.functional.grid_sample(x, grid, align_corners=True)
        return x

class AttentionDecoder(nn.Module):
    """
    沿用/微调 V3 的 RNN Attention Decoder。平滑承接 Transformer Encoder的输出。
    """
    def __init__(self, output_dim, emb_dim=64, enc_hid_dim=64, dec_hid_dim=128):
        super(AttentionDecoder, self).__init__()
        self.output_dim = output_dim
        self.embedding = nn.Embedding(output_dim, emb_dim)
        
        # Attention 层：计算 Decoder hidden states 与 Encoder outputs 的相关性
        self.attn = nn.Linear(enc_hid_dim + dec_hid_dim, dec_hid_dim)
        self.v = nn.Linear(dec_hid_dim, 1, bias=False)
        
        # RNN 接收 (Embedding + Context Vector)
        self.rnn = nn.GRU(emb_dim + enc_hid_dim, dec_hid_dim, batch_first=True)
        # 最终分类器
        self.fc_out = nn.Linear(dec_hid_dim + emb_dim + enc_hid_dim, output_dim)
        
    def forward(self, input_step, hidden, encoder_outputs):
        # input_step: (B,) 当前步输入的字符索引 (通常是前一次的预测或 Target)
        # hidden: (1, B, dec_hid_dim) Decoder 当前隐藏状态
        # encoder_outputs: (B, T, enc_hid_dim) Transformer Encoder 的输出
        
        B, T, _ = encoder_outputs.shape
        input_step = input_step.unsqueeze(1) # (B, 1)
        embedded = self.embedding(input_step) # (B, 1, emb_dim)
        
        # 计算 Attention
        hidden_expanded = hidden.repeat(T, 1, 1).transpose(0, 1) # (B, T, dec_hid_dim)
        energy = torch.tanh(self.attn(torch.cat((hidden_expanded, encoder_outputs), dim=2))) # (B, T, dec_hid_dim)
        attention_weights = torch.softmax(self.v(energy).squeeze(2), dim=1) # (B, T)
        
        # 计算 Context Vector
        attention_weights = attention_weights.unsqueeze(1) # (B, 1, T)
        context = torch.bmm(attention_weights, encoder_outputs) # (B, 1, enc_hid_dim)
        
        # 融入 RNN
        rnn_input = torch.cat((embedded, context), dim=2) # (B, 1, emb_dim + enc_hid_dim)
        output, hidden = self.rnn(rnn_input, hidden) # output: (B, 1, dec_hid_dim)
        
        # 预测预测
        prediction = self.fc_out(torch.cat((output.squeeze(1), context.squeeze(1), embedded.squeeze(1)), dim=1))
        
        return prediction, hidden, attention_weights.squeeze(1)

# Hook 用于捕捉 Transformer Attention Map
class TransformerEncoderWithHook(nn.Module):
    def __init__(self, encoder_layer, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([import_layer(encoder_layer) for _ in range(num_layers)])
        self.self_attn_weights = []

    def forward(self, src):
        self.self_attn_weights = []
        output = src
        for layer in self.layers:
            # 需要魔改一层来抓取 attention... TODO: Simplest way is to define a custom layer or just use PyTorch 2.0+ `need_weights=True`
            # For simplicity, we construct a custom Transformer Encoder Layer
            output, attn = layer(output)
            self.self_attn_weights.append(attn)
        return output
        
def import_layer(layer):
    import copy
    return copy.deepcopy(layer)

class CustomTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=256, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
    def forward(self, src):
        src2, attn = self.self_attn(src, src, src, need_weights=True)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        
        src2 = self.linear2(self.dropout(torch.relu(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src, attn

class TransformerEncoderHooked(nn.Module):
    def __init__(self, d_model, nhead, num_layers, dim_feedforward=256, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([CustomTransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout) for _ in range(num_layers)])
        self.self_attn_weights = [] # Store attention maps
        
    def forward(self, src):
        self.self_attn_weights = []
        output = src
        for layer in self.layers:
            output, attn = layer(output)
            self.self_attn_weights.append(attn) # (B, T, T)
        return output

class V4JointModel(nn.Module):
    def __init__(self, output_dim_attn, output_dim_ctc, d_model=64, nhead=2, num_layers=1, dec_hid_dim=128, use_stn=False):
        super(V4JointModel, self).__init__()
        self.use_stn = use_stn
        if self.use_stn:
            self.stn = SpatialTransformerNetwork()
            
        self.cnn = EncoderCNN()
        
        # 空间平展后映射到 d_model
        # H=4, W=32. 为了得到 1D 序列，按列切片 (就像 V2 CRNN 那样):
        # 每一列相当于一个 TimeStep。所以 T=32, Feature Dim = C * H = 64 * 4 = 256
        self.feature_proj = nn.Linear(64 * 4, d_model)
        
        self.pos_encoder = PositionalEncoding(d_model)
        
        # Transformer Encoder 捕获全局上下文，替代 Bi-LSTM
        self.transformer_encoder = TransformerEncoderHooked(
            d_model=d_model, 
            nhead=nhead, 
            num_layers=num_layers, 
            dim_feedforward=256
        )
        
        # --- 分支 1: CTC 强制对齐分支 (Head) ---
        self.ctc_head = nn.Linear(d_model, output_dim_ctc)
        
        # --- 分支 2: RNN Attention 语义分支 (Head) ---
        self.decoder = AttentionDecoder(
            output_dim=output_dim_attn, 
            emb_dim=64, 
            enc_hid_dim=d_model, 
            dec_hid_dim=dec_hid_dim
        )
        
        # 桥接 Transformer 状态给 RNN Hidden
        self.bridge = nn.Linear(d_model, dec_hid_dim)

    def get_ctc_output(self, x):
        """仅做推理，获取 CTC 对齐图"""
        if self.use_stn:
            x = self.stn(x)
        # CNN (B, 1, 32, 256) -> (B, 64, 4, 32)
        conv_out = self.cnn(x)
        B, C, H, W = conv_out.shape
        # (B, W, C * H) -> (B, T, D)
        features = conv_out.permute(0, 3, 1, 2).reshape(B, W, C * H)
        
        # 投影并加位置编码
        features = self.feature_proj(features) # (B, 32, d_model)
        features = features.transpose(0, 1) # (32, B, d_model) for PE
        features = self.pos_encoder(features)
        features = features.transpose(0, 1) # (B, 32, d_model) for batch_first Transformer
        
        # Transformer
        encoder_outputs = self.transformer_encoder(features) # (B, T, d_model)
        
        # CTC 分发
        ctc_out = self.ctc_head(encoder_outputs) # (B, T, ctc_classes)
        return ctc_out, self.transformer_encoder.self_attn_weights

    def forward(self, x, trg=None, teacher_forcing_ratio=0.5):
        """
        联合前向转播
        x: (B, 1, 32, 256)
        trg: (B, L) Seq2Seq target sequence, if none means inference mode
        """
        if self.use_stn:
            x = self.stn(x)
        # 1. CNN Feature Extraction
        conv_out = self.cnn(x) # (B, 64, 4, 32)
        B, C, H, W = conv_out.shape
        
        # 将空间转换为序列 [B, W, C*H] 
        # 这样 Transformer Encoder 看到的维度是 (B, 32, 256) -> W 变成了时间步 T
        features = conv_out.permute(0, 3, 1, 2).reshape(B, W, C * H)
        features = self.feature_proj(features) # (B, 32, d_model)
        
        # 2. Add Positional Encoding
        features = features.transpose(0, 1) # (32, B, d_model) for PE
        features = self.pos_encoder(features)
        features = features.transpose(0, 1) # (B, 32, d_model) back to batch_first
        
        # 3. Transformer Encoder (Global Context)
        encoder_outputs = self.transformer_encoder(features) # (B, T, d_model)
        
        # 4. CTC Branch
        ctc_outputs = self.ctc_head(encoder_outputs) # (B, T, output_dim_ctc)
        
        # 5. Attention Branch
        # 初始化 Hidden State (用 encoder_outputs 的全局平均汇聚)
        hidden = torch.tanh(self.bridge(encoder_outputs.mean(dim=1))).unsqueeze(0) # (1, B, dec_hid_dim)
        
        if trg is not None:
            max_len = trg.shape[1]
        else:
            max_len = 20 # 默认推理最长步数
            
        outputs_attn = torch.zeros(B, max_len, self.decoder.output_dim).to(x.device)
        attention_maps = torch.zeros(B, max_len, W).to(x.device)
        
        # 永远以 SOS 开始 (假设 SOS index 为 0，等下在 Trainer 对齐)
        # TODO: 这里需要确保 SOS 对应 index = 0
        input_step = trg[:, 0] if trg is not None else torch.zeros(B, dtype=torch.long).to(x.device)
        
        import random
        for t in range(1, max_len):
            prediction, hidden, attn_weights = self.decoder(input_step, hidden, encoder_outputs)
            
            outputs_attn[:, t, :] = prediction
            attention_maps[:, t, :] = attn_weights
            
            # 拿到预测里最高的
            top1 = prediction.argmax(1)
            
            # Teacher forcing (如果是验证或推理，trg=None, tf=0，自动回退取 top1)
            teacher_force = random.random() < teacher_forcing_ratio if trg is not None else False
            input_step = trg[:, t] if teacher_force else top1
            
        return ctc_outputs, outputs_attn, attention_maps, self.transformer_encoder.self_attn_weights
