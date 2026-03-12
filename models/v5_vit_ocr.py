import torch
import torch.nn as nn
import math

# ============================================================
# V5: 纯 Vision Transformer OCR
# 彻底抛弃 CNN，用 Patch Embedding 将图像碎为 Token 序列
# ViT Encoder (Self-Attention) + Transformer Decoder (Cross-Attention)
# ============================================================


class PatchEmbedding(nn.Module):
    """
    将图像切成固定大小的 Patch 并线性映射为 d_model 维 Token
    输入: (B, 1, 32, 256) 灰度图
    Patch 尺寸: 4x8 → 行方向 32/4=8, 列方向 256/8=32 → 共 256 个 Patch
    输出: (B, 256, d_model)
    """
    def __init__(self, img_h=32, img_w=256, patch_h=4, patch_w=8, in_channels=1, d_model=64):
        super().__init__()
        self.num_patches_h = img_h // patch_h  # 8
        self.num_patches_w = img_w // patch_w  # 32
        self.num_patches = self.num_patches_h * self.num_patches_w  # 256
        self.patch_dim = in_channels * patch_h * patch_w  # 1 * 4 * 8 = 32

        # 用 Conv2d 实现 Patch 切分 + 线性映射（等效于手动切片 + nn.Linear）
        self.proj = nn.Conv2d(in_channels, d_model, kernel_size=(patch_h, patch_w), stride=(patch_h, patch_w))

        # 可学习位置编码（区别于 V4 的正弦位置编码）
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches, d_model) * 0.02)

    def forward(self, x):
        # x: (B, 1, 32, 256)
        x = self.proj(x)        # (B, d_model, 8, 32)
        x = x.flatten(2)        # (B, d_model, 256)
        x = x.transpose(1, 2)   # (B, 256, d_model)
        x = x + self.pos_embed  # 加上可学习位置
        return x


class ViTEncoderLayer(nn.Module):
    """带 Hook 的自定义 Encoder 层，用于捕获 Self-Attention 权重"""
    def __init__(self, d_model=64, nhead=2, dim_ff=256, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_ff, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.drop1 = nn.Dropout(dropout)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, src):
        # Self-Attention + Residual
        src2, attn_weights = self.self_attn(src, src, src, need_weights=True)
        src = src + self.drop1(src2)
        src = self.norm1(src)
        # Feed Forward + Residual
        src2 = self.ff(src)
        src = src + self.drop2(src2)
        src = self.norm2(src)
        return src, attn_weights  # attn_weights: (B, num_patches, num_patches)


class ViTEncoder(nn.Module):
    """多层 ViT Encoder，自动收集每层的 Self-Attention 权重"""
    def __init__(self, d_model=64, nhead=2, num_layers=2, dim_ff=256, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            ViTEncoderLayer(d_model, nhead, dim_ff, dropout) for _ in range(num_layers)
        ])
        self.self_attn_maps = []

    def forward(self, src):
        self.self_attn_maps = []
        output = src
        for layer in self.layers:
            output, attn = layer(output)
            self.self_attn_maps.append(attn)  # (B, num_patches, num_patches)
        return output


class TransformerDecoderLayer(nn.Module):
    """带 Hook 的 Decoder 层：Masked Self-Attention + Cross-Attention"""
    def __init__(self, d_model=64, nhead=2, dim_ff=256, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_ff, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.drop1 = nn.Dropout(dropout)
        self.drop2 = nn.Dropout(dropout)
        self.drop3 = nn.Dropout(dropout)

    def forward(self, tgt, memory, tgt_mask=None):
        # 1. Masked Self-Attention（防止窥看未来 Token）
        tgt2, _ = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask)
        tgt = tgt + self.drop1(tgt2)
        tgt = self.norm1(tgt)
        # 2. Cross-Attention（文本 Token 注意图像 Patch）
        tgt2, cross_attn_weights = self.cross_attn(tgt, memory, memory, need_weights=True)
        tgt = tgt + self.drop2(tgt2)
        tgt = self.norm2(tgt)
        # 3. Feed Forward
        tgt2 = self.ff(tgt)
        tgt = tgt + self.drop3(tgt2)
        tgt = self.norm3(tgt)
        return tgt, cross_attn_weights  # cross_attn_weights: (B, tgt_len, num_patches)


class TransformerOCRDecoder(nn.Module):
    """多层 Transformer Decoder，自动收集 Cross-Attention 权重"""
    def __init__(self, vocab_size, d_model=64, nhead=2, num_layers=2, dim_ff=256, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, 50, d_model) * 0.02)  # 最多 50 步
        self.layers = nn.ModuleList([
            TransformerDecoderLayer(d_model, nhead, dim_ff, dropout) for _ in range(num_layers)
        ])
        self.fc_out = nn.Linear(d_model, vocab_size)
        self.cross_attn_maps = []

    def _generate_square_subsequent_mask(self, sz, device):
        mask = torch.triu(torch.ones(sz, sz, device=device), diagonal=1).bool()
        return mask

    def forward(self, tgt_ids, memory):
        """
        tgt_ids: (B, L) 目标序列的 Token ID
        memory: (B, num_patches, d_model) Encoder 输出
        """
        self.cross_attn_maps = []
        B, L = tgt_ids.shape

        # Token Embedding + Position
        tgt = self.embedding(tgt_ids) * math.sqrt(self.d_model)
        tgt = tgt + self.pos_embed[:, :L, :]

        # Causal Mask（防止窥看未来）
        tgt_mask = self._generate_square_subsequent_mask(L, tgt.device)

        for layer in self.layers:
            tgt, cross_attn = layer(tgt, memory, tgt_mask=tgt_mask)
            self.cross_attn_maps.append(cross_attn)

        logits = self.fc_out(tgt)  # (B, L, vocab_size)
        return logits


class V5ViTOCR(nn.Module):
    """
    V5 完整模型：PatchEmbedding → ViTEncoder → TransformerDecoder
    彻底无 CNN 的端到端纯 Transformer OCR
    """
    def __init__(self, vocab_size, d_model=64, nhead=2, num_enc_layers=2, num_dec_layers=2, dim_ff=256):
        super().__init__()
        self.patch_embed = PatchEmbedding(d_model=d_model)
        self.encoder = ViTEncoder(d_model, nhead, num_enc_layers, dim_ff)
        self.decoder = TransformerOCRDecoder(vocab_size, d_model, nhead, num_dec_layers, dim_ff)

    def forward(self, img, tgt_ids):
        """
        img: (B, 1, 32, 256)
        tgt_ids: (B, L) 含 <SOS> 前缀、不含末尾 <EOS> 的目标序列
        返回: logits (B, L, vocab_size)
        """
        patches = self.patch_embed(img)          # (B, 256, d_model)
        memory = self.encoder(patches)            # (B, 256, d_model)
        logits = self.decoder(tgt_ids, memory)    # (B, L, vocab_size)
        return logits

    def inference(self, img, sos_id, eos_id, max_len=20):
        """
        自回归推理（贪心解码）
        返回: predictions list, encoder_self_attn, decoder_cross_attn
        """
        self.eval()
        with torch.no_grad():
            patches = self.patch_embed(img)
            memory = self.encoder(patches)

            B = img.size(0)
            input_ids = torch.full((B, 1), sos_id, dtype=torch.long, device=img.device)
            predictions = []

            for step in range(max_len):
                logits = self.decoder(input_ids, memory)
                next_token = logits[:, -1, :].argmax(dim=-1)  # (B,)
                predictions.append(next_token.item())

                if next_token.item() == eos_id:
                    break

                input_ids = torch.cat([input_ids, next_token.unsqueeze(1)], dim=1)

        return predictions, self.encoder.self_attn_maps, self.decoder.cross_attn_maps
