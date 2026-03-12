import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image, ImageDraw, ImageFont

class SyntheticDigitDataset(Dataset):
    """
    动态生成 0-9 印刷体数字图像的 PyTorch 数据集。
    通过随机字体大小、轻微旋转和位置偏移实现数据增强，以提高基础 CNN 模型的泛化能力。
    """
    def __init__(self, num_samples=10000, img_size=(32, 32), transform=None):
        self.num_samples = num_samples
        self.img_size = img_size
        self.transform = transform
        
        # 预加载/配置默认字体 (此处采用预设系统常见字体作为演示基底)
        # Windows 常见的 Arial 字体路径
        self.font_path = "C:/Windows/Fonts/arial.ttf"
        if not os.path.exists(self.font_path):
            self.font_path = "arial.ttf" # Fallback, 依赖 PIL 寻找默认
            
    def _generate_digit_image(self, digit):
        # 1. 创建干净的白色背景 RGB 图像
        img = Image.new('RGB', self.img_size, color=(255, 255, 255))
        draw = ImageDraw.Draw(img)
        
        # 2. 随机配置：大小、角度、偏移量
        font_size = random.randint(20, 28)
        try:
            font = ImageFont.truetype(self.font_path, font_size)
        except IOError:
            font = ImageFont.load_default()
            
        # 文本色彩 (增加少量灰度扰动防止过拟合纯黑色)
        color_val = random.randint(0, 50)
        text_color = (color_val, color_val, color_val)
        
        # 计算绘制居中与抖动
        # 使用 getbbox 获取文本尺寸 (PIL 10.0.0+)
        bbox = draw.textbbox((0, 0), str(digit), font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        x_offset = (self.img_size[0] - text_width) // 2 + random.randint(-2, 2)
        y_offset = (self.img_size[1] - text_height) // 2 + random.randint(-2, 2)
        # 避免超出边界
        x_offset = max(0, min(self.img_size[0] - text_width, x_offset))
        y_offset = max(0, min(self.img_size[1] - text_height, y_offset))
        
        # 3. 绘制文字
        # 注意: 我们可以通过绘制到一个稍大的透明图像上再旋转粘贴来实现旋转，
        # 为了极致简单 KISS 原则，我们先使用 torchvision 的 transforms 实现旋转(见训练代码配置)
        draw.text((x_offset, y_offset), str(digit), fill=text_color, font=font)
        
        # 最后，为了贴合灰度处理后的 CNN 输入需求，转为灰度 (L 模式)
        return img.convert('L')

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # 完全随机生成 0-9 标志签
        label = random.randint(0, 9)
        img = self._generate_digit_image(label)
        
        if self.transform:
            img = self.transform(img)
            
        return img, label

if __name__ == "__main__":
    # 模块单独测试运行逻辑
    dataset = SyntheticDigitDataset(num_samples=5)
    for i in range(5):
        img, label = dataset[i]
        print(f"Sample {i}: Label {label}, Size {img.size}, Mode {img.mode}")
        # 测试时可保存样例：img.save(f"test_digit_{i}_{label}.png")

class SyntheticTextDataset(Dataset):
    """
    (V2 专用版) 动态生成长条状纯数字文本的 PyTorch 数据集。
    尺寸固定为 32x256。内容为变长的数字序列组合。
    序列生成后将在图像内部居左对齐，右侧以白色背景留白 (Padded)。
    """
    def __init__(self, num_samples=10000, img_size=(256, 32), min_len=3, max_len=8, transform=None):
        self.num_samples = num_samples
        self.img_size = img_size # W, H
        self.min_len = min_len
        self.max_len = max_len
        self.transform = transform
        
        self.font_path = "C:/Windows/Fonts/arial.ttf"
        if not os.path.exists(self.font_path):
            self.font_path = "arial.ttf"
            
    def _generate_text_image(self, text):
        img = Image.new('RGB', self.img_size, color=(255, 255, 255))
        draw = ImageDraw.Draw(img)
        
        # 为了能够在狭长的 32px 高度内容纳，字号通常需要受限。
        font_size = random.randint(20, 26)
        try:
            font = ImageFont.truetype(self.font_path, font_size)
        except IOError:
            font = ImageFont.load_default()
            
        color_val = random.randint(0, 50)
        text_color = (color_val, color_val, color_val)
        
        # 计算文本总体尺寸
        bbox = draw.textbbox((0, 0), text, font=font)
        text_height = bbox[3] - bbox[1]
        
        # 居左但留一定微小随机 padding
        x_offset = random.randint(2, 10)
        # 上下居中加入微小震荡
        y_offset = (self.img_size[1] - text_height) // 2 + random.randint(-2, 2)
        y_offset = max(0, min(self.img_size[1] - text_height, y_offset))
        
        draw.text((x_offset, y_offset), text, fill=text_color, font=font)
        
        return img.convert('L')

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # 随机决议本条文本的长度
        length = random.randint(self.min_len, self.max_len)
        # 产生随机连续字符例如 "01239"
        text = "".join([str(random.randint(0, 9)) for _ in range(length)])
        
        img = self._generate_text_image(text)
        
        if self.transform:
            img = self.transform(img)
            
        # 必须返回图片与原始文本标签序列，以便让 CTC 计算对齐
        # PyTorch 的 CTCLoss 需要将 Target 处理成分体数字并给出 target_lengths
        # 这可以在 dataloader 的 collate_fn 或是训练代码里处理。
        # 这里仅仅输出原本的 text string 和 length
        return img, text

class SemanticTextDataset(SyntheticTextDataset):
    """
    (V3 专用版) 继承自 V2 定长图片集基类。
    为了让 Seq2Seq+Attention 体现出其“语言模型(Language Modeling)”特性，不仅生成随机数字，
    还会按照一定概率生成具有严谨格式的伪语义字符串（例如带各种分隔符的组合）：
    - 伪车牌格式: "A1234B"
    - 日期格式: "2026-03-09"
    - 带分隔符: "135-246"
    字典将扩大包含 `-`, `:`, 及部分英文字母。
    """
    def __init__(self, num_samples=10000, img_size=(256, 32), transform=None):
        # 限制长度不要太长以适应 256
        super().__init__(num_samples, img_size, min_len=4, max_len=10, transform=transform)
        # V3 字符字典表 (13个非数字类字符 + 10个数字 + 特殊标记在处理代码里做)
        self.char_list = list("0123456789-:ABC")

    def __getitem__(self, idx):
        # 随机决议本条文本的语义类型
        fmt_type = random.random()
        
        if fmt_type < 0.2:
            # 20%: 短日期格式 YYYY-MM
            y = random.randint(1990, 2030)
            m = random.randint(1, 12)
            text = f"{y}-{m:02d}"
        elif fmt_type < 0.4:
            # 20%: 组合流水对 A001-B
            c1, c2 = random.choice("ABC"), random.choice("ABC")
            num = random.randint(1, 999)
            text = f"{c1}{num:03d}-{c2}"
        elif fmt_type < 0.6:
            # 20%: 时间跨度 12:30
            h = random.randint(0, 23)
            m = random.randint(0, 59)
            text = f"{h:02d}:{m:02d}"
        else:
            # 40%: 完全随机数字/带单破折号
            length = random.randint(4, 9)
            if random.random() < 0.3:
                mid = length // 2
                text = "".join([str(random.randint(0, 9)) for _ in range(mid)]) + "-" + \
                       "".join([str(random.randint(0, 9)) for _ in range(length - mid - 1)])
            else:
                text = "".join([str(random.randint(0, 9)) for _ in range(length)])
                
        img = self._generate_text_image(text)
        
        if self.transform:
            img = self.transform(img)
            
        return img, text
