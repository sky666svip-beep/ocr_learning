import cv2
import numpy as np

def unified_enhance_image(image_pil):
    """
    [V1-V5 统一推断层护城河]
    应对真实环境下图片模糊、对比度低、噪点密布等 Domain Shift 现象。
    经过此管线后，返回高度对比、干净锐利的等尺寸 PIL Image 供下游模型继续处理。
    """
    # 1. 转为通用 RGB OpenCV 格式以兼容所有上传格式
    img_cv = np.array(image_pil.convert('RGB'))
    
    # 2. 灰度化同化
    gray_img = cv2.cvtColor(img_cv, cv2.COLOR_RGB2GRAY)
    
    # 3. 中值特征降噪 (专门抹除胡椒小黑点噪波，但保留字体实质锐度边缘)
    denoised = cv2.medianBlur(gray_img, 3)
    
    # 4. CLAHE 自适应直方图均衡，拔亮暗部，压制过曝阴阳脸
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(denoised)
    
    # 默认认为文字是黑字白底，若检测到反色也可在此强制 Invert
    # 为保证和训练时高度一致，我们返回 L 模式图像
    from PIL import Image
    return Image.fromarray(enhanced).convert('L')

def process_image(image_pil):
    """
    对输入 PIL 图片进行预处理：转灰度 -> 二值化 -> 闭操作去噪补洞 -> 垂直投影分割
    返回 4 个核心要素以便分析与可视化：
    1. gray_img (np.array 灰度图片矩阵)
    2. processed_img (np.array 形态学修复后的黑白图，背景纯黑，文字部分纯白)
    3. projection (np.array 纯1D数组表征当前垂向总白色像素强度)
    4. bounding_boxes (List[Tuple] 即 [(x1, y1, x2, y2), ...] 文字切图定位依据)
    """
    # 1. 转为 NumPy OpenCV 格式，通常是从前端来的 RGB 或者 RGBA
    img_cv = np.array(image_pil.convert('RGB'))
    
    # 2. 灰度化
    gray_img = cv2.cvtColor(img_cv, cv2.COLOR_RGB2GRAY)
    
    # 3. 二值化 (Otsu 方法自动寻找阈值)。
    # 注意 OCR 经典处理：需将前景(文字)变为白色(255)，背景变黑色(0)以方便计算信号。
    # 原始图像如果在白纸上黑字，需要 THRESH_BINARY_INV。
    ret, binary_img = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # 4. 形态学闭操作 (先膨胀后腐蚀)，连接可能断裂的笔画(如虚线数字)，消除黑白噪点
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    # 对于较好的渲染数字，甚至可以直接进行一遍膨胀，让数字更连贯以对抗粘连测试
    processed_img = cv2.morphologyEx(binary_img, cv2.MORPH_CLOSE, kernel)
    
    # 5. 垂直投影
    # 将每一列的白色像素加和起来，构成 1D 直方图数据
    projection = np.sum(processed_img, axis=0)  # shape (width, )
    
    # 6. 基于投影进行切割 (核心寻峰寻谷逻辑)
    # 当一列投影值低于某个阈值时，认为处于字符间隙
    bounding_boxes = []
    threshold = 5 * 255 # 容忍列存在少量散落像素 (如最大 5 倍噪点)
    
    in_char = False
    start_x = 0
    height, width = processed_img.shape
    
    # 扫描每一列
    for x in range(width):
        val = projection[x]
        if not in_char and val > threshold:
            in_char = True
            start_x = x
        elif in_char and val <= threshold:
            in_char = False
            # 防御过小的噪点框：如果字符宽度不够，抛弃
            if x - start_x > 5:
                # 寻找这个 X 区间的 Y 方向确切边界（水平投影剔除上下留白）
                char_strip = processed_img[:, start_x:x]
                horizontal_proj = np.sum(char_strip, axis=1)
                
                y_indices = np.where(horizontal_proj > 0)[0]
                if len(y_indices) > 0:
                    y_start = max(0, y_indices[0] - 2) # 上下给2个像素缓冲
                    y_end = min(height, y_indices[-1] + 2)
                    bounding_boxes.append((start_x, y_start, x, y_end))
                    
    # 处理字符贴住右边界的边界条件
    if in_char:
        x = width - 1
        if x - start_x > 5:
            char_strip = processed_img[:, start_x:x]
            horizontal_proj = np.sum(char_strip, axis=1)
            y_indices = np.where(horizontal_proj > 0)[0]
            if len(y_indices) > 0:
                y_start = max(0, y_indices[0] - 2)
                y_end = min(height, y_indices[-1] + 2)
                bounding_boxes.append((start_x, y_start, x, y_end))

    return gray_img, processed_img, projection, bounding_boxes

def extract_character_patches(gray_img, bounding_boxes, target_size=(32, 32)):
    """
    依据 Bounding_boxes 从原图取片，以供 CNN 进行批处理推理。
    输入为 黑字白底的 gray_img (这也是原始图转化出的等价数据)，
    由于我们的 CNN 是在 torchvision (Invert处理后) 下训练的，
    这里切片直接截取原图灰度。在后续模型推理时同样需要经过 transform Invert。
    """
    patches = []
    # 这里我们返回 PIL Image 对象集合，便于复用训练时的 Transform Pipeline
    from PIL import Image
    for bbox in bounding_boxes:
        x1, y1, x2, y2 = bbox
        # 裁剪出灰度局部区域 (仍然是白底黑字)
        patch_array = gray_img[y1:y2, x1:x2]
        # 如果 patch 长宽比太极端，应该填充成正方形，避免拉伸导致变形
        h, w = patch_array.shape
        max_side = max(h, w)
        # 【BUG FIX】CNN 训练数据是包含显著四周边距的（字号20~28，画布32）
        # 如果这里切出来紧贴边缘并拉伸满 32x32，CNN 将失去特征参照物完全瞎猜。
        # 因此我们必须为切片补充 Margin 背景！
        padding = int(max_side * 0.25)
        bg_size = max_side + 2 * padding
        
        # 用 255 (白色) 填充背景
        square_patch = np.full((bg_size, bg_size), 255, dtype=np.uint8)
        
        # 将原始 patch 放置在正中间
        y_offset = padding + (max_side - h) // 2
        x_offset = padding + (max_side - w) // 2
        square_patch[y_offset:y_offset+h, x_offset:x_offset+w] = patch_array
        
        # 转换并 Resize。Resize 会稍微改变像素抗锯齿度，采用 PIL LANCZOS
        patch_img = Image.fromarray(square_patch).resize(target_size, Image.Resampling.LANCZOS)
        patches.append(patch_img)
        
    return patches
