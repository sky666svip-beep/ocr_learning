import cv2
import numpy as np

def unified_enhance_image(image_pil, apply_skew_correction=False):
    """
    [V1-V5 统一推断层护城河]
    应对真实环境下图片模糊、对比度低、噪点密布等 Domain Shift 现象。
    经过此管线后，返回高度对比、干净锐利的等尺寸 PIL Image 供下游模型继续处理。
    """
    # 1. 转为通用 RGB OpenCV 格式以兼容所有上传格式
    img_cv = np.array(image_pil.convert('RGB'))
    
    # 2. 灰度化同化
    gray_img = cv2.cvtColor(img_cv, cv2.COLOR_RGB2GRAY)
    
    skew_angle = 0.0
    debug_img_before_skew = None
    
    if apply_skew_correction:
        # 记录原始用于 debug
        debug_img_before_skew = img_cv.copy()
        # 使用边缘检测寻找直线
        edges = cv2.Canny(gray_img, 50, 150, apertureSize=3)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=100, maxLineGap=10)
        
        angles = []
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                if x2 - x1 != 0:
                    angle = np.degrees(np.arctan((y2 - y1) / (x2 - x1)))
                    # 过滤掉接近垂直的线，只看水平走向的文本行倾斜
                    if -45 <= angle <= 45:
                        angles.append(angle)
        
        # 另一种求倾斜的主流方式是用所有文字像素点的外接矩形 (MinAreaRect)
        # 结合霍夫更稳定。这里如果跑不出直线，用 MinAreaRect 兜底
        if len(angles) > 0:
            skew_angle = np.median(angles)
        else:
            coords = np.column_stack(np.where(gray_img < 128)) # 假设黑字白底寻找坐标
            if len(coords) > 0:
                angle = cv2.minAreaRect(coords)[-1]
                if angle < -45:
                    angle = -(90 + angle)
                else:
                    angle = -angle
                if -45 <= angle <= 45:
                    skew_angle = angle
        
        if abs(skew_angle) > 0.5: # 倾斜大于 0.5 度才发生旋转
            h, w = gray_img.shape
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, skew_angle, 1.0)
            # 采用 BORDER_REPLICATE 或者白色 (255) 填充
            gray_img = cv2.warpAffine(gray_img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=255)
            # 把原彩图也转一下供展示用
            debug_img_before_skew = img_cv.copy() 
    
    # 3. 中值特征降噪 (专门抹除胡椒小黑点噪波，但保留字体实质锐度边缘)
    denoised = cv2.medianBlur(gray_img, 3)
    
    # 4. CLAHE 自适应直方图均衡，拔亮暗部，压制过曝阴阳脸
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(denoised)
    
    # 默认认为文字是黑字白底，若检测到反色也可在此强制 Invert
    # 为保证和训练时高度一致，我们返回 L 模式图像
    from PIL import Image
    
    result = {
        'image': Image.fromarray(enhanced).convert('L'),
        'skew_angle': skew_angle,
        'debug_img_before_skew': debug_img_before_skew,
        'debug_img_after_skew': gray_img if apply_skew_correction else None
    }
    return result

def extract_text_lines(image_pil):
    """
    [V2-V5 前置切割组件]
    接收经过 unified_enhance_image 处理过的 L 模式 PIL Image (黑字白底或白字黑底的强化图)。
    1. 反色二值化处理 (变成黑底白字提取信号)
    2. 最大连通域边界提取 (Connect Component BBox)，过滤杂色微小孤立噪点
    3. 水平投影 (Horizontal Projection) 切割多行文本
    返回: List[PIL.Image] 行图像切片数组 (高度已自适应裁剪)
    """
    img_np = np.array(image_pil)
    
    # 因为输入进来的是经过 CLAHE 的灰度图，我们需要先 Otsu 二值化提取信号
    # 假设背景偏亮(白纸黑字)，反向二值化得到 黑底白字
    _, binary_inv = cv2.threshold(img_cv if 'img_cv' in locals() else img_np, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # 改善小噪点 - 求所有轮廓
    contours, _ = cv2.findContours(binary_inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return [image_pil] # 没找到，原样退回
        
    # 过滤掉极小的噪点 (例如面积小于整图面积的千分之一，或者固定很小的绝对面积)
    min_area = (img_np.shape[0] * img_np.shape[1]) * 0.001 
    valid_contours = [c for c in contours if cv2.contourArea(c) > min_area]
    
    # 若被过滤的太狠一个不剩，则放宽标准降级处理
    if not valid_contours:
        valid_contours = [c for c in contours if cv2.contourArea(c) > 5]
        
    if not valid_contours: # 实在没有，退回
        return [image_pil]
        
    # 仅在有效轮廓的掩码下进行水平投影
    mask = np.zeros_like(binary_inv)
    cv2.drawContours(mask, valid_contours, -1, 255, thickness=cv2.FILLED)
    
    # 利用投影拆分多行 (Horizontal Projection)
    horizontal_proj = np.sum(mask, axis=1) # shape: (H,)
    
    # 寻谷逻辑，按行切割
    row_boxes = []
    in_text_line = False
    start_y = 0
    
    # 允许行间有一两个噪点像素不断，设定一个行间隔容忍阈值
    row_threshold = 255 * 5 # 一行少于5个高亮像素视为间隙
    
    for y, val in enumerate(horizontal_proj):
        if not in_text_line and val > row_threshold:
            in_text_line = True
            start_y = y
        elif in_text_line and val <= row_threshold:
            in_text_line = False
            # 过滤高度太矮的面条框 (噪点线)
            if y - start_y > 8:
                row_boxes.append((start_y, y))
                
    if in_text_line and (horizontal_proj.shape[0] - start_y > 8):
        row_boxes.append((start_y, horizontal_proj.shape[0]))
        
    # 如果没切出行，就整体框一下最大边界
    if not row_boxes:
         x, y, w, h = cv2.boundingRect(np.vstack(valid_contours))
         row_boxes.append((y, y+h))
         
    # 根据 Y 轴行段，精确抠取每一行的 X 轴边界
    line_images = []
    from PIL import Image
    margin = 5
    for (y_start, y_end) in row_boxes:
        # 截取该行的掩码
        row_mask = mask[y_start:y_end, :]
        # 寻找该行的有效 X 边界
        coords = cv2.findNonZero(row_mask)
        if coords is not None:
            x, _, w, _ = cv2.boundingRect(coords)
            
            crop_y_s = max(0, y_start - margin)
            crop_y_e = min(img_np.shape[0], y_end + margin)
            crop_x_s = max(0, x - margin)
            crop_x_e = min(img_np.shape[1], x + w + margin)
            
            line_img = image_pil.crop((crop_x_s, crop_y_s, crop_x_e, crop_y_e))
            line_images.append(line_img)
            
    if not line_images:
        return [image_pil]
        
    return line_images

def process_image(image_pil, use_adaptive_thresh=False, block_size=11, C=2, use_heuristic_split=False):
    """
    对输入 PIL 图片进行预处理：转灰度 -> 二值化 -> 闭操作去噪补洞 -> 垂直投影分割
    返回核心要素以便分析与可视化：
    1. gray_img (np.array 灰度图片矩阵)
    2. processed_img (np.array 形态学修复后的黑白图，背景纯黑，文字部分纯白)
    3. projection (np.array 纯1D数组表征当前垂向总白色像素强度)
    4. bounding_boxes (List[Tuple] 即 [(x1, y1, x2, y2), ...] 文字切图定位依据)
    5. heuristic_lines (List[Tuple] 即 [(x_cut, y_start, y_end), ...] 启发式强切线记录)
    """
    # 1. 转为 NumPy OpenCV 格式，通常是从前端来的 RGB 或者 RGBA
    img_cv = np.array(image_pil.convert('RGB'))
    
    # 2. 灰度化
    gray_img = cv2.cvtColor(img_cv, cv2.COLOR_RGB2GRAY)
    
    # 3. 二值化
    if use_adaptive_thresh:
        # 确保 block_size 是奇数
        if block_size % 2 == 0:
            block_size += 1
        # 自适应阈值，返回的是白底黑字（假设输入多为白纸黑字），我们要黑底白字提取信号
        binary_inv = cv2.adaptiveThreshold(gray_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, block_size, C)
        binary_img = binary_inv
    else:
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
                    
    heuristic_lines = []
    
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

    # 7. 基于粘连检测进行启发式强切 (Heuristic Water Drop Split)
    if use_heuristic_split:
        final_boxes = []
        # 预估字符正常宽高比约为 0.5 到 1.0 (根据字体不同)
        # 如果宽度远大于高度的 1.2 倍，强烈怀疑是粘连数字
        for box in bounding_boxes:
            bx1, by1, bx2, by2 = box
            bh = by2 - by1
            bw = bx2 - bx1
            if bw > bh * 1.2 and bw > 15: # 怀疑有两个或以上字符粘连
                # 寻找上下轮廓最深的凹陷进行对其连线切开
                # 只在中间 60% 区域寻找，防止切到边缘
                search_x1 = bx1 + int(bw * 0.2)
                search_x2 = bx2 - int(bw * 0.2)
                
                # 计算顶部轮廓 (从上往下第一个白色像素的 Y 坐标)
                top_contour = []
                for cx in range(search_x1, search_x2):
                    col = processed_img[by1:by2, cx]
                    white_ys = np.where(col > 0)[0]
                    if len(white_ys) > 0:
                        top_contour.append((cx, white_ys[0]))
                    else:
                        top_contour.append((cx, by2-by1)) # 如果全是黑的，抛到底部

                # 计算底部轮廓 (从下往上第一个白色像素的 Y 坐标)
                bottom_contour = []
                for cx in range(search_x1, search_x2):
                    col = processed_img[by1:by2, cx]
                    white_ys = np.where(col > 0)[0]
                    if len(white_ys) > 0:
                        bottom_contour.append((cx, (by2-by1) - white_ys[-1])) # 距离底部的距离
                    else:
                        bottom_contour.append((cx, by2-by1))
                        
                # 分别找 top 和 bottom contour depth 最深的点（y 值最大）
                # 为了启发式强切，我们直接选择上下轮廓“最靠近彼此的那个 X 候选点”
                if len(top_contour) > 0 and len(bottom_contour) > 0:
                    min_thickness = bh
                    best_cut_x = search_x1 + (search_x2 - search_x1) // 2 # 兜底切中间
                    
                    for i, cx in enumerate(range(search_x1, search_x2)):
                         thickness = top_contour[i][1] + bottom_contour[i][1] # 当前列空白的厚度
                         actual_ink_thickness = bh - thickness
                         if actual_ink_thickness < min_thickness and actual_ink_thickness > 0:
                             min_thickness = actual_ink_thickness
                             best_cut_x = cx
                    
                    # 执行撕裂一分为二
                    final_boxes.append((bx1, by1, best_cut_x, by2))
                    final_boxes.append((best_cut_x, by1, bx2, by2))
                    heuristic_lines.append((best_cut_x, by1, by2))
                else:
                    final_boxes.append(box)
            else:
                 final_boxes.append(box)
        bounding_boxes = final_boxes

    return gray_img, processed_img, projection, bounding_boxes, heuristic_lines

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
