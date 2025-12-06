# 用于验证样本的正确性
import sys
import os
import json
import random
from difflib import SequenceMatcher

# --- 1. 路径引导 ---
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if PROJECT_ROOT not in sys.path: sys.path.insert(0, PROJECT_ROOT)

# 引用你之前的清洗逻辑，确保标准一致
from src.normalization import normalize_text
from src.taxonomy import guess_type

# --- 2. 配置 ---
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "OmniDocBench", "exp_table_subset")
GT_FILE = os.path.join(DATA_DIR, "table_gt.json")
PRED_VT64 = os.path.join(DATA_DIR, "preds_vt64.json")
IMG_DIR = os.path.join(DATA_DIR, "images")

def load_data(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    res = {}
    raw_res = {} # 保留原始文本方便对比
    for item in data:
        img = item['image']
        if 'conversations' in item:
            txt = item['conversations'][1]['value']
        elif 'pred' in item:
            txt = item['pred']
        else:
            txt = ""
        res[img] = normalize_text(txt) # 这里的 normalize_text 就是你之前改过的“暴力清洗版”
        raw_res[img] = txt
    return res, raw_res

def main():
    print(f"🕵️‍♂️ 正在启动肉眼验证程序...")
    gt_map, gt_raw = load_data(GT_FILE)
    pred_map, pred_raw = load_data(PRED_VT64)
    
    common_imgs = list(set(gt_map.keys()) & set(pred_map.keys()))
    
    # 随机抽 3 张图，或者你可以指定特定的图片名
    # selected_imgs = random.sample(common_imgs, 3) 
    # 为了复现，我们选前 3 张有数字错误的图
    
    shown_images = 0
    for img in common_imgs:
        if shown_images >= 3: break
        
        print(f"\n{'='*20} 图片: {img} {'='*20}")
        print(f"👉 请打开: {os.path.join(IMG_DIR, img)}")
        
        gt_tokens = gt_map[img].split()
        pred_tokens = pred_map[img].split()
        
        # 使用和评测脚本完全一样的对齐逻辑
        matcher = SequenceMatcher(None, gt_tokens, pred_tokens)
        
        has_number_error = False
        
        print(f"\n{'[状态]':<6} | {'GT (清洗后)':<20} | {'Pred (清洗后)':<20} | {'说明'}")
        print("-" * 70)
        
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            # 获取这一段的 token
            gt_chunk = gt_tokens[i1:i2]
            pred_chunk = pred_tokens[j1:j2]
            
            if tag == 'equal':
                # 如果完全一样，只打印数字相关的，证明“对的也没判错”
                for t in gt_chunk:
                    if guess_type(t) == 'number':
                        print(f"✅ MATCH  | {t:<20} | {t:<20} | 识别正确")
            
            else:
                # 如果不一样，检查是不是数字错误
                for k, t in enumerate(gt_chunk):
                    if guess_type(t) == 'number':
                        has_number_error = True
                        # 尝试找到对应的错误预测（如果有的话）
                        p_txt = pred_chunk[k] if k < len(pred_chunk) else "---"
                        print(f"❌ ERROR  | {t:<20} | {p_txt:<20} | 判错！")
        
        if has_number_error:
            shown_images += 1
            print("-" * 70)
            print("【原始文本对比 (Raw Text)】")
            # 打印一小段原始文本，让你看看没清洗前长啥样
            print(f"GT 原文片段: {gt_raw[img][:100].replace(chr(10), ' ')}...")
            print(f"Pred 原文片段: {pred_raw[img][:100].replace(chr(10), ' ')}...")
        else:
            print("(这张图没有数字错误，跳过展示)")

if __name__ == "__main__":
    main()