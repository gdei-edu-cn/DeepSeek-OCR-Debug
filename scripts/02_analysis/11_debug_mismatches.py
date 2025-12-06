# 用于调试OmniDocBench的表格子集的推理结果

import sys
import os
import json
from difflib import SequenceMatcher

# --- 路径引导 ---
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if PROJECT_ROOT not in sys.path: sys.path.insert(0, PROJECT_ROOT)

from src.normalization import normalize_text
from src.taxonomy import guess_type

# --- 配置 ---
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "OmniDocBench", "exp_table_subset")
GT_FILE = os.path.join(DATA_DIR, "table_gt.json")
PRED_VT64 = os.path.join(DATA_DIR, "preds_vt64.json")

def load_data(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    res = {}
    for item in data:
        img = item['image']
        if 'conversations' in item:
            txt = item['conversations'][1]['value']
        elif 'pred' in item:
            txt = item['pred']
        else:
            txt = ""
        res[img] = normalize_text(txt)
    return res

def main():
    print(f"🔍 正在诊断 '冤假错案' (对比 GT 和 vt64)...")
    gt_map = load_data(GT_FILE)
    pred_map = load_data(PRED_VT64)
    
    common_imgs = set(gt_map.keys()) & set(pred_map.keys())
    
    print("\n=== 🔴 Money (金额) 错误抽查 ===")
    print(f"{'图片名':<20} | {'真值 (GT)':<20} | {'预测 (Pred)':<20} | {'类型'}")
    print("-" * 80)
    
    shown_count = 0
    
    for img in common_imgs:
        if shown_count >= 15: break # 只看前 15 个例子
        
        gt_tokens = gt_map[img].split()
        pred_tokens = pred_map[img].split()
        
        matcher = SequenceMatcher(None, gt_tokens, pred_tokens)
        
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag != 'equal': # 只要是不相等的
                # 检查 GT 里被错判的词是不是 Money 或 Number
                for k in range(i1, i2):
                    token = gt_tokens[k]
                    ctype = guess_type(token)
                    
                    if ctype in ['money', 'number']:
                        # 找到对应的预测片段（如果有的话）
                        pred_fragment = " ".join(pred_tokens[j1:j2]) if j2 > j1 else "(丢失/空)"
                        
                        # 打印出来看看
                        print(f"{img[:20]:<20} | {token:<20} | {pred_fragment:<20} | {ctype}")
                        shown_count += 1
                        if shown_count >= 15: break
                if shown_count >= 15: break

    print("-" * 80)
    print("【诊断指南】")
    print("1. 格式差异？(如: GT='1,000' vs Pred='1000') -> 这叫误判，需要改 normalization.py")
    print("2. 乱码/空？(如: GT='1,000' vs Pred='(丢失/空)' 或 'the') -> 这是真错，说明模型瞎了。")

if __name__ == "__main__":
    main()