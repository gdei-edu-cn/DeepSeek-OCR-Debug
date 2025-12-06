# 用于计算稳定性测试结果

import sys, os, json
import numpy as np
from difflib import SequenceMatcher

# --- 路径配置 ---
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if PROJECT_ROOT not in sys.path: sys.path.insert(0, PROJECT_ROOT)

from src.normalization import normalize_text
from src.taxonomy import guess_type

DATA_DIR = os.path.join(PROJECT_ROOT, "data", "Fox", "exp_fox100")
STABILITY_DIR = os.path.join(DATA_DIR, "stability")
GT_FILE = os.path.join(DATA_DIR, "en_page_ocr_100.json")

def align_and_check(gt_tokens, pred_tokens):
    matcher = SequenceMatcher(None, gt_tokens, pred_tokens)
    correct_mask = [False] * len(gt_tokens)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == 'equal':
            for k in range(i1, i2): correct_mask[k] = True
    return correct_mask

def main():
    if not os.path.exists(GT_FILE):
        print("❌ 找不到 GT 文件")
        return

    # 读取 GT
    with open(GT_FILE, 'r') as f:
        gt_data = {item['image']: item['conversations'][1]['value'] for item in json.load(f)}

    # 读取 5 次 Runs
    runs = []
    for i in range(1, 6):
        path = os.path.join(STABILITY_DIR, f"preds_run{i}.json")
        if not os.path.exists(path):
            print(f"❌ 缺文件: {path}，请先运行 8_run_stability.py！")
            return
        with open(path, 'r') as f:
            runs.append({item['image']: item['pred'] for item in json.load(f)})
    
    print(f"✅ 加载完成。分析系统性错误...")

    systematic_errors = 0
    random_errors = 0
    total_number_tokens = 0
    
    target_images = list(runs[0].keys())

    for img in target_images:
        if img not in gt_data: continue
        
        gt_text = normalize_text(gt_data[img])
        gt_tokens = gt_text.split()
        
        # 找到所有数字的位置
        number_indices = [i for i, t in enumerate(gt_tokens) if guess_type(t) in ['number', 'money', 'number+unit']]
        total_number_tokens += len(number_indices)
        
        # 统计正确性矩阵
        correctness_matrix = []
        for run_idx in range(5):
            pred_text = normalize_text(runs[run_idx].get(img, ""))
            pred_tokens = pred_text.split()
            mask = align_and_check(gt_tokens, pred_tokens)
            correctness_matrix.append([mask[i] for i in number_indices])
            
        correctness_matrix = np.array(correctness_matrix) # (5, N)
        
        # 计算错误次数
        failures = 5 - np.sum(correctness_matrix, axis=0)
        
        for f in failures:
            if f >= 4: # 5次里错了4-5次 -> 系统性
                systematic_errors += 1
            elif f > 0: # 5次里错了1-3次 -> 随机性
                random_errors += 1

    total_err = systematic_errors + random_errors
    if total_err == 0:
        print("🎉 厉害！这几张图在这几次运行里居然一个数字都没错（不太可能）")
        return

    print("\n=== 🎲 稳定性实验结果 ===")
    print(f"分析页面数: {len(target_images)}")
    print(f"涉及数字总数: {total_number_tokens}")
    print(f"总错误数: {total_err}")
    print("-" * 30)
    print(f"🔴 系统性错误 (Systematic): {systematic_errors} ({systematic_errors/total_err:.1%})")
    print(f"🔵 随机性错误 (Random):     {random_errors} ({random_errors/total_err:.1%})")
    print("-" * 30)

if __name__ == "__main__":
    main()