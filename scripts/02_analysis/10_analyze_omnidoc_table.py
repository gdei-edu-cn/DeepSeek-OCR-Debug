# 用于分析OmniDocBench的表格子集的推理结果
import sys
import os
import json
import collections
from difflib import SequenceMatcher

# --- 1. 路径引导 ---
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if PROJECT_ROOT not in sys.path: sys.path.insert(0, PROJECT_ROOT)

from src.normalization import normalize_text
from src.taxonomy import guess_type

# --- 2. 配置 ---
# OmniDocBench 的：
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "OmniDocBench", "exp_table_subset")
GT_FILE = os.path.join(DATA_DIR, "table_gt.json")

# 改成 Fox 的：
# DATA_DIR = os.path.join(PROJECT_ROOT, "data", "Fox", "exp_fox100")
# GT_FILE = os.path.join(DATA_DIR, "en_page_ocr_100.json")

# 修改前
# PRED_VT64 = os.path.join(DATA_DIR, "preds_vt64.json")

# 修改后,使用dynamic routing的结果(omnidocbench)
PRED_VT64 = os.path.join(DATA_DIR, "preds_vt64.json")
PRED_VT100 = os.path.join(DATA_DIR, "preds_vt100.json")

def load_json_map(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    # 兼容两种格式：一种是 list of dict，一种是 Fox 的 conversations 格式
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

def get_errors_and_gt(gt_map, pred_map):
    """同时计算错误分子和 GT 分母"""
    gt_counter = collections.Counter()
    err_counter = collections.Counter()
    
    # 共同的图片
    common_imgs = set(gt_map.keys()) & set(pred_map.keys())
    print(f"   分析样本量: {len(common_imgs)} 页")

    for img in common_imgs:
        gt_tokens = gt_map[img].split()
        pred_tokens = pred_map[img].split()
        
        # 1. 统计分母 (GT)
        for t in gt_tokens:
            gt_counter[guess_type(t)] += 1
            
        # 2. 对齐并统计分子 (Error)
        matcher = SequenceMatcher(None, gt_tokens, pred_tokens)
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag != 'equal':
                # GT 区间 [i1, i2) 的 token 被错认/删除/替换了
                for k in range(i1, i2):
                    tok_type = guess_type(gt_tokens[k])
                    err_counter[tok_type] += 1
                
                # 如果是纯插入 (insert)，分母里没有，通常不计入 Type Error Rate 的分母
                # 但为了严谨，这里主要关注 GT 里的东西丢没丢 (Recall-oriented)
                
    return gt_counter, err_counter

def print_report(name, gt_counts, err_counts):
    print(f"\n=== {name} 核心指标 (Section 5 素材) ===")
    print(f"{'Type':<12} | {'GT Count':<10} | {'Errors':<10} | {'Error Rate (%)':<15}")
    print("-" * 55)
    
    # 我们只关心核心类型
    focus_types = ['number', 'word', 'math_symbol', 'money']
    
    for t in focus_types:
        total = gt_counts[t]
        errs = err_counts[t]
        rate = (errs / total * 100) if total > 0 else 0.0
        print(f"{t:<12} | {total:<10} | {errs:<10} | {rate:<15.2f}%")

def main():
    if not os.path.exists(GT_FILE):
        print(f"❌ 缺文件: {GT_FILE} (请检查 build_omnidoc_table.py 是否成功)")
        return

    print("正在加载数据...")
    gt_map = load_json_map(GT_FILE)
    pred_64 = load_json_map(PRED_VT64)
    pred_100 = load_json_map(PRED_VT100)
    
    # 分析 vt64
    print("\n正在分析 vt64 (高压缩)...")
    gt_counts, err_64 = get_errors_and_gt(gt_map, pred_64)
    print_report("vt64 (Table Subset)", gt_counts, err_64)
    
    # 分析 vt100
    print("\n正在分析 vt100 (中等压缩)...")
    _, err_100 = get_errors_and_gt(gt_map, pred_100) # GT 分母是一样的
    print_report("vt100 (Table Subset)", gt_counts, err_100)

if __name__ == "__main__":
    main()