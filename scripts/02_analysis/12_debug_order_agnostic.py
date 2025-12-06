# 用于调试顺序无关的数字召回率
import sys
import os
import json
import collections
import re
import unicodedata

# --- 路径引导 ---
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if PROJECT_ROOT not in sys.path: sys.path.insert(0, PROJECT_ROOT)

# --- 配置 ---
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "OmniDocBench", "exp_table_subset")
GT_FILE = os.path.join(DATA_DIR, "table_gt.json")

# 原始 vt64 结果
PRED_VT64 = os.path.join(DATA_DIR, "preds_vt64.json")
# 【新增】动态路由结果
PRED_DYNAMIC = os.path.join(DATA_DIR, "preds_dynamic.json")

def normalize_for_set(text):
    """暴力清洗，只留数字和小数点，忽略所有分隔符"""
    if text is None: return ""
    text = str(text)
    text = unicodedata.normalize('NFKC', text)
    text = text.replace(",", "")
    numbers = re.findall(r'-?\d+\.?\d*', text)
    return numbers

def load_number_bag(path):
    if not os.path.exists(path):
        print(f"⚠️ 文件不存在: {path}")
        return {}
        
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
        
        nums = normalize_for_set(txt)
        res[img] = collections.Counter(nums)
    return res

def evaluate_recall(gt_bags, pred_bags, name):
    print(f"\n=== 🟢 {name} (忽略顺序，只看数字是否召回) ===")
    
    if not pred_bags:
        print("  (无数据)")
        return 1.0

    total_gt_nums = 0
    correct_found = 0
    
    common_imgs = set(gt_bags.keys()) & set(pred_bags.keys())
    
    for img in common_imgs:
        gt_counter = gt_bags[img]
        pred_counter = pred_bags[img]
        
        total_gt_nums += sum(gt_counter.values())
        
        # 计算交集（找回了多少个）
        intersection = gt_counter & pred_counter
        correct_found += sum(intersection.values())

    if total_gt_nums == 0:
        print("GT 里没有找到数字？")
        return 0.0

    recall = correct_found / total_gt_nums
    miss_rate = 1 - recall
    
    print(f"GT 数字总数: {total_gt_nums}")
    print(f"找回的数字: {correct_found}")
    print(f"数字召回率 (Recall): {recall:.2%}")
    print(f"数字丢失率 (Miss Rate): {miss_rate:.2%}")
    
    return miss_rate

def main():
    print("正在加载数据...")
    gt_bags = load_number_bag(GT_FILE)
    
    # 1. 评估 vt64 (基线)
    pred64_bags = load_number_bag(PRED_VT64)
    evaluate_recall(gt_bags, pred64_bags, "vt64 (Baseline)")
    
    # 2. 评估 Dynamic (你的方案)
    pred_dynamic_bags = load_number_bag(PRED_DYNAMIC)
    evaluate_recall(gt_bags, pred_dynamic_bags, "Dynamic Routing (Your Solution)")

if __name__ == "__main__":
    main()