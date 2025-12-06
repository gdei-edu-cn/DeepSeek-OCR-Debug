# 用于最终评估所有模型的结果
import sys
import os
import json
import collections
import re
import unicodedata
from difflib import SequenceMatcher

# --- 路径配置 ---
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if PROJECT_ROOT not in sys.path: sys.path.insert(0, PROJECT_ROOT)

# --- 1. 统一的清洗逻辑 (Normalization) ---
def normalize_text(text):
    if text is None: return ""
    text = str(text)
    # 全角转半角
    text = unicodedata.normalize('NFKC', text)
    # 暴力清洗：去逗号、去货币符号 (为了公平比较数字)
    text = text.replace(",", "")
    for char in "$￥€£":
        text = text.replace(char, "")
    # 去除表格符
    text = text.replace("|", " ").replace("-", " ")
    # 压缩空白
    text = re.sub(r'\s+', ' ', text).strip()
    return text.lower()

# --- 2. 统一的类型判断 (Taxonomy) ---
def guess_type(token):
    token = token.strip()
    if not token: return 'other'
    # 纯数字 (允许小数点和负号)
    if re.match(r'^-?\d+(\.\d+)?%?$', token):
        return 'number'
    # 单词 (字母开头)
    if re.match(r'^[a-z\u4e00-\u9fa5]', token):
        return 'word'
    # 数学符号
    if token in ['=', '+', '-', '*', '/', '>', '<', '≥', '≤', '≈']:
        return 'math_symbol'
    return 'other'

# --- 3. 核心测评函数 ---
def evaluate_dataset(name, gt_path, preds_dict):
    print(f"\n{'='*20} 正在测评数据集: {name} {'='*20}")
    
    # 加载 GT
    if not os.path.exists(gt_path):
        print(f"❌ 找不到 GT 文件: {gt_path}")
        return
    
    with open(gt_path, 'r', encoding='utf-8') as f:
        gt_data = json.load(f)
    
    gt_map = {}
    for item in gt_data:
        # 兼容两种 GT 格式
        if 'conversations' in item:
            txt = item['conversations'][1]['value']
        else:
            txt = item.get('text', '')
        gt_map[item['image']] = normalize_text(txt)

    # 遍历每种模型配置 (vt64, vt100, Dynamic)
    for model_name, pred_path in preds_dict.items():
        if not os.path.exists(pred_path):
            print(f"⚠️  缺文件: {model_name} ({pred_path})")
            continue
            
        with open(pred_path, 'r', encoding='utf-8') as f:
            pred_data = json.load(f)
        
        pred_map = {item['image']: normalize_text(item.get('pred', '')) for item in pred_data}
        
        # 开始计算
        gt_counter = collections.Counter()
        err_counter = collections.Counter()
        
        common_imgs = set(gt_map.keys()) & set(pred_map.keys())
        
        for img in common_imgs:
            gt_tokens = gt_map[img].split()
            pred_tokens = pred_map[img].split()
            
            # 统计 GT 分母
            for t in gt_tokens: 
                gt_counter[guess_type(t)] += 1
            
            # 统计错误分子
            matcher = SequenceMatcher(None, gt_tokens, pred_tokens)
            for tag, i1, i2, j1, j2 in matcher.get_opcodes():
                if tag != 'equal':
                    for k in range(i1, i2):
                        err_counter[guess_type(gt_tokens[k])] += 1
        
        # 输出该模型的结果
        print(f"\n--- 配置: {model_name} ---")
        print(f"{'Type':<10} | {'GT Count':<8} | {'Errors':<8} | {'Error Rate'}")
        print("-" * 45)
        for t in ['number', 'word']:
            total = gt_counter[t]
            err = err_counter[t]
            rate = (err / total * 100) if total > 0 else 0.0
            print(f"{t:<10} | {total:<8} | {err:<8} | {rate:.1f}%")

# --- 4. 主执行逻辑 ---
if __name__ == "__main__":
    # Fox-100 配置
    FOX_GT = os.path.join(PROJECT_ROOT, "data/Fox/exp_fox100/en_page_ocr_100.json")
    FOX_PREDS = {
        "vt64": os.path.join(PROJECT_ROOT, "data/Fox/exp_fox100/preds_vt64.json"),
        "vt100": os.path.join(PROJECT_ROOT, "data/Fox/exp_fox100/preds_vt100.json"),
        "Dynamic": os.path.join(PROJECT_ROOT, "data/Fox/exp_fox100/preds_dynamic.json")
    }
    
    # OmniDoc 配置
    OMNI_GT = os.path.join(PROJECT_ROOT, "data/OmniDocBench/exp_table_subset/table_gt.json")
    OMNI_PREDS = {
        "vt64": os.path.join(PROJECT_ROOT, "data/OmniDocBench/exp_table_subset/preds_vt64.json"),
        "vt100": os.path.join(PROJECT_ROOT, "data/OmniDocBench/exp_table_subset/preds_vt100.json"),
        "Dynamic": os.path.join(PROJECT_ROOT, "data/OmniDocBench/exp_table_subset/preds_dynamic.json")
    }
    
    evaluate_dataset("Fox-100 (文档场景)", FOX_GT, FOX_PREDS)
    evaluate_dataset("OmniDocBench (表格场景)", OMNI_GT, OMNI_PREDS)