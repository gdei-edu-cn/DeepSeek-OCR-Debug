# 会统计 GT（标准答案）里一共有多少个单词、多少个数字，用来做分母。

import sys
import os
import json
import collections

# --- 1. 路径引导 (必须加) ---
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.taxonomy import guess_type  # 复用你的核心分类逻辑
from transformers import AutoTokenizer

# --- 2. 配置 ---
# 确保这里的路径指向你的 GT json 文件
GT_PATH = os.path.join(PROJECT_ROOT, "data", "Fox", "exp_fox100", "en_page_ocr_100.json")
# 指向 DeepSeek tokenizer 的位置 (如果找不到，脚本会自动下载或你可以指定本地路径)
TOKENIZER_PATH = os.path.join(PROJECT_ROOT, "3rdparty", "deepseek_ocr") 

def tokenize_text(text):
    # 简单的空白分词，为了和对齐脚本保持一致
    # 如果你之前的对齐用了 tokenizer，这里最好也简单 split 即可，
    # 因为我们是统计“类型”，正则对单词级生效更好。
    import re
    # 清理空白
    text = text.replace("\n", " ").strip()
    # 按空格切分
    return [t for t in re.split(r"\s+", text) if t]

def main():
    if not os.path.exists(GT_PATH):
        print(f"❌ 找不到 GT 文件: {GT_PATH}")
        return

    print(f"正在统计 GT 分母: {GT_PATH} ...")
    
    # 读取 GT
    with open(GT_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 统计器
    gt_counts = collections.Counter()
    total_pages = 0

    for entry in data:
        total_pages += 1
        # DeepSeek 格式: conversations[1]['value'] 是输出
        if "conversations" in entry:
            gt_text = entry["conversations"][1]["value"]
        else:
            continue # 格式不对跳过
        
        # 分词
        tokens = tokenize_text(gt_text)
        
        # 分类
        for tok in tokens:
            t = guess_type(tok)
            gt_counts[t] += 1

    print("\n=== GT (真值) 类型统计 ===")
    print(f"总页数: {total_pages}")
    print(f"{'Type':<15} | {'Total Count (Denominator)':<25}")
    print("-" * 45)
    
    for t, count in gt_counts.most_common():
        print(f"{t:<15} | {count:<25}")

    # 保存结果以便后续画图
    out_path = os.path.join(PROJECT_ROOT, "results", "reports", "fox100_gt_counts.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(gt_counts, f, ensure_ascii=False, indent=2)
    print(f"\n✅ 统计完成，已保存至: {out_path}")

if __name__ == "__main__":
    main()