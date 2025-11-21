# 数据集构建脚本

import sys
import os
# 动态计算项目根目录 (scripts/xx/xx.py -> ../../ -> root)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import json
import re
from transformers import AutoTokenizer


# ====【根据你的实际路径修改这里】====
# 获取数据根目录
FOX_ROOT = os.path.join(PROJECT_ROOT, "data", "Fox")
FOCUS_ROOT = os.path.join(FOX_ROOT, "raw", "focus_benchmark_test")
IMG_DIR = os.path.join(FOCUS_ROOT, "en_pdf_png")
ANN_PATH = os.path.join(FOCUS_ROOT, "en_page_ocr.json")
# ===================================


# 这里填你推理时用的模型路径或 HF 名称
DEEPSEEK_MODEL = os.path.join(PROJECT_ROOT, "3rdparty", "deepseek_ocr")
# ===================================


def step2_read_annotations():
    with open(ANN_PATH, "r", encoding="utf-8") as f:
        anns = json.load(f)
    print("总条目数（总页数）:", len(anns))
    print("示例 keys:", list(anns[0].keys()))
    print("示例 image:", anns[0].get("image", None))
    print("示例 conversations[1]:", anns[0]["conversations"][1])
    return anns

def step3_count_tokens(anns):
    print("\n=== 加载 DeepSeek-OCR tokenizer ===")
    tokenizer = AutoTokenizer.from_pretrained(DEEPSEEK_MODEL)
    print("词表大小 vocab_size:", tokenizer.vocab_size)

    records = []
    for idx, ann in enumerate(anns):
        img_name = ann["image"]
        img_path = os.path.join(IMG_DIR, img_name)

        # 1) 直接拿原始 GT 文本
        gt_text = ann["conversations"][1]["value"]

        # 2) 不做任何规范化，直接送进 tokenizer
        enc = tokenizer(gt_text, add_special_tokens=False)
        n_tok = len(enc.input_ids)

        rec = {
            "idx": idx,
            "image": img_name,
            "img_path": img_path,
            "n_tokens": n_tok,
            "gt_text": gt_text,
        }
        records.append(rec)

        if idx < 3:
            print(f"[样例 {idx}] {img_name}, tokens={n_tok}")

    print("统计完毕，总样本数:", len(records))
    return records



def step4_filter_600_1300(records):
    selected = [r for r in records if 600 <= r["n_tokens"] < 1300]
    print("\n=== Step 4: 过滤 600–1300 tokens 的页面 ===")
    print("满足条件的页数:", len(selected))

    # 打印前几条
    for r in selected[:5]:
        print(f"  {r['image']}: {r['n_tokens']} tokens")

    return selected

def save_selected_list(selected, path=None):
    if path is None:
        path = os.path.join(FOX_ROOT, "exp_fox100", "selected_pages_raw.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    # 只保存 image 名和 token 数等关键信息
    simple = [
        {"image": r["image"], "n_tokens": r["n_tokens"], "idx": r["idx"]}
        for r in sorted(selected, key=lambda x: x["image"])
    ]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(simple, f, ensure_ascii=False, indent=2)
    print("已将 100 页列表保存到:", os.path.abspath(path))

def step5_check_bins(selected):
    print("\n=== Step 5: 按区间统计 ===")
    bins = [
        (600, 700),
        (700, 800),
        (800, 900),
        (900, 1000),
        (1000, 1100),
        (1100, 1200),
        (1200, 1300),
    ]

    for lo, hi in bins:
        cnt = sum(lo <= r["n_tokens"] < hi for r in selected)
        print(f"{lo}-{hi}: {cnt}")


if __name__ == "__main__":
    anns = step2_read_annotations()
    records = step3_count_tokens(anns)
    selected = step4_filter_600_1300(records)
    step5_check_bins(selected)
    save_selected_list(selected)

