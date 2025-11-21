# 生成 errors.jsonl 的对齐脚本

# align_fox100_errors.py
import sys
import os
# 动态计算项目根目录 (scripts/xx/xx.py -> ../../ -> root)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import json

from src.normalization import normalize_text  # 使用统一的规范化函数

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

FOX_DIR = os.path.join(PROJECT_ROOT, "data", "Fox")
EXP_DIR = os.path.join(FOX_DIR, "exp_fox100")

GT_PATH = os.path.join(EXP_DIR, "en_page_ocr_100.json")
PRED_VT64_PATH = os.path.join(EXP_DIR, "preds_vt64.json")
PRED_VT100_PATH = os.path.join(EXP_DIR, "preds_vt100.json")

OUT_ERR_VT64 = os.path.join(EXP_DIR, "fox100_errors_vt64.jsonl")
OUT_ERR_VT100 = os.path.join(EXP_DIR, "fox100_errors_vt100.jsonl")


# ---------- 1. 读入 GT / 预测，并按 image 对齐 ----------

def load_gt():
    with open(GT_PATH, "r", encoding="utf-8") as f:
        anns = json.load(f)
    return {ann["image"]: ann["conversations"][1]["value"] for ann in anns}


def load_pred(pred_path):
    with open(pred_path, "r", encoding="utf-8") as f:
        preds = json.load(f)
    return {p["image"]: p["pred"] for p in preds}


def build_pairs(gt_by_image, pred_by_image):
    images = sorted(gt_by_image.keys())
    pairs = []
    missing = []
    for img in images:
        if img not in pred_by_image:
            missing.append(img)
            continue
        pairs.append({
            "image": img,
            "gt": gt_by_image[img],
            "pred": pred_by_image[img],
        })
    if missing:
        print(f"⚠ 预测中缺少 {len(missing)} 张图片，例如: {missing[:5]}")
    print(f"✅ 成功对齐图片数量: {len(pairs)}")
    return pairs


# ---------- 2. 分词 & 对齐 ----------

def tokenize_words(text: str):
    """
    先用 normalize_text 做统一规范化，再按空白切分成“词级 token”。
    注意：split() 会把换行也当成空白处理掉。
    """
    norm = normalize_text(text)
    if not norm:
        return []
    return norm.split()


def align_tokens(gt_tokens, pred_tokens):
    """
    用 Levenshtein 在“词级”上对齐，返回一个操作序列：
    每个元素: {"op": "eq/sub/ins/del", "gt_idx": int or None, "pred_idx": int or None}
    """
    n, m = len(gt_tokens), len(pred_tokens)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    back = [[None] * (m + 1) for _ in range(n + 1)]

    # 初始化边界：全删 / 全插入
    for i in range(1, n + 1):
        dp[i][0] = i
        back[i][0] = ("del", i - 1, 0)
    for j in range(1, m + 1):
        dp[0][j] = j
        back[0][j] = ("ins", 0, j - 1)

    # 动态规划
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if gt_tokens[i - 1] == pred_tokens[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
                back[i][j] = ("eq", i - 1, j - 1)
            else:
                del_cost = dp[i - 1][j] + 1
                ins_cost = dp[i][j - 1] + 1
                sub_cost = dp[i - 1][j - 1] + 1

                best = min(del_cost, ins_cost, sub_cost)
                dp[i][j] = best
                if best == sub_cost:
                    back[i][j] = ("sub", i - 1, j - 1)
                elif best == del_cost:
                    back[i][j] = ("del", i - 1, j)
                else:
                    back[i][j] = ("ins", i, j - 1)

    # 回溯，生成操作序列
    i, j = n, m
    ops = []
    while i > 0 or j > 0:
        op, pi, pj = back[i][j]

        if op in ("eq", "sub"):
            gt_idx = i - 1
            pred_idx = j - 1
        elif op == "del":
            gt_idx = i - 1
            pred_idx = None
        elif op == "ins":
            gt_idx = None
            pred_idx = j - 1
        else:
            raise ValueError(f"未知操作: {op}")

        ops.append({
            "op": op,
            "gt_idx": gt_idx,
            "pred_idx": pred_idx,
        })

        i, j = pi, pj

    ops.reverse()
    return ops


# ---------- 3. 抽取单页错误 ----------

def extract_errors_for_page(image_name, gt_text, pred_text, mode):
    """
    对单页做：
      GT / pred 规范化 + 分词 + 对齐
    返回一个 list，每个元素是一条“非 eq”的错误记录。
    """
    gt_tokens = tokenize_words(gt_text)
    pred_tokens = tokenize_words(pred_text)

    ops = align_tokens(gt_tokens, pred_tokens)

    errors = []
    for step in ops:
        op = step["op"]
        gt_idx = step["gt_idx"]
        pred_idx = step["pred_idx"]

        if op == "eq":
            continue  # 正确 token 不记录

        gt_tok = gt_tokens[gt_idx] if gt_idx is not None and 0 <= gt_idx < len(gt_tokens) else ""
        pred_tok = pred_tokens[pred_idx] if pred_idx is not None and 0 <= pred_idx < len(pred_tokens) else ""

        # 给一点上下文，方便人工检查
        gt_prev = gt_tokens[gt_idx - 1] if gt_idx is not None and gt_idx - 1 >= 0 else ""
        gt_next = gt_tokens[gt_idx + 1] if gt_idx is not None and gt_idx + 1 < len(gt_tokens) else ""

        errors.append({
            "image": image_name,
            "mode": mode,          # vt64 / vt100
            "op": op,              # sub / ins / del
            "gt_token": gt_tok,
            "pred_token": pred_tok,
            "gt_index": gt_idx,
            "pred_index": pred_idx,
            "gt_prev": gt_prev,
            "gt_next": gt_next,
        })

    return errors


# ---------- 4. 整个模式（vt64 / vt100）批量抽取 ----------

def save_errors_for_mode(pairs, mode, out_path):
    total_err = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for item in pairs:
            img = item["image"]
            gt = item["gt"]
            pred = item["pred"]

            errs = extract_errors_for_page(img, gt, pred, mode=mode)
            total_err += len(errs)
            for e in errs:
                f.write(json.dumps(e, ensure_ascii=False) + "\n")

    print(f"✅ {mode} 错误抽取完成，共 {total_err} 条错误，写入 {out_path}")


def main():
    print("SCRIPT_DIR:", SCRIPT_DIR)
    print("GT_PATH:", GT_PATH)
    print("PRED_VT64_PATH:", PRED_VT64_PATH)
    print("PRED_VT100_PATH:", PRED_VT100_PATH)

    gt_by_image = load_gt()
    print("读取 GT 条目数:", len(gt_by_image))

    # vt64
    if os.path.exists(PRED_VT64_PATH):
        print("\n--- 抽取 vt64 错误 ---")
        pred64 = load_pred(PRED_VT64_PATH)
        pairs64 = build_pairs(gt_by_image, pred64)
        save_errors_for_mode(pairs64, mode="vt64", out_path=OUT_ERR_VT64)
    else:
        print("⚠ 找不到 preds_vt64.json，跳过 vt64")

    # vt100
    if os.path.exists(PRED_VT100_PATH):
        print("\n--- 抽取 vt100 错误 ---")
        pred100 = load_pred(PRED_VT100_PATH)
        pairs100 = build_pairs(gt_by_image, pred100)
        save_errors_for_mode(pairs100, mode="vt100", out_path=OUT_ERR_VT100)
    else:
        print("⚠ 找不到 preds_vt100.json，跳过 vt100")


if __name__ == "__main__":
    main()
