# 这是为了评估 Fox-100 数据集的两组推理配置：vt64 和 vt100 的字符级错误率（CER），输出每页统计和整体 CER
# 计算 CER 的脚本
import sys
import os
# 动态计算项目根目录 (scripts/xx/xx.py -> ../../ -> root)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import json

from src.normalization import normalize_text   # 导入文本规范化函数


# ========== 0. 路径设置（根据你现在的目录结构） ==========

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

FOX_DIR = os.path.join(PROJECT_ROOT, "data", "Fox")
EXP_DIR = os.path.join(FOX_DIR, "exp_fox100")

GT_PATH = os.path.join(EXP_DIR, "en_page_ocr_100.json")
PRED_VT64_PATH = os.path.join(EXP_DIR, "preds_vt64.json")
PRED_VT100_PATH = os.path.join(EXP_DIR, "preds_vt100.json")  # 如果暂时还没跑完 vt100，可以先只评 vt64

OUT_STATS_VT64 = os.path.join(EXP_DIR, "stats_vt64_pages.json")
OUT_STATS_VT100 = os.path.join(EXP_DIR, "stats_vt100_pages.json")





# ========== 2. Levenshtein 距离（字符级），用于 CER ==========

def levenshtein_distance(a: str, b: str) -> int:
    """
    计算字符串 a, b 的 Levenshtein 编辑距离（字符级）。
    这里用经典 DP，仅返回距离，不回溯路径。
    为了效率，用两行滚动数组。
    """
    n, m = len(a), len(b)
    if n == 0:
        return m
    if m == 0:
        return n

    # prev[j] = distance(a[:i-1], b[:j])
    prev = list(range(m + 1))
    curr = [0] * (m + 1)

    for i in range(1, n + 1):
        curr[0] = i
        ca = a[i - 1]
        for j in range(1, m + 1):
            cb = b[j - 1]
            cost = 0 if ca == cb else 1
            # 删除 a 中一个字符、插入一个字符、替换
            curr[j] = min(
                prev[j] + 1,      # deletion
                curr[j - 1] + 1,  # insertion
                prev[j - 1] + cost
            )
        prev, curr = curr, prev

    return prev[m]


# ========== 3. 读取 GT 和预测，并对齐 image 名 ==========

def load_gt():
    with open(GT_PATH, "r", encoding="utf-8") as f:
        anns = json.load(f)
    gt_by_image = {
        ann["image"]: ann["conversations"][1]["value"]
        for ann in anns
    }
    return gt_by_image


def load_pred(pred_path):
    with open(pred_path, "r", encoding="utf-8") as f:
        preds = json.load(f)
    pred_by_image = {p["image"]: p["pred"] for p in preds}
    return pred_by_image


def build_pairs(gt_by_image, pred_by_image):
    """
    把 GT 和预测按 image 对齐，返回一个列表：
    [
      {"image": "...", "gt": "...", "pred": "..."},
      ...
    ]
    """
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


# ========== 4. 对一组 pairs 计算 CER，并保存每页统计 ==========

def eval_pairs(pairs, out_path, tag="vt64"):
    """
    计算给定预测下，每页的 CER 和整体 CER。
    结果写入 out_path (JSON)。
    """
    page_stats = []
    total_chars = 0
    total_dist = 0

    print(f"\n=== 开始评测 {tag}，样本数 = {len(pairs)} ===")

    for i, item in enumerate(pairs):
        img = item["image"]
        gt_raw = item["gt"]
        pred_raw = item["pred"]

        gt = normalize_text(gt_raw)
        pred = normalize_text(pred_raw)

        dist = levenshtein_distance(gt, pred)
        n_char = len(gt)
        cer = dist / n_char if n_char > 0 else 0.0

        total_chars += n_char
        total_dist += dist

        page_stats.append({
            "image": img,
            "n_char": n_char,
            "edit_distance": dist,
            "cer": cer,
        })

        if i < 3:
            print(f"[样例 {i}] {img}: n_char={n_char}, dist={dist}, CER={cer:.4f}")

    overall_cer = total_dist / total_chars if total_chars > 0 else 0.0

    # 保存每页统计
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "overall_cer": overall_cer,
            "total_chars": total_chars,
            "total_edit_distance": total_dist,
            "pages": page_stats,
        }, f, ensure_ascii=False, indent=2)

    print(f"\n✅ {tag} 评测完成：")
    print(f"   总字符数 = {total_chars}")
    print(f"   总编辑距离 = {total_dist}")
    print(f"   整体 CER = {overall_cer:.4%}")
    print(f"   详细结果已保存到: {out_path}")


# ========== 5. 主函数：分别评 vt64 / vt100 ==========

def main():
    print("SCRIPT_DIR:", SCRIPT_DIR)
    print("GT_PATH:", GT_PATH)
    print("PRED_VT64_PATH:", PRED_VT64_PATH)
    print("PRED_VT100_PATH:", PRED_VT100_PATH)

    # 1) 读取 GT
    gt_by_image = load_gt()
    print(f"读取 GT 条目数: {len(gt_by_image)}")

    # 2) 评测 vt64（如果预测文件存在）
    if os.path.exists(PRED_VT64_PATH):
        print("\n--- 评测 vt64 ---")
        pred64 = load_pred(PRED_VT64_PATH)
        pairs64 = build_pairs(gt_by_image, pred64)
        eval_pairs(pairs64, OUT_STATS_VT64, tag="vt64")
    else:
        print(f"⚠ 找不到 {PRED_VT64_PATH}，跳过 vt64 评测")

    # 3) 评测 vt100（如果预测文件存在）
    if os.path.exists(PRED_VT100_PATH):
        print("\n--- 评测 vt100 ---")
        pred100 = load_pred(PRED_VT100_PATH)
        pairs100 = build_pairs(gt_by_image, pred100)
        eval_pairs(pairs100, OUT_STATS_VT100, tag="vt100")
    else:
        print(f"⚠ 找不到 {PRED_VT100_PATH}，跳过 vt100 评测")


if __name__ == "__main__":
    main()
