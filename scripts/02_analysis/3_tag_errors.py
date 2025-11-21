# 核心逻辑。包含你的正则分类规则（Regex），这是论文 RQ1 的核心。
# tag_fox100_error_types.py
import sys
import os
# 动态计算项目根目录 (scripts/xx/xx.py -> ../../ -> root)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import json

from src.taxonomy import guess_type

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

FOX_DIR = os.path.join(PROJECT_ROOT, "data", "Fox")
EXP_DIR = os.path.join(FOX_DIR, "exp_fox100")

IN_VT64 = os.path.join(EXP_DIR, "fox100_errors_vt64.jsonl")
IN_VT100 = os.path.join(EXP_DIR, "fox100_errors_vt100.jsonl")

OUT_VT64 = os.path.join(EXP_DIR, "fox100_errors_vt64_typed.jsonl")
OUT_VT100 = os.path.join(EXP_DIR, "fox100_errors_vt100_typed.jsonl")


def process(in_path, out_path, tag):
    print(f"\n--- 处理 {tag}: {in_path} ---")
    counts = {}

    with open(in_path, "r", encoding="utf-8") as fin, \
         open(out_path, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)

            # 优先用 gt_token，没有就用 pred_token
            tok = rec.get("gt_token") or rec.get("pred_token") or ""
            t = guess_type(tok)
            rec["type"] = t

            counts[t] = counts.get(t, 0) + 1

            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"写入带类型的错误日志: {out_path}")
    print("类型分布：")
    for k, v in sorted(counts.items(), key=lambda x: -x[1]):
        print(f"  {k:12s} : {v}")


def main():
    if os.path.exists(IN_VT64):
        process(IN_VT64, OUT_VT64, "vt64")
    else:
        print("⚠ 找不到 fox100_errors_vt64.jsonl")

    if os.path.exists(IN_VT100):
        process(IN_VT100, OUT_VT100, "vt100")
    else:
        print("⚠ 找不到 fox100_errors_vt100.jsonl")


if __name__ == "__main__":
    main()
