# summ_fox100_error_types.py
import sys
import os
# 动态计算项目根目录 (scripts/xx/xx.py -> ../../ -> root)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import json
from collections import Counter

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FOX_DIR = os.path.join(PROJECT_ROOT, "data", "Fox")
EXP_DIR = os.path.join(FOX_DIR, "exp_fox100")

def load_typed_error_counts(path):
    cnt = Counter()
    total = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            t = rec.get("type", "unknown")
            cnt[t] += 1
            total += 1
    return total, cnt

def main():
    for mode in ["vt64", "vt100"]:
        path = os.path.join(EXP_DIR, f"fox100_errors_{mode}_typed.jsonl")
        print(f"\n=== {mode} ===")
        total, cnt = load_typed_error_counts(path)
        print(f"总错误数: {total}")
        print("类型分布：")
        for t, c in cnt.most_common():
            print(f"  {t:12s} {c:6d}  ({c/total:6.2%})")

if __name__ == "__main__":
    main()
