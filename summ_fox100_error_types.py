# summ_fox100_error_types.py
import os
import json
from collections import Counter

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EXP_DIR = os.path.join(SCRIPT_DIR, "data", "Fox", "exp_fox100")

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
