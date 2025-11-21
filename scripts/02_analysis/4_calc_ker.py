# 核心逻辑。包含 ECI 权重定义和关键错误率计算，这是论文 RQ2 的核心。

# ker_fox100_simplified.py
import sys
import os
# 动态计算项目根目录 (scripts/xx/xx.py -> ../../ -> root)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import json

from src.metrics import compute_stats as compute_error_stats

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FOX_DIR = os.path.join(PROJECT_ROOT, "data", "Fox")
EXP_DIR = os.path.join(FOX_DIR, "exp_fox100")

def summarize(path):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return compute_error_stats(records)

def main():
    for mode in ["vt64", "vt100"]:
        path = os.path.join(EXP_DIR, f"fox100_errors_{mode}_typed.jsonl")
        stats = summarize(path)
        print(f"\n=== {mode} ===")
        print(f"总错误数           : {stats['total_err']}")
        print(f"关键类型错误数     : {stats['critical_err']}  "
              f"({stats['critical_err']/stats['total_err']:.2%} of errors)")

        print(f"错误总权重 ECI_all : {stats['total_weight']:.2f}")
        print(f"关键错误权重 ECI_crit : {stats['critical_weight']:.2f}  "
              f"({stats['critical_weight']/stats['total_weight']:.2%} of weight)")

if __name__ == "__main__":
    main()
