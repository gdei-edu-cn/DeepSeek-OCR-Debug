# 核心逻辑。包含 ECI 权重定义和关键错误率计算，这是论文 RQ2 的核心。

# ker_fox100_simplified.py
import os
import json

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EXP_DIR = os.path.join(SCRIPT_DIR, "data", "Fox", "exp_fox100")

# 按照错误率分析.md 4.2 的思路，给每种类型一个权重
WEIGHTS = {
    "word":         1.0,   # 一般内容词
    "punct":        0.5,   # 标点/空白类
    "number":       3.0,   # 数字
    "number+unit":  3.0,   # 数值+单位
    "money":        3.0,   # 金额
    "date":         3.0,   # 日期
    "math_symbol":  3.0,   # 数学/公式符号
    "negation":     2.0,   # 否定词 / ± 之类
    "comparator":   2.0,   # ≥ ≤ > < 等比较符号
}

# 认为下面这些是“关键错误类型”
CRITICAL_TYPES = {
    "number", "number+unit", "money", "date",
    "math_symbol", "negation", "comparator",
}

def compute_stats(path):
    total_err = 0
    total_weight = 0.0

    critical_err = 0
    critical_weight = 0.0

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            t = rec.get("type", "word")

            w = WEIGHTS.get(t, 1.0)

            total_err += 1
            total_weight += w

            if t in CRITICAL_TYPES:
                critical_err += 1
                critical_weight += w

    return {
        "total_err": total_err,
        "total_weight": total_weight,
        "critical_err": critical_err,
        "critical_weight": critical_weight,
    }

def main():
    for mode in ["vt64", "vt100"]:
        path = os.path.join(EXP_DIR, f"fox100_errors_{mode}_typed.jsonl")
        stats = compute_stats(path)
        print(f"\n=== {mode} ===")
        print(f"总错误数           : {stats['total_err']}")
        print(f"关键类型错误数     : {stats['critical_err']}  "
              f"({stats['critical_err']/stats['total_err']:.2%} of errors)")

        print(f"错误总权重 ECI_all : {stats['total_weight']:.2f}")
        print(f"关键错误权重 ECI_crit : {stats['critical_weight']:.2f}  "
              f"({stats['critical_weight']/stats['total_weight']:.2%} of weight)")

if __name__ == "__main__":
    main()
