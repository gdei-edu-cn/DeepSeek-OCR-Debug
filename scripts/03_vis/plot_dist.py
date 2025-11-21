import sys
import os
# 动态计算项目根目录 (scripts/xx/xx.py -> ../../ -> root)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import matplotlib.pyplot as plt
import numpy as np

types = ["word", "number", "math_symbol", "punct", "negation",
         "money", "number+unit", "date", "comparator"]

vt64_pct = [76.10, 12.68, 10.33, 0.37, 0.22, 0.18, 0.11, 0.01, 0.01]
vt100_pct = [82.69, 7.22, 7.29, 2.43, 0.14, 0.14, 0.03, 0.07, 0.0]

x = np.arange(len(types))
width = 0.35

fig, ax = plt.subplots(figsize=(7, 3))
bars64 = ax.bar(x - width/2, vt64_pct, width, label="vt64")
bars100 = ax.bar(x + width/2, vt100_pct, width, label="vt100")

ax.set_ylabel("Error percentage (%)")
ax.set_xlabel("Error type")
ax.set_xticks(x)
ax.set_xticklabels(types, rotation=30, ha="right")
ax.legend()
ax.set_ylim(0, 90)  # 留一点顶部空白

# 在柱子上标数值（>0.3% 再写，避免 0.00% 太挤）
for bars in (bars64, bars100):
    for b in bars:
        h = b.get_height()
        if h > 0.3:
            ax.text(b.get_x() + b.get_width()/2, h + 0.5,
                    f"{h:.1f}",
                    ha="center", va="bottom", fontsize=8)

fig.tight_layout()
out_path = os.path.join(PROJECT_ROOT, "results", "figures", "fox100_error_types.png")
os.makedirs(os.path.dirname(out_path), exist_ok=True)
plt.savefig(out_path, dpi=300)
