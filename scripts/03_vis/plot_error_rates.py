# 类型错误率对比图

import matplotlib.pyplot as plt
import numpy as np
import os
import sys

# --- 路径引导 ---
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# --- 你的真实数据 (手动填入或读取json，这里我帮你填好了) ---
# GT 分母
gt_counts = {
    "word": 61838, "number": 552, "math_symbol": 69, "money": 73, "negation": 430
}
# vt64 错误数
err_vt64 = {
    "word": 18539, "number": 234, "math_symbol": 46, "money": 28, "negation": 41
}
# vt100 错误数
err_vt100 = {
    "word": 2596, "number": 158, "math_symbol": 14, "money": 2, "negation": 4
}

types = ["word", "number", "math_symbol", "money", "negation"]
# 计算错误率 (%)
rate64 = [100 * err_vt64[t] / gt_counts[t] for t in types]
rate100 = [100 * err_vt100[t] / gt_counts[t] for t in types]

# --- 画图 ---
x = np.arange(len(types))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 6))
rects1 = ax.bar(x - width/2, rate64, width, label='vt64 (High Compression)', color='#d65f5f', alpha=0.9)
rects2 = ax.bar(x + width/2, rate100, width, label='vt100 (Medium Compression)', color='#4c72b0', alpha=0.9)

# 装饰
ax.set_ylabel('Error Rate (%)', fontsize=12)
ax.set_title('Type-Specific Error Rate: Systematic Bias in Compression', fontsize=14, pad=15)
ax.set_xticks(x)
ax.set_xticklabels(types, fontsize=11)
ax.legend(fontsize=11)
ax.grid(axis='y', linestyle='--', alpha=0.5)

# 在柱子上标数字
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.1f}%',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)

autolabel(rects1)
autolabel(rects2)

# 保存
save_path = os.path.join(PROJECT_ROOT, "results/figures/error_rate_comparison.png")
os.makedirs(os.path.dirname(save_path), exist_ok=True)
plt.savefig(save_path, dpi=300, bbox_inches='tight')
print(f"图表已生成: {save_path}")