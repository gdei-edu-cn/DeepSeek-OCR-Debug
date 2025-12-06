# 用于在OmniDocBench的表格子集上跑推理，并使用动态路由策略
import sys
import os
import json
import torch
import re

# --- 1. 路径引导 ---
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if PROJECT_ROOT not in sys.path: sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "3rdparty"))

from deepseek_ocr.modeling_deepseekocr import DeepseekOCRForCausalLM
from transformers import AutoConfig, AutoTokenizer

# --- 2. 配置 ---
# 数据路径：还是用 OmniDocBench 的表格子集
DATA_ROOT = os.path.join(PROJECT_ROOT, "data", "OmniDocBench", "exp_table_subset")
IMG_DIR = os.path.join(DATA_ROOT, "images")
OUT_DIR = DATA_ROOT

PROMPT = "<image>\nFree OCR."

# --- 3. 定义“路由器” (The Router) ---
def needs_high_res(text: str) -> bool:
    """
    Router V2: 基于密度和特征的智能路由
    """
    if not text:
        return False

    # 1. 预处理：去掉空白，只看实字符
    # 这样算出来的 ratio 才是“含金量”
    non_space_chars = [c for c in text if not c.isspace()]
    total_len = len(non_space_chars)
    
    if total_len < 50:  # 文本太短，信息不足，保守一点不触发
        return False

    # 2. 统计基础指标
    digit_count = sum(c.isdigit() for c in non_space_chars)
    digit_ratio = digit_count / total_len
    pipe_count = text.count('|')

    # --- 判定策略 ---

    # 规则 1: Markdown 表格特征 (最强特征)
    # 如果有 8 个以上的竖线，几乎肯定是个表
    if pipe_count >= 8:
        return True

    # 规则 2: 数字极其密集 (典型财务报表)
    # 占比 > 18% 且 数量 > 40 (过滤掉只有年份的短文本)
    if digit_ratio > 0.18 and digit_count >= 40:
        return True

    # 规则 3: 长数字串特征 (ID/金额/坐标)
    # 查找连续4位以上的数字
    long_runs = re.findall(r'\d{4,}', text)
    long_run_count = len(long_runs)
    # 计算最长的一串有多长
    max_run = max((len(m) for m in long_runs), default=0)

    # 如果有很长的数字(6位+)，且出现了好几次，且整体数字密度也不低
    if max_run >= 6 and long_run_count >= 3 and digit_ratio > 0.08:
        return True

    # 规则 4: 绝对数量兜底 (应对超大纯数字表)
    if digit_count > 80:
        return True

    return False

# --- 4. 加载模型 ---
def load_model():
    path = os.path.join(PROJECT_ROOT, "3rdparty", "deepseek_ocr")
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    config = AutoConfig.from_pretrained(path, local_files_only=True)
    model = DeepseekOCRForCausalLM.from_pretrained(path, config=config, torch_dtype=torch.bfloat16, local_files_only=True)
    if torch.cuda.is_available(): model = model.eval().cuda()
    return tokenizer, model

def run_dynamic_inference():
    print(f"\n🚀 启动动态路由推理 (Dynamic Routing Inference)...")
    print(f"策略: 先用 vt64 探路 -> 发现密集数字 -> 切换 Gundam 重跑")
    
    tokenizer, model = load_model()
    images = [f for f in os.listdir(IMG_DIR) if f.lower().endswith(('.png', '.jpg'))]
    results = []
    
    gundam_triggered_count = 0
    
    for idx, img in enumerate(images):
        img_path = os.path.join(IMG_DIR, img)
        if idx % 10 == 0: print(f"处理进度: {idx}/{len(images)}... (Gundam触发次数: {gundam_triggered_count})")
        
        # === Stage 1: 快速浏览 (vt64) ===
        # base_size=512, crop_mode=False -> 极速、低成本
        res_fast = model.infer(
            tokenizer,
            prompt=PROMPT,
            image_file=img_path,
            output_path=os.path.join(OUT_DIR, "tmp_dynamic"),
            base_size=512,
            image_size=512,
            crop_mode=False, # 关闭切片，只看全图
            save_results=False,
            eval_mode=True
        )
        pred_text = str(res_fast)
        mode_used = "vt64"
        
        # === Router: 判官决策 ===
        if needs_high_res(pred_text):
            # === Stage 2: 深度精读 (High-Res Base) ===
            # 触发了！切换到高清模式
            gundam_triggered_count += 1
            
            # 【修改点解析】
            # 1. base_size=1024: 提供足够的分辨率看清数字。
            # 2. crop_mode=False: 关闭动态切片，避开源码 Bug，保证 100% 能跑通。
            
            res_slow = model.infer(
                tokenizer,
                prompt=PROMPT,
                image_file=img_path,
                output_path=os.path.join(OUT_DIR, "tmp_dynamic"),
                base_size=1024,  
                image_size=1024,
                crop_mode=False, # <--- 必须是 False！
                save_results=False,
                eval_mode=True
            )
            pred_text = str(res_slow)
            mode_used = "High-Res"
        
        results.append({
            "image": img,
            "pred": pred_text,
            "mode": mode_used # 记录一下到底用了哪个模式，论文里要分析比例
        })
        
    # 保存结果
    out_file = os.path.join(OUT_DIR, "preds_dynamic.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print("\n✅ 动态推理完成！")
    print(f"   总图片数: {len(images)}")
    print(f"   Gundam 触发率: {gundam_triggered_count / len(images) * 100:.1f}%")
    print(f"   结果已保存至: {out_file}")

if __name__ == "__main__":
    run_dynamic_inference()