# 用于在Fox-100和OmniDocBench的表格子集上跑推理，并使用动态路由策略
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

# --- 2. 配置数据源 ---
FOX_DIR = os.path.join(PROJECT_ROOT, "data", "Fox", "exp_fox100", "images")
OMNI_DIR = os.path.join(PROJECT_ROOT, "data", "OmniDocBench", "exp_table_subset", "images")

# 结果输出目录
OUT_DIR = os.path.join(PROJECT_ROOT, "results", "reports", "mixed_test")
os.makedirs(OUT_DIR, exist_ok=True)

PROMPT = "<image>\nFree OCR."

# --- 3. 路由器 V2 (保持一致) ---
def needs_high_res(text: str) -> bool:
    if not text: return False
    non_space_chars = [c for c in text if not c.isspace()]
    total_len = len(non_space_chars)
    if total_len < 50: return False 

    digit_count = sum(c.isdigit() for c in non_space_chars)
    digit_ratio = digit_count / total_len
    pipe_count = text.count('|')

    if pipe_count >= 8: return True
    if digit_ratio > 0.18 and digit_count >= 40: return True
    
    long_runs = re.findall(r'\d{4,}', text)
    max_run = max((len(m) for m in long_runs), default=0)
    if max_run >= 6 and len(long_runs) >= 3 and digit_ratio > 0.08: return True
    
    if digit_count > 80: return True
    return False

# --- 4. 加载模型 ---
def load_model():
    path = os.path.join(PROJECT_ROOT, "3rdparty", "deepseek_ocr")
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    config = AutoConfig.from_pretrained(path, local_files_only=True)
    model = DeepseekOCRForCausalLM.from_pretrained(path, config=config, torch_dtype=torch.bfloat16, local_files_only=True)
    if torch.cuda.is_available(): model = model.eval().cuda()
    return tokenizer, model

# --- 5. 主程序 (实战版) ---
def run_mixed_test():
    print(f"\n🚀 启动混合场景测试 (Mixed-100: 50 Fox + 50 OmniDoc)...")
    print(f"📝 模式：真实推理 + 结果保存")
    
    # 1. 准备数据
    fox_imgs = sorted([f for f in os.listdir(FOX_DIR) if f.lower().endswith(('.png', '.jpg'))])
    omni_imgs = sorted([f for f in os.listdir(OMNI_DIR) if f.lower().endswith(('.png', '.jpg'))])
    
    # 各取前 50 张
    test_set = []
    for img in fox_imgs[:50]:
        test_set.append({"source": "Fox", "path": os.path.join(FOX_DIR, img), "name": img})
    for img in omni_imgs[:50]:
        test_set.append({"source": "OmniDoc", "path": os.path.join(OMNI_DIR, img), "name": img})
    
    print(f"   准备就绪：共 {len(test_set)} 张图片。")
    
    # 2. 加载模型
    tokenizer, model = load_model()
    
    # 3. 运行推理
    results = []
    stats = {
        "Fox": {"total": 0, "triggered": 0},
        "OmniDoc": {"total": 0, "triggered": 0}
    }
    
    for idx, item in enumerate(test_set):
        if idx % 10 == 0: print(f"进度: {idx}/{len(test_set)}...")
        
        source = item["source"]
        stats[source]["total"] += 1
        
        # === Stage 1: vt64 探针 ===
        res = model.infer(
            tokenizer, prompt=PROMPT, image_file=item["path"],
            output_path=os.path.join(OUT_DIR, "tmp"),
            base_size=512, image_size=512, crop_mode=False,
            save_results=False, eval_mode=True
        )
        pred_text = str(res)
        mode_used = "vt64"
        
        # === Router 判断 ===
        if needs_high_res(pred_text):
            stats[source]["triggered"] += 1
            
            # === Stage 2: High-Res 精读 (真实调用！) ===
            res_high = model.infer(
                tokenizer, prompt=PROMPT, image_file=item["path"],
                output_path=os.path.join(OUT_DIR, "tmp"),
                base_size=1024, image_size=1024, crop_mode=False, # 稳健高清模式
                save_results=False, eval_mode=True
            )
            pred_text = str(res_high) # 更新预测结果
            mode_used = "High-Res"
            
        # 保存单条结果
        results.append({
            "image": item["name"],
            "source": source,
            "mode": mode_used,
            "pred": pred_text
        })
            
    # 4. 保存文件
    out_file = os.path.join(OUT_DIR, "preds_mixed.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # 5. 输出统计
    print("\n" + "="*40)
    print("📊 混合场景效率分析报告 (Efficiency Report)")
    print("="*40)
    
    total_trigger = 0
    
    for src in ["Fox", "OmniDoc"]:
        data = stats[src]
        count = data["total"]
        trig = data["triggered"]
        rate = trig / count * 100 if count > 0 else 0
        total_trigger += trig
        
        print(f"📌 {src}:")
        print(f"   - 触发率: {trig}/{count} ({rate:.1f}%)")
        print("-" * 20)
        
    all_rate = total_trigger / 100 * 100
    print(f"🏆 综合结果 (Mixed-100):")
    print(f"   - 总体高清调用率: {all_rate:.1f}%")
    print(f"   - 详细结果已保存至: {out_file}")
    print("="*40)

if __name__ == "__main__":
    run_mixed_test()