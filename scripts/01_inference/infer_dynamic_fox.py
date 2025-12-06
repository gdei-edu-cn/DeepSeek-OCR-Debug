# 用于在Fox-100数据集上跑推理，并使用动态路由策略
import sys
import os
import json
import torch

# 路径引导
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if PROJECT_ROOT not in sys.path: sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "3rdparty"))

from deepseek_ocr.modeling_deepseekocr import DeepseekOCRForCausalLM
from transformers import AutoConfig, AutoTokenizer

# --- 配置 ---
DATA_ROOT = os.path.join(PROJECT_ROOT, "data", "Fox", "exp_fox100")
IMG_DIR = os.path.join(DATA_ROOT, "images")
OUT_DIR = DATA_ROOT
PROMPT = "<image>\nFree OCR."

# --- 路由器 V2 (与 infer_mixed_test.py 保持一致) ---
def needs_high_res(text):
    if not text: return False
    non_space_chars = [c for c in text if not c.isspace()]
    total_len = len(non_space_chars)
    if total_len < 50: return False
    
    digit_count = sum(c.isdigit() for c in non_space_chars)
    digit_ratio = digit_count / total_len
    pipe_count = text.count('|')
    
    if pipe_count >= 8: return True
    if digit_ratio > 0.18 and digit_count >= 40: return True
    
    import re
    long_runs = re.findall(r'\d{4,}', text)
    max_run = max((len(m) for m in long_runs), default=0)
    long_run_count = len(long_runs)
    if max_run >= 6 and long_run_count >= 3 and digit_ratio > 0.08: return True
    
    if digit_count > 80: return True
    return False

def load_model():
    path = os.path.join(PROJECT_ROOT, "3rdparty", "deepseek_ocr")
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    config = AutoConfig.from_pretrained(path, local_files_only=True)
    model = DeepseekOCRForCausalLM.from_pretrained(path, config=config, torch_dtype=torch.bfloat16, local_files_only=True)
    if torch.cuda.is_available(): model = model.eval().cuda()
    return tokenizer, model

def run_fox_dynamic():
    print(f"\n🚀 启动 Fox-100 动态路由推理 (保存结果版)...")
    tokenizer, model = load_model()
    images = sorted([f for f in os.listdir(IMG_DIR) if f.lower().endswith(('.png', '.jpg'))])
    
    gundam_count = 0
    results = []
    
    for idx, img in enumerate(images):
        if idx % 10 == 0: print(f"进度: {idx}/{len(images)}... (Gundam触发: {gundam_count})")
        
        # Stage 1: vt64 探针
        res = model.infer(tokenizer, prompt=PROMPT, image_file=os.path.join(IMG_DIR, img),
                          output_path=os.path.join(OUT_DIR, "tmp"),
                          base_size=512, image_size=512, crop_mode=False, save_results=False, eval_mode=True)
        pred_text = str(res)
        mode_used = "vt64"
        
        # Router 判断
        if needs_high_res(pred_text):
            gundam_count += 1
            # Stage 2: High-Res 精读 (用 1024 分辨率，不开切片，防止报错)
            res_slow = model.infer(tokenizer, prompt=PROMPT, image_file=os.path.join(IMG_DIR, img),
                                   output_path=os.path.join(OUT_DIR, "tmp"),
                                   base_size=1024, image_size=1024, crop_mode=False, save_results=False, eval_mode=True)
            pred_text = str(res_slow)
            mode_used = "High-Res"
            
        results.append({"image": img, "pred": pred_text, "mode": mode_used})

    # 保存结果 (关键！)
    out_file = os.path.join(OUT_DIR, "preds_dynamic.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Fox-100 动态测试完成！")
    print(f"   Gundam 触发率: {gundam_count}/{len(images)} = {gundam_count/len(images)*100:.1f}%")
    print(f"   结果已保存至: {out_file}")

if __name__ == "__main__":
    run_fox_dynamic()