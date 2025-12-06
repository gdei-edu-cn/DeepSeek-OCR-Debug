# 用于在OmniDocBench的表格子集上跑推理
# 关键点：这次我们要测Benchmark 性能，所以不需要加 do_sample=True。虽然我们改了源代码支持 kwargs，但只要你不传参数，它默认就是贪心（Greedy），这正是我们想要的。
import sys, os, json, torch

# 路径引导
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if PROJECT_ROOT not in sys.path: sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "3rdparty"))

from deepseek_ocr.modeling_deepseekocr import DeepseekOCRForCausalLM
from transformers import AutoConfig, AutoTokenizer

# --- 配置 ---
DATA_ROOT = os.path.join(PROJECT_ROOT, "data", "OmniDocBench", "exp_table_subset")
IMG_DIR = os.path.join(DATA_ROOT, "images")
OUT_DIR = DATA_ROOT  # 结果直接存在这里

PROMPT = "<image>\nFree OCR." # 保持一致

def load_model():
    path = os.path.join(PROJECT_ROOT, "3rdparty", "deepseek_ocr")
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    config = AutoConfig.from_pretrained(path, local_files_only=True)
    model = DeepseekOCRForCausalLM.from_pretrained(path, config=config, torch_dtype=torch.bfloat16, local_files_only=True)
    if torch.cuda.is_available(): model = model.eval().cuda()
    return tokenizer, model

def run_inference(tokenizer, model, mode_name, base_size, image_size):
    print(f"\n🚀 正在运行 OmniDocBench 表格测试: {mode_name} ...")
    
    images = [f for f in os.listdir(IMG_DIR) if f.lower().endswith(('.png', '.jpg'))]
    results = []
    
    # 这里的 output_path 只是为了存放临时文件，不重要
    tmp_path = os.path.join(OUT_DIR, f"tmp_{mode_name}")
    
    for idx, img in enumerate(images):
        if idx % 10 == 0: print(f"Processing {idx}/{len(images)}...")
        
        # 注意：这里我们不传 do_sample，让它默认使用贪心解码，保证最高精度
        res = model.infer(
            tokenizer,
            prompt=PROMPT,
            image_file=os.path.join(IMG_DIR, img),
            output_path=tmp_path,
            base_size=base_size,
            image_size=image_size,
            crop_mode=False,
            save_results=False,
            eval_mode=True
        )
        results.append({"image": img, "pred": str(res)})
        
    out_file = os.path.join(OUT_DIR, f"preds_{mode_name}.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"✅ {mode_name} 完成，结果保存至: {out_file}")

def main():
    tokenizer, model = load_model()
    
    # 跑 vt64 (高压缩)
    run_inference(tokenizer, model, "vt64", 512, 512)
    
    # 跑 vt100 (中等压缩)
    run_inference(tokenizer, model, "vt100", 640, 640)

if __name__ == "__main__":
    main()