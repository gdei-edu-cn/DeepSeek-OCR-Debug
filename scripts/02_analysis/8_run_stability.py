# 用于跑稳定性测试, 修改了do_sample参数和temperature参数。使得每次推理结果不同。
import sys
import os
import json
import torch

# --- 1. 路径引导 ---
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

THIRDPARTY_DIR = os.path.join(PROJECT_ROOT, "3rdparty")
if THIRDPARTY_DIR not in sys.path:
    sys.path.insert(0, THIRDPARTY_DIR)

from deepseek_ocr.modeling_deepseekocr import DeepseekOCRForCausalLM
from transformers import AutoConfig, AutoTokenizer

# --- 2. 配置 ---
FOX_DIR = os.path.join(PROJECT_ROOT, "data", "Fox")
IMG_DIR = os.path.join(FOX_DIR, "exp_fox100", "images")
STABILITY_DIR = os.path.join(FOX_DIR, "exp_fox100", "stability")
os.makedirs(STABILITY_DIR, exist_ok=True)

# 刚才找出的 5 个重灾区页面 (Hardcoded for simplicity)
TARGET_IMAGES = ['en_79.png', 'en_19.png', 'en_34.png', 'en_104.png', 'en_15.png']

PROMPT_FREE_OCR = "<image>\nFree OCR."

# --- 3. 加载模型 ---
def load_local_model():
    model_dir = os.path.join(PROJECT_ROOT, "3rdparty", "deepseek_ocr")
    print(f"🔄 加载模型: {model_dir}")
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    config = AutoConfig.from_pretrained(model_dir, local_files_only=True)
    model = DeepseekOCRForCausalLM.from_pretrained(model_dir, config=config, torch_dtype=torch.bfloat16, local_files_only=True)
    
    if torch.cuda.is_available():
        model = model.eval().cuda()
    return tokenizer, model

# --- 4. 核心：带 Temperature 的推理函数 ---
def run_stability_test():
    tokenizer, model = load_local_model()
    
    # 我们只跑 vt64 模式 (base_size=512)，因为这是问题最严重的地方
    base_size = 512
    image_size = 512
    crop_mode = False

    print(f"\n🚀 开始稳定性测试 (N=5, Temp=0.7)")
    print(f"目标图片: {TARGET_IMAGES}")

    # 循环跑 5 次 (Run 1 to Run 5)
    for run_idx in range(1, 6):
        print(f"\n--- Run {run_idx}/5 ---")
        run_results = []
        
        for img_name in TARGET_IMAGES:
            img_path = os.path.join(IMG_DIR, img_name)
            print(f"处理 {img_name} ...")
            
            # 关键修改：我们在调用 infer 时传入 generation_config 参数
            # 注意：DeepSeek 的 infer 源码里通常会把 **kwargs 传给 generate
            # 如果你的 infer 源码不支持 **kwargs，我们可能需要临时改一下 deepseek_ocr/modeling_deepseekocr.py
            # 但通常 transformer 风格的模型都支持。
            
            try:
                res = model.infer(
                    tokenizer,
                    prompt=PROMPT_FREE_OCR,
                    image_file=img_path,
                    output_path=os.path.join(STABILITY_DIR, f"tmp_run{run_idx}"),
                    base_size=base_size,
                    image_size=image_size,
                    crop_mode=crop_mode,
                    save_results=False,
                    eval_mode=True,
                    # --- 这里是注入的随机性参数 ---
                    do_sample=True,      # 开启采样（不开启就是贪心，每次结果一样）
                    temperature=0.7,     # 增加随机性
                    top_p=0.9,           # 常用采样参数
                )
            except TypeError:
                # 如果报错说不支持 do_sample，说明 infer 函数没透传 kwargs
                # 那就只能用默认贪心跑 (此时 5 次结果会完全一样，证明是 Deterministic Error)
                print("⚠️ 警告: model.infer 不支持 do_sample 参数，将使用默认贪心解码。")
                print("如果 5 次结果完全一致，说明错误是 100% 必然的 (Deterministic)。")
                res = model.infer(
                    tokenizer,
                    prompt=PROMPT_FREE_OCR,
                    image_file=img_path,
                    output_path=os.path.join(STABILITY_DIR, f"tmp_run{run_idx}"),
                    base_size=base_size,
                    image_size=image_size,
                    crop_mode=crop_mode,
                    save_results=False,
                    eval_mode=True
                )

            pred_text = str(res) if res is not None else ""
            run_results.append({"image": img_name, "pred": pred_text})
        
        # 保存这一次 Run 的结果
        out_json = os.path.join(STABILITY_DIR, f"preds_run{run_idx}.json")
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(run_results, f, ensure_ascii=False, indent=2)
        print(f"✅ Run {run_idx} 完成，保存至 {out_json}")

if __name__ == "__main__":
    run_stability_test()