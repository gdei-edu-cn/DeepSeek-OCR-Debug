# 这是为了在本地的 DeepSeek-OCR 模型上批量跑 Fox-100 数据集的两组推理配置：vt64 和 vt100，并把每张图的 OCR 结果输出为 JSON。
# 推理脚本
import sys
import os
import json
import torch

# ========== 0. 设置工作目录 ==========

script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
print(f"✅ 工作目录已设置为: {os.getcwd()}")

# ========== 1. 离线模式（和你原来一样） ==========

os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

# ========== 2. 加入本地包路径（让 deepseek_ocr 能 import 到） ==========

sys.path.insert(0, "/newdata/home/liangweitang/Desktop/DeepSeek-OCR-debug/3rdparty")

from deepseek_ocr.modeling_deepseekocr import DeepseekOCRForCausalLM
from transformers import AutoConfig, AutoTokenizer

# ========== 3. 路径配置：Fox-100 ==========

# 项目根目录：当前脚本所在目录
PROJECT_ROOT = script_dir

# Fox 数据目录：data/Fox
FOX_DIR = os.path.join(PROJECT_ROOT, "data", "Fox")
EXP_DIR = os.path.join(FOX_DIR, "exp_fox100")
IMG_DIR = os.path.join(EXP_DIR, "images")  # 你导出的 100 张 en_*.png 在这里

# 预测结果输出
PRED_VT64_PATH = os.path.join(EXP_DIR, "preds_vt64.json")
PRED_VT100_PATH = os.path.join(EXP_DIR, "preds_vt100.json")

# 临时结果目录（给 infer 的 output_path 参数用）
TMP_OUT_VT64 = os.path.join(EXP_DIR, "runs_vt64")
TMP_OUT_VT100 = os.path.join(EXP_DIR, "runs_vt100")

# OCR 提示词：论文在 Fox 上用的是无版面输出 Free OCR
PROMPT_FREE_OCR = "<image>\nFree OCR."

# ========== 4. 加载本地 tokenizer / config / model ==========

def load_local_model():
    """
    使用你本地的 deepseek_ocr 目录加载 tokenizer、config 和 DeepseekOCRForCausalLM 模型
    """
    model_dir = "/newdata/home/liangweitang/Desktop/DeepSeek-OCR-debug/3rdparty/deepseek_ocr"

    print(f"🔄 正在从本地加载 tokenizer 和 config: {model_dir}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_dir,
        local_files_only=True
    )

    config = AutoConfig.from_pretrained(
        model_dir,
        local_files_only=True
    )

    print("🔄 正在从本地加载 DeepseekOCRForCausalLM 权重...")
    model = DeepseekOCRForCausalLM.from_pretrained(
        model_dir,
        config=config,
        torch_dtype=torch.bfloat16,
        local_files_only=True
    )

    if torch.cuda.is_available():
        model = model.eval().cuda()
        print("✅ 模型已移动到 CUDA (bfloat16)")
    else:
        model = model.eval()
        print("⚠ 未检测到 GPU，将在 CPU 上运行（会很慢）")

    return tokenizer, model

# ========== 5. 遍历 Fox-100 的图片列表 ==========

def list_fox100_images():
    assert os.path.isdir(IMG_DIR), f"❌ 找不到 Fox-100 图片目录: {IMG_DIR}"

    files = sorted(
        f for f in os.listdir(IMG_DIR)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    )
    print(f"📂 在 {IMG_DIR} 找到图片数量: {len(files)}")
    if files:
        print("前 5 张图片:", files[:5])
    return files

# ========== 6. 对 Fox-100 跑一遍指定配置 ==========

def run_fox100_mode(tokenizer, model,
                    mode_name,
                    base_size,
                    image_size,
                    crop_mode,
                    tmp_out_dir,
                    json_out_path):
    """
    在 Fox-100 上，用指定的 base_size / image_size / crop_mode 运行一遍，
    把每张图片的 OCR 文本输出到 json_out_path。
    """
    os.makedirs(tmp_out_dir, exist_ok=True)

    images = list_fox100_images()
    # images = images[:1]   # ⭐ DEBUG：只保留前 1 张图，方便排查问题

    results = []

    print(f"\n=== 开始运行模式 {mode_name} ===")
    print(f"    base_size={base_size}, image_size={image_size}, crop_mode={crop_mode}")
    print(f"    临时输出目录: {tmp_out_dir}")
    print(f"    输出 JSON: {json_out_path}")

    for idx, img_name in enumerate(images):
        img_path = os.path.join(IMG_DIR, img_name)
        print(f"[{mode_name}] ({idx+1}/{len(images)}) 处理 {img_name} ...")

        # 调用你本地的 DeepseekOCRForCausalLM.infer 接口
        # 参数含义：
        # - prompt: 使用 Free OCR 提示，不带版面
        # - image_file: Fox 页面的图片路径
        # - output_path: 模型内部保存图像/中间结果的目录
        # - base_size / image_size / crop_mode: 控制视觉 token 数和分辨率
        # - save_results=False: 不必保存每张图的可视化结果，节省空间
        # - test_compress=True: 在终端打印压缩比等信息，方便之后写论文
        res = model.infer(
            tokenizer,
            prompt=PROMPT_FREE_OCR,
            image_file=img_path,
            output_path=tmp_out_dir,
            base_size=base_size,
            image_size=image_size,
            crop_mode=crop_mode,
            save_results=False,
            test_compress=True,   # 要打印压缩信息可以保留 True
            eval_mode=True        # ⭐ 关键：让 infer 返回 OCR 文本
        )

        # infer 在 eval_mode=True 时应该直接返回字符串
        if res is None:
            print(f"⚠️ infer 返回 None: {img_name}，先写空字符串，后面再排查")
            pred_text = ""
        elif isinstance(res, str):
            pred_text = res
        else:
            # 保险起见，如果是别的结构，先转成字符串
            pred_text = str(res)

        results.append({
            "image": img_name,
            "pred": pred_text
        })


    # 写出 JSON 文件
    with open(json_out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n✅ 模式 {mode_name} 完成，共处理 {len(results)} 页。")
    print(f"   预测结果已保存到: {json_out_path}")

# ========== 7. 主函数：跑 vt64 + vt100 ==========

def main():
    print("📌 PROJECT_ROOT:", PROJECT_ROOT)
    print("📌 FOX_DIR:", FOX_DIR)
    print("📌 FOX exp_fox100 目录:", EXP_DIR)
    print("📌 图片目录 IMG_DIR:", IMG_DIR)

    tokenizer, model = load_local_model()

    # --- 模式 1：vt=64，Tiny 风格 ---
    # 推荐：base_size=512, image_size=512, crop_mode=False
    run_fox100_mode(
        tokenizer=tokenizer,
        model=model,
        mode_name="vt64",
        base_size=512,
        image_size=512,
        crop_mode=False,
        tmp_out_dir=TMP_OUT_VT64,
        json_out_path=PRED_VT64_PATH
    )

    # --- 模式 2：vt=100，Small 风格 ---
    # 推荐：base_size=640, image_size=640, crop_mode=False
    run_fox100_mode(
        tokenizer=tokenizer,
        model=model,
        mode_name="vt100",
        base_size=640,
        image_size=640,
        crop_mode=False,
        tmp_out_dir=TMP_OUT_VT100,
        json_out_path=PRED_VT100_PATH
    )

    print("\n🎉 所有 Fox-100 推理完成！")
    print(f"   vt=64 结果: {PRED_VT64_PATH}")
    print(f"   vt=100 结果: {PRED_VT100_PATH}")

if __name__ == "__main__":
    main()
