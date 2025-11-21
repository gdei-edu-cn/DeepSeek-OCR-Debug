import sys
import os
import torch

# 设置工作目录为脚本所在目录（解决调试时路径问题）
script_dir = os.path.dirname(os.path.abspath(__file__))  # 获取当前脚本所在的目录
os.chdir(script_dir)  # 将工作目录切换为脚本所在的目录
print(f"✅ 工作目录已设置为: {os.getcwd()}")  # 打印当前的工作目录

# 离线模式，防止联网下载模型(不用也可以，保险起见)
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

# 1. 加入本地包路径（让 deepseek_ocr 能 import 到）
sys.path.insert(0, "/newdata/home/liangweitang/Desktop/DeepSeek-OCR-debug/3rdparty")

# 这是 decoder（解码器），而不是 deepencoder。
# 该行代码从 3rdparty/deepseek_ocr/modeling_deepseekocr.py 文件中导入 DeepseekOCRForCausalLM 类，这是实现 Causal Language Model（因果语言模型）推理的主力类，也是下文加载和推理的核心部分。
# 该类是 DeepSeek-OCR 的主力推理模型（带 CausalLM 能力），后续用于加载 OCR 模型和执行推理任务。
from deepseek_ocr.modeling_deepseekocr import DeepseekOCRForCausalLM

# 这里导入的是 transformers 库中的 AutoConfig 和 AutoTokenizer：
# - AutoConfig 用于从本地或远程目录加载模型的配置信息（如层数、hidden_size 等结构超参数）。
# - AutoTokenizer 用于加载分词器（tokenizer），将文本转为模型可处理的 token 序列。
from transformers import AutoConfig, AutoTokenizer

# 2. 加载本地 tokenizer
# 作用：加载本地 DeepSeek-OCR 模型的分词器（tokenizer），用于后续将自然语言文本转为模型可识别的 token 序列等
tokenizer = AutoTokenizer.from_pretrained(
    "/newdata/home/liangweitang/Desktop/DeepSeek-OCR-debug/3rdparty/deepseek_ocr",
    local_files_only=True
)

# 3. 加载本地 config
# 作用：加载本地 DeepSeek-OCR 模型的配置（config），用于初始化模型结构与推理参数等
config = AutoConfig.from_pretrained(
    "/newdata/home/liangweitang/Desktop/DeepSeek-OCR-debug/3rdparty/deepseek_ocr",
    local_files_only=True
)

# 4. 加载本地权重和模型（不用AutoModel、也不用trust_remote_code了！）
# 作用：加载本地 DeepSeek-OCR 模型的权重文件，并实例化为 model 对象，供后续推理使用。
model = DeepseekOCRForCausalLM.from_pretrained(
    "/newdata/home/liangweitang/Desktop/DeepSeek-OCR-debug/3rdparty/deepseek_ocr",
    config=config,
    torch_dtype=torch.bfloat16,
    local_files_only=True
)
# 
# 设置模型为 eval 模式并移动到 GPU（cuda）
model = model.eval().cuda()

# 5. 推理
prompt = "<image>\n<|grounding|>Convert the document to markdown. "
image_file = '../../assets/show4.jpg'
# 使用绝对路径，避免在不同目录运行时创建错误的输出目录
output_path = os.path.join(script_dir, 'runs/hf/output')

# 使用 DeepSeek-OCR 模型进行推理，传入 tokenizer、提示词(prompt)、图片路径、输出路径等参数
# base_size: 动态裁剪的基础分辨率；image_size: 模型输入的图像大小
# crop_mode: 是否开启动态裁剪；save_results: 是否保存推理结果图片和文本
# test_compress: 是否测试图片压缩效果
res = model.infer(
    tokenizer, 
    prompt=prompt, 
    image_file=image_file, 
    output_path=output_path,
    base_size=1024,           # 动态裁剪基础分辨率
    image_size=640,           # 模型输入图像大小
    crop_mode=True,           # 开启动态裁剪
    save_results=True,        # 推理阶段保存结果
    test_compress=True        # 测试图片压缩影响
)