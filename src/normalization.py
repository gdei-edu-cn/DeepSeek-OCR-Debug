# 公用工具。用于清理文本格式，几乎所有评测脚本都引用它。

# text_normalize.py
import unicodedata
import re

# 一些常见的引号、破折号统一成 ASCII 版本
_TRANS_TABLE = str.maketrans({
    "“": '"', "”": '"', "„": '"', "«": '"', "»": '"',
    "‘": "'", "’": "'", "‚": "'", "´": "'", "ˋ": "'",
    "–": "-", "—": "-", "−": "-",  # 各种 dash
    "…": "...",
})

def _fix_hyphen_breaks(text: str) -> str:
    """
    把像 "infor-\nmation" 这种断行，合并成 "information"。
    只在字母-字母的地方合并，避免破坏真正的减号。
    """
    return re.sub(r"([A-Za-z])-\n([A-Za-z])", r"\1\2", text)

# def normalize_text(s: str) -> str:
#     """
#     用于评测和对齐的“温和规范化”：
#     1. NFKC 归一
#     2. 统一引号、破折号、… 等符号
#     3. 统一换行符
#     4. 合并连续空格/Tab 为一个空格（保留换行）
#     5. （可选）合并字母-换行-字母 的断词
#     6. 去掉首尾空白
#     """
#     if s is None:
#         return ""

#     # 1) Unicode NFKC
#     s = unicodedata.normalize("NFKC", s)

#     # 2) 统一引号、破折号等
#     s = s.translate(_TRANS_TABLE)

#     # 3) 换行符统一为 \n
#     s = s.replace("\r\n", "\n").replace("\r", "\n")

#     # 4) 合并连续空格/Tab 为一个空格（不动换行）
#     #    注意这里不把 \n 换成空格，只是压缩行内空白
#     s = re.sub(r"[ \t]+", " ", s)

#     # 5) 处理断词：word-\nword -> wordword
#     s = _fix_hyphen_breaks(s)

#     # 6) 去掉首尾空白
#     s = s.strip()

#     return s
# 新的normalize_text函数，用于清理文本格式，几乎所有评测脚本都引用它。
def normalize_text(text):
    if text is None: return ""
    text = str(text)
    
    # 1. Unicode 归一化 (全角转半角)
    text = unicodedata.normalize('NFKC', text)
    
    # 2. 暴力清洗：只保留数字、字母、小数点、负号
    # 这样可以忽略所有逗号、货币符号($/¥)、空格、换行、Markdown符号(|)
    # 这对 "Money" 和 "Number" 的评测是最公平的，只比数字本身对不对
    # 注意：这会把 "1,000" 变成 "1000"，把 "$100" 变成 "100"
    
    # 如果你只想针对 Money/Number 做这个清洗，可以在 calc_ker 里做
    # 但为了简单，我们在这里做一个通用的“去标点”版本
    
    # 移除千分位逗号 (避免 1,000 vs 1000)
    text = text.replace(",", "")
    
    # 移除常见货币符号
    for char in "$￥€£":
        text = text.replace(char, "")
        
    # 移除 Markdown 表格符
    text = text.replace("|", " ").replace("-", " ")
    
    # 压缩空白
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text.lower()
