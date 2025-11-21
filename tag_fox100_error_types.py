# 核心逻辑。包含你的正则分类规则（Regex），这是论文 RQ1 的核心。
# tag_fox100_error_types.py
import os
import json
import re

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = SCRIPT_DIR

FOX_DIR = os.path.join(PROJECT_ROOT, "data", "Fox")
EXP_DIR = os.path.join(FOX_DIR, "exp_fox100")

IN_VT64 = os.path.join(EXP_DIR, "fox100_errors_vt64.jsonl")
IN_VT100 = os.path.join(EXP_DIR, "fox100_errors_vt100.jsonl")

OUT_VT64 = os.path.join(EXP_DIR, "fox100_errors_vt64_typed.jsonl")
OUT_VT100 = os.path.join(EXP_DIR, "fox100_errors_vt100_typed.jsonl")


NEG_WORDS = {"not", "no", "never", "none", "cannot", "can't", "n't"}
COMPARATORS = {">", "<", ">=", "<=", "≥", "≤", "≠", "≈", "="}


def guess_type(token: str) -> str:
    if token is None:
        return "blank"
    
    tok = token.strip()
    if tok == "":
        return "blank"
    
    lower = tok.lower()

    # --- 1. 优先判定日期 (覆盖更多英文格式) ---
    # 匹配: 2023-10-12, 10/12/2023, Jan 10, 2023
    if re.fullmatch(r"\d{4}[-/]\d{1,2}[-/]\d{1,2}", tok) or \
       re.fullmatch(r"\d{1,2}[-/]\d{1,2}[-/]\d{2,4}", tok) or \
       re.match(r"^(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s+\d{1,2},?\s+\d{4}$", lower):
        return "date"

    # --- 2. 货币 (更严格) ---
    # 必须包含货币符号，且可能有数字
    if re.match(r"^[$£€¥]s?[\d,.]*$", tok) or re.match(r"^[\d,.]*[$£€¥]$", tok):
        return "money"

    # --- 3. 数字 + 单位 (扩展单位列表) ---
    # 增加常见物理/货币单位
    units = r"(%|kg|g|mg|µg|km|m|cm|mm|ml|l|°c|°f|k|hz|khz|mhz|ghz|kb|mb|gb|tb|s|sec|min|hr|usd|eur|cny|aud|cad)"
    if re.fullmatch(r"-?[\d,.]+\s*" + units, lower):
        return "number+unit"

    # --- 4. 纯数字 (避免 IP 或 乱码) ---
    # 允许负号，允许小数，允许千分位，但不允许连续点 (..)
    # 排除纯序号如 "1." (往往是列表项，算 word 或 structure 更合适，这里先归为 number)
    if re.fullmatch(r"-?\$?\d{1,3}(,\d{3})*(\.\d+)?%?", tok) or re.fullmatch(r"\d+", tok):
        return "number"

    # --- 5. 比较符 & 否定词 ---
    if lower in NEG_WORDS:
        return "negation"
    if tok in COMPARATORS:
        return "comparator"

    # --- 6. 数学符号 (修复 Bug：排除连字符单词) ---
    # 只有当它是单个符号，或者明显是算式一部分时才算
    # 排除 "high-level" 中的 "-", 但保留 "a-b" (如果是分词分开了通常是独立的 -)
    # 更加安全的做法：只匹配单字符数学符，或者由数学符组成的串
    if re.fullmatch(r"[+×÷=<>≠≤≥≈±^|/]", tok) or \
       (tok == "-" and len(tok) == 1): # 只有单独的 - 才是减号，在单词里不算
        return "math_symbol"

    # --- 7. 标点 ---
    if re.fullmatch(r"[.,;:!?()\[\]{}""''`]+", tok): # 增加引号
        return "punct"

    # 默认归为词
    return "word"


def process(in_path, out_path, tag):
    print(f"\n--- 处理 {tag}: {in_path} ---")
    counts = {}

    with open(in_path, "r", encoding="utf-8") as fin, \
         open(out_path, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)

            # 优先用 gt_token，没有就用 pred_token
            tok = rec.get("gt_token") or rec.get("pred_token") or ""
            t = guess_type(tok)
            rec["type"] = t

            counts[t] = counts.get(t, 0) + 1

            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"写入带类型的错误日志: {out_path}")
    print("类型分布：")
    for k, v in sorted(counts.items(), key=lambda x: -x[1]):
        print(f"  {k:12s} : {v}")


def main():
    if os.path.exists(IN_VT64):
        process(IN_VT64, OUT_VT64, "vt64")
    else:
        print("⚠ 找不到 fox100_errors_vt64.jsonl")

    if os.path.exists(IN_VT100):
        process(IN_VT100, OUT_VT100, "vt100")
    else:
        print("⚠ 找不到 fox100_errors_vt100.jsonl")


if __name__ == "__main__":
    main()
