import re

NEG_WORDS = {"not", "no", "never", "none", "cannot", "can't", "n't"}
COMPARATORS = {">", "<", ">=", "<=", "≥", "≤", "≠", "≈", "="}

def guess_type(token: str) -> str:
    if not token or not token.strip(): return "blank"
    tok = token.strip()
    lower = tok.lower()

    # 1. 日期 (增强: 支持 Jan 10, 2023 等)
    if re.fullmatch(r"\d{4}[-/]\d{1,2}[-/]\d{1,2}", tok) or \
       re.fullmatch(r"\d{1,2}[-/]\d{1,2}[-/]\d{2,4}", tok) or \
       re.match(r"^(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s+\d{1,2},?\s+\d{4}$", lower):
        return "date"
    
    # 2. 货币 (严格匹配符号)
    if re.match(r"^[$£€¥]s?[\d,.]*$", tok) or re.match(r"^[\d,.]*[$£€¥]$", tok):
        return "money"
    
    # 3. 数字+单位 (增强单位库)
    units = r"(%|kg|g|mg|µg|km|m|cm|mm|ml|l|°c|°f|k|hz|khz|mhz|ghz|kb|mb|gb|tb|s|sec|min|hr|usd|eur|cny|aud|cad)"
    if re.fullmatch(r"-?[\d,.]+\s*" + units, lower):
        return "number+unit"
    
    # 4. 纯数字 (排除 1.2.3 等版本号)
    if re.fullmatch(r"-?\$?\d{1,3}(,\d{3})*(\.\d+)?%?", tok) or re.fullmatch(r"\d+", tok):
        return "number"
    
    if lower in NEG_WORDS: return "negation"
    if tok in COMPARATORS: return "comparator"
    
    # 5. 数学符号 (严格单字符或特定符号，防止单词连字符误判)
    if re.fullmatch(r"[+×÷=<>≠≤≥≈±^|/]", tok) or (tok == "-" and len(tok) == 1):
        return "math_symbol"
    
    if re.fullmatch(r"[.,;:!?()\[\]\"\"''`{}]+", tok): return "punct"
    
    return "word"

