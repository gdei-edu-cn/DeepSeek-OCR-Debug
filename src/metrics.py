# 数学符号提升至 3.0 (同数字级)
WEIGHTS = {
    "word": 1.0, "punct": 0.5,
    "number": 3.0, "number+unit": 3.0, "money": 3.0, "date": 3.0,
    "math_symbol": 3.0, "negation": 2.0, "comparator": 2.0,
}
CRITICAL_TYPES = {"number", "number+unit", "money", "date", "math_symbol", "negation", "comparator"}

def compute_stats(records):
    total_err = 0; total_weight = 0.0
    crit_err = 0; crit_weight = 0.0
    for rec in records:
        t = rec.get("type", "word")
        w = WEIGHTS.get(t, 1.0)
        total_err += 1; total_weight += w
        if t in CRITICAL_TYPES:
            crit_err += 1; crit_weight += w
    return {"total_err": total_err, "total_weight": total_weight, 
            "critical_err": crit_err, "critical_weight": crit_weight}

