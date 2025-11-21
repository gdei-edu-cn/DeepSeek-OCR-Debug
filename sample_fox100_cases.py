# sample_fox100_cases.py
import os
import json
import random

random.seed(42)


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EXP_DIR = os.path.join(SCRIPT_DIR, "data", "Fox", "exp_fox100")

CRIT_TYPES = {
    "number", "number+unit", "money", "date",
    "negation", "comparator", "math_symbol",
}

def load_typed_errors(path):
    errors = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            errors.append(json.loads(line))
    return errors

def main():
    vt64_path = os.path.join(EXP_DIR, "fox100_errors_vt64_typed.jsonl")
    vt100_path = os.path.join(EXP_DIR, "fox100_errors_vt100_typed.jsonl")

    vt64_errs = load_typed_errors(vt64_path)
    vt100_errs = load_typed_errors(vt100_path)

    # 只要关键类型
    vt64_crit = [e for e in vt64_errs if e.get("type") in CRIT_TYPES]
    vt100_crit = [e for e in vt100_errs if e.get("type") in CRIT_TYPES]

    print("vt64 关键错误数:", len(vt64_crit))
    print("vt100 关键错误数:", len(vt100_crit))
    print()

    random.seed(42)

    def show_sample(errors, mode, k=20):
        n = min(k, len(errors))
        print(f"=== {mode} 随机抽样 {n} 条关键错误 ===")
        for e in random.sample(errors, n):
            img = e["image"]
            t = e.get("type")
            op = e["op"]
            gt_prev = e.get("gt_prev", "")
            gt_tok = e.get("gt_token", "")
            gt_next = e.get("gt_next", "")
            pred_tok = e.get("pred_token", "")

            print(f"[{img}][{mode}][{t}][{op}]")
            print(f"  GT  : ... {gt_prev} {gt_tok} {gt_next} ...")
            print(f"  PRED: ... {pred_tok} ...")
            print()

    show_sample(vt64_crit, "vt64")
    print()
    show_sample(vt100_crit, "vt100")

if __name__ == "__main__":
    main()
