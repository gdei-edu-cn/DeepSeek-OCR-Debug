# 用于找到最难的5个页面
import sys, os, json, collections

# 路径引导
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if PROJECT_ROOT not in sys.path: sys.path.insert(0, PROJECT_ROOT)

# 你的错误日志文件
ERR_FILE = os.path.join(PROJECT_ROOT, "data", "Fox", "exp_fox100", "fox100_errors_vt64_typed.jsonl")

def main():
    if not os.path.exists(ERR_FILE):
        print(f"❌ 找不到文件: {ERR_FILE}，请先运行 3_tag_errors.py")
        return

    # 统计每张图片的数字错误数
    page_scores = collections.Counter()
    
    with open(ERR_FILE, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            # 只关心 vt64 下的数字错误
            if rec.get("type") in ["number", "money", "number+unit"]:
                page_scores[rec["image"]] += 1

    # 打印前 5 名
    print("\n=== 🚩 数字错误最多的 5 个页面 (The Hell Pages) ===")
    top5 = page_scores.most_common(5)
    for img, count in top5:
        print(f"Image: {img:<20} | Number Errors: {count}")
    
    # 保存列表供下一步使用
    target_images = [img for img, _ in top5]
    print(f"\n✅ 请记住这 5 张图的文件名：{target_images}")

if __name__ == "__main__":
    main()