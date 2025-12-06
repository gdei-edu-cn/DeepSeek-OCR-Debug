import os
import json
import shutil
import sys

# --- 路径引导 ---
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if PROJECT_ROOT not in sys.path: sys.path.insert(0, PROJECT_ROOT)

# 配置
SOURCE_JSON = os.path.join(PROJECT_ROOT, "data", "OmniDocBench", "OmniDocBench.json")
SOURCE_IMG_DIR = os.path.join(PROJECT_ROOT, "data", "OmniDocBench", "images")
TARGET_DIR = os.path.join(PROJECT_ROOT, "data", "OmniDocBench", "exp_table_subset")
os.makedirs(os.path.join(TARGET_DIR, "images"), exist_ok=True)

def main():
    if not os.path.exists(SOURCE_JSON):
        print(f"❌ 找不到文件: {SOURCE_JSON}")
        return

    print(f"📖 正在读取标注: {SOURCE_JSON} ...")
    with open(SOURCE_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)

    selected_pages = []
    count = 0
    target_count = 200 

    for entry in data:
        if count >= target_count: break
        
        # 1. 获取图片路径
        img_name = None
        if 'page_info' in entry and 'image_path' in entry['page_info']:
            img_name = entry['page_info']['image_path']
        
        if not img_name: continue
        img_basename = os.path.basename(img_name)

        # 2. 筛选逻辑：检查 layout_dets 里是否有 table
        layout_dets = entry.get('layout_dets', [])
        if not layout_dets: continue

        is_table = False
        for det in layout_dets:
            cat = str(det.get('category_type', '')).lower()
            if 'table' in cat:
                is_table = True
                break
        
        if is_table:
            src_path = os.path.join(SOURCE_IMG_DIR, img_name)
            tgt_path = os.path.join(TARGET_DIR, "images", img_basename)
            
            if os.path.exists(src_path):
                shutil.copy(src_path, tgt_path)
                
                # 3. 构造 GT (关键修改：增加 None 值检查)
                # 使用 (x.get('order') or 0) 确保即使 order 是 None 也能当作 0 处理
                try:
                    sorted_dets = sorted(layout_dets, key=lambda x: (x.get('order') if x.get('order') is not None else 0))
                except Exception as e:
                    # 万一还有其他奇葩情况，直接不排序了，按原序
                    print(f"⚠️ 排序警告 ({img_basename}): {e}，将使用默认顺序")
                    sorted_dets = layout_dets
                
                # 拼接文本，用换行符分隔
                full_text = "\n".join([d.get('text', '') for d in sorted_dets if d.get('text')])
                
                selected_pages.append({
                    "image": img_basename,
                    "conversations": [
                        {"value": "human"}, 
                        {"value": full_text} 
                    ]
                })
                count += 1

    # 4. 保存
    out_gt = os.path.join(TARGET_DIR, "table_gt.json")
    with open(out_gt, "w", encoding="utf-8") as f:
        json.dump(selected_pages, f, indent=2, ensure_ascii=False)
        
    print(f"\n✅ 筛选完成！")
    print(f"   共筛选出 {len(selected_pages)} 张表格图片")
    print(f"   GT 已保存至: {out_gt}")

if __name__ == "__main__":
    main()