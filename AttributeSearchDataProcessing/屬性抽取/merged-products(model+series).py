#!/usr/bin/env python3
# series_merge.py
# 1. 刪掉 model 欄位是中文的資料（系列標題列）
# 2. 依照 series.json 把 series 名稱寫回每一個型號

import os
import json
import re

# === 基本路徑：以這支 py 檔所在的資料夾為基準 ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 這三個檔案都放在同一個資料夾（例如 c:\DanceLight\merging）
PRODUCTS_FILE = os.path.join(BASE_DIR, "final_attribute_products.json")           # 原本的產品 JSON
SERIES_FILE   = os.path.join(BASE_DIR, "series.json")                    # Excel 轉出的系列對照
OUTPUT_FILE   = os.path.join(BASE_DIR, "final.json")  # 輸出檔案


def has_cjk(text: str) -> bool:
    """檢查字串裡有沒有中文（CJK）字元，有就視為中文 model。"""
    if not isinstance(text, str):
        return False
    return bool(re.search(r"[\u4e00-\u9fff]", text))


def load_json(path: str):
    """讀取 JSON 檔，並做基本檢查。"""
    if not os.path.exists(path):
        raise FileNotFoundError(f" 找不到檔案：{path}")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def main():
    # 1) 讀入兩個 JSON
    print(f"📥 讀取產品資料：{PRODUCTS_FILE}")
    products = load_json(PRODUCTS_FILE)
    if not isinstance(products, list):
        raise ValueError(" merged_products.json 格式錯誤，最外層應該是陣列(list)。")

    print(f"📥 讀取系列對照：{SERIES_FILE}")
    series_map = load_json(SERIES_FILE)   # 預期格式：{ "系列名": ["D-XXXX", "D-YYYY", ...], ... }
    if not isinstance(series_map, dict):
        raise ValueError("series.json 格式錯誤，最外層應該是物件(dict)，內容為 系列名 → 型號列表。")

    # 2) 反轉成 model -> series 的查表
    model_to_series = {}
    for series_name, models in series_map.items():
        if not isinstance(models, (list, tuple)):
            continue
        for m in models:
            code = str(m).strip()
            if not code:
                continue
            # 若同一型號出現在兩個系列，只保留第一次，並印出警告
            if code in model_to_series and model_to_series[code] != series_name:
                print(
                    f"型號 {code} 同時出現在系列 "
                    f"{model_to_series[code]} 和 {series_name}，暫時沿用第一個。"
                )
                continue
            model_to_series[code] = series_name

    print(f"🔗 已建立 model → series 對照，共 {len(model_to_series)} 筆型號。")

    # 3) 清掉 model 是中文的資料，並加上 series 欄位
    cleaned = []
    removed = 0
    added_series = 0

    for item in products:
        if not isinstance(item, dict):
            continue

        model = str(item.get("model", "")).strip()
        if not model:
            # 沒 model 直接丟掉
            continue

        # (1) 如果 model 有中文，視為系列列 → 丟掉
        if has_cjk(model):
            removed += 1
            continue

        # (2) 如果在 series 對照表裡，就加上 series 名稱
        series_name = model_to_series.get(model)
        if series_name:
            if item.get("series") != series_name:
                item["series"] = series_name
                added_series += 1

        cleaned.append(item)

    # 4) 輸出新的 JSON
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(cleaned, f, ensure_ascii=False, indent=2)

    print("處理完成！")
    print(f"   原始資料：{len(products)} 筆")
    print(f"   移除 model 為中文的系列列：{removed} 筆")
    print(f"   成功寫入 series 名稱：{added_series} 筆")
    print(f"輸出檔案：{OUTPUT_FILE}")


if __name__ == "__main__":
    main()
