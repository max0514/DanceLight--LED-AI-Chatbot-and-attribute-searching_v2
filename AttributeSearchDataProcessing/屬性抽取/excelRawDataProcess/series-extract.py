import os
import json
import pandas as pd

# ==========================================
# 自動讀取 catalogs 資料夾內的 Excel
# ==========================================
def find_excel_in_catalogs(folder="catalogs"):
    if not os.path.exists(folder):
        raise FileNotFoundError(f"找不到資料夾：{folder}")

    for f in os.listdir(folder):
        if f.lower().endswith(".xlsx"):
            return os.path.join(folder, f)

    raise FileNotFoundError("catalogs 中沒有找到 .xlsx 檔案")


# ==========================================
# 主要處理函式：Excel → JSON
# ==========================================
def excel_to_json(excel_path, output_path):
    print(f"📘 正在讀取 Excel：{excel_path}")

    df = pd.read_excel(excel_path)

    # 嘗試找欄位
    possible_model_cols = ["型號", "產品型號", "Product Code", "Model"]
    possible_name_cols = ["品名", "名稱", "Name"]

    model_col = next((c for c in df.columns if c in possible_model_cols), None)
    name_col = next((c for c in df.columns if c in possible_name_cols), None)

    if not model_col or not name_col:
        raise ValueError(f"Excel 必須包含：型號 / 品名 欄位，目前欄位為：{list(df.columns)}")

    # --- 建立 series -> models 對照 ---
    series_dict = {}

    for _, row in df.iterrows():
        model = str(row[model_col]).strip()
        name = str(row[name_col]).strip()

        if not model or model.lower() == "nan":
            continue

        # 系列名稱 = 品名空白前的中文字（依你 Excel 樣式）
        # 例如：米開朗柔性軌道-12W投射排燈 → 系列：米開朗柔性軌道
        series = name.split("-")[0].strip()

        if series not in series_dict:
            series_dict[series] = []

        series_dict[series].append(model)

    # --- 輸出 JSON ---
    print(f" 輸出 JSON：{output_path}")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(series_dict, f, ensure_ascii=False, indent=2)

    print(f"✅ 完成！共 {len(series_dict)} 個系列。")
    return output_path


# ==========================================
# 主程式：自動偵測 Excel → JSON
# ==========================================
if __name__ == "__main__":
    print("🔍 自動搜尋 catalogs 資料夾中的 Excel...")

    excel_path = find_excel_in_catalogs("catalogs")
    output_path = os.path.join("catalogs", "series.json")

    excel_to_json(excel_path, output_path)
