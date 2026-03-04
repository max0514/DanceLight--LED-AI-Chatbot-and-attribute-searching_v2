#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
price_from_images_folder_simple.py
讀取 price/ 資料夾所有截圖（png/jpg/jpeg/webp），用 GPT-4o 抽出 {model, price}，
彙整成「單一 JSON 檔」且不含 image/source 欄位；同時也會把 JSON 印到 stdout。

用法：
  python price_from_images_folder_simple.py
  python price_from_images_folder_simple.py --out prices.json
  python price_from_images_folder_simple.py --drop_timeprice
"""

import os, io, re, json, time, base64, argparse, sys
from PIL import Image
from dotenv import load_dotenv
from openai import OpenAI

# -------------------- API Key --------------------
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise SystemExit("❌ 找不到 OPENAI_API_KEY，請在環境或 .env 設定。")
client = OpenAI(api_key=OPENAI_API_KEY)

# -------------------- Helpers --------------------
def _to_number_or_none(x):
    """把字串中的數字抓出來；若無法解析（例如『時價』）回 None。"""
    if x is None:
        return None
    s = str(x).strip()
    if not s:
        return None
    if any(k in s for k in ["時價", "面議", "洽詢", "電洽", "tba", "TBA"]):
        return None
    s = s.replace(",", "")
    s = re.sub(r"[^\d.\-+]", "", s)
    m = re.search(r"[-+]?\d+(?:\.\d+)?", s)
    return float(m.group(0)) if m else None

def _clean_model(m: str) -> str:
    """正規化型號：去空白、大寫、處理全形與常見誤讀。"""
    if not m:
        return ""
    s = re.sub(r"\s+", "", str(m)).upper()
    s = (s.replace("Ｏ", "O").replace("０", "0").replace("１", "1").replace("５", "5")
           .replace("I", "1").replace("O", "0"))
    return s

def _find_json(blob: str):
    """從模型輸出文字中擷取 JSON 陣列（允許前後有雜訊）。"""
    if not blob:
        return None
    m = re.search(r"\[\s*\{.*?\}\s*\]", blob, flags=re.S)
    if not m:
        m = re.search(r"\{\s*\".*?\}\s*", blob, flags=re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None

def _image_to_data_url(path: str, max_w=2000, quality=90) -> str:
    img = Image.open(path).convert("RGB")
    if img.width > max_w:
        h = int(img.height * (max_w / img.width))
        img = img.resize((max_w, h), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality, optimize=True)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"

# -------------------- VLM Call --------------------
def extract_prices_from_image(path: str, model="gpt-4o", retries=2, keep_non_numeric=True):
    """
    從單張圖片抽出 [{model, price}]。
    - 支援不同表頭（型號/牌價/售價/價格）
    - 支援一張圖多個表格
    - 允許輸出 '時價'
    - 不回傳來源欄位（無 image/page）
    """
    data_url = _image_to_data_url(path)
    system = (
        "你是燈具價格表抽取助手。請從圖片中抽取所有『型號』與『價格』的配對；"
        "表頭可能為『型號』『牌價』『售價』『價格』等，需辨識同義欄位；"
        "若同一張圖有多個表格（左右欄或分區），要全部抽出合併；"
        "對於價格：移除貨幣符號與千分位逗號，輸出純數字；"
        "若價格標示『時價』『面議』『洽詢』等，請輸出 price='時價'；"
        "只輸出 JSON 陣列，不能有任何解釋文字；"
        "範例："
        '[{"model":"LED-1234","price":1999},{"model":"LED-5678","price":"時價"}]'
    )
    user = [
        {"type": "text", "text": "請抽取所有表格中的型號與對應價格；若無資料請輸出 []。"},
        {"type": "image_url", "image_url": {"url": data_url}},
    ]

    last_err = None
    for _ in range(retries + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": system},
                          {"role": "user", "content": user}],
                temperature=0.1,
                max_tokens=1600,
            )
            out = (resp.choices[0].message.content or "").strip()
            js = _find_json(out) or json.loads(out)
            items = js if isinstance(js, list) else [js]

            cleaned = []
            for it in items or []:
                if not isinstance(it, dict):
                    continue
                raw_model = it.get("model", "")
                raw_price = it.get("price", "")
                model_id = _clean_model(raw_model)

                price_num = _to_number_or_none(raw_price)
                if price_num is not None:
                    price_val = float(price_num)
                else:
                    if keep_non_numeric and any(k in str(raw_price) for k in ["時價", "面議", "洽詢", "電洽", "tba", "TBA"]):
                        price_val = "時價"
                    else:
                        continue

                if model_id and (price_val == "時價" or (isinstance(price_val, (int, float)) and price_val > 0)):
                    cleaned.append({"model": model_id, "price": price_val})

            return cleaned

        except Exception as e:
            last_err = e
            time.sleep(0.8)

    if last_err:
        print(f"  ⚠️ 解析失敗：{os.path.basename(path)} -> {last_err}", file=sys.stderr)
    return []

# -------------------- Main --------------------
def main():
    ap = argparse.ArgumentParser(description="讀取 price/ 資料夾所有截圖，抽出 {model, price}，彙整為單一 JSON（不含 image 欄位）。")
    ap.add_argument("--folder", default="price", help="圖片資料夾（預設 price）")
    ap.add_argument("--model", default="gpt-4o", help="OpenAI 模型（預設 gpt-4o）")
    ap.add_argument("--out", default="products_price.json", help="輸出檔名（預設 products_price.json）")
    ap.add_argument("--drop_timeprice", action="store_true", help="丟棄『時價/面議』等非數字價格（預設保留為 '時價'）")
    args = ap.parse_args()

    if not os.path.isdir(args.folder):
        raise SystemExit(f"❌ 找不到資料夾：{args.folder}")

    exts = (".png", ".jpg", ".jpeg", ".webp")
    imgs = [os.path.join(args.folder, f) for f in sorted(os.listdir(args.folder)) if f.lower().endswith(exts)]
    if not imgs:
        raise SystemExit(f"❌ 資料夾 {args.folder} 內沒有 png/jpg/jpeg/webp 圖片。")

    total = len(imgs)
    print(f"💰 從資料夾 {args.folder} 讀取 {total} 張圖片...\n")

    all_prices = []
    for idx, path in enumerate(imgs, start=1):
        print(f"🖼️  {idx}/{total} -> {os.path.basename(path)} ... ", end="")
        sys.stdout.flush()
        items = extract_prices_from_image(
            path, model=args.model, retries=2, keep_non_numeric=(not args.drop_timeprice)
        )
        all_prices.extend(items)
        print(f"{'✅ ' + str(len(items)) + ' 筆' if items else '⚠️ 無效'}")

    # 單一檔案輸出（不產生逐張稽核）
    try:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(all_prices, f, ensure_ascii=False, indent=2)
        print(f"\n💾 已寫出 {args.out}（{len(all_prices)} 筆）\n")
    except Exception as e:
        print(f"\n⚠️ 寫入 {args.out} 失敗：{e}\n", file=sys.stderr)

    # stdout：輸出彙整 JSON
    print(json.dumps(all_prices, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
