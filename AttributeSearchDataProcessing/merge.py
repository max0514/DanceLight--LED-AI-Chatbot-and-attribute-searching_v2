#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
merge_prices_with_progress.py
從 data/ 讀取 products_attr.json 與 products_price.json，
先精準再模糊比對，補上 price，顯示進度並輸出 data/merged_products.json。

預設檔名：
  data/products_attr.json
  data/products_price.json
輸出檔名：
  data/merged_products.json

用法：
  python merge_prices_with_progress.py
  python merge_prices_with_progress.py --cutoff 0.87 --no_fuzzy
"""

import os, json, re, argparse, sys
from difflib import get_close_matches

DATA_DIR = "data"
ATTR_FILE = os.path.join(DATA_DIR, "products_attr.json")
PRICE_FILE = os.path.join(DATA_DIR, "products_price.json")
OUT_FILE = os.path.join(DATA_DIR, "merged_products.json")

def canon_model(m: str) -> str:
    """正規化型號：去空白、大寫、處理全形與常見誤讀。"""
    if not m:
        return ""
    s = re.sub(r"\s+", "", str(m)).upper()
    s = (s.replace("Ｏ","O").replace("０","0").replace("１","1").replace("５","5")
           .replace("I","1").replace("O","0"))
    return s

def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"{path} 內容不是陣列。")
    return data

def progress_bar(done, total, width=20):
    percent = int((done / total) * 100) if total else 100
    filled = percent // (100 // width)
    bar = "█" * filled + "-" * (width - filled)
    return f"[{bar}] {percent:3d}% ({done}/{total})"

def main():
    ap = argparse.ArgumentParser(description="合併 data/ 下的屬性與價格 JSON，補上 price 並顯示進度。")
    ap.add_argument("--attr", default=ATTR_FILE, help=f"屬性 JSON 路徑（預設 {ATTR_FILE}）")
    ap.add_argument("--price", default=PRICE_FILE, help=f"價格 JSON 路徑（預設 {PRICE_FILE}）")
    ap.add_argument("--out", default=OUT_FILE, help=f"輸出檔路徑（預設 {OUT_FILE}）")
    ap.add_argument("--cutoff", type=float, default=0.87, help="模糊比對門檻（0~1，預設 0.87）")
    ap.add_argument("--no_fuzzy", action="store_true", help="只做精準對齊，不做模糊比對")
    args = ap.parse_args()

    if not os.path.isfile(args.attr):
        raise SystemExit(f"❌ 找不到屬性檔：{args.attr}")
    if not os.path.isfile(args.price):
        raise SystemExit(f"❌ 找不到價格檔：{args.price}")

    attrs = load_json(args.attr)
    prices = load_json(args.price)

    # 建立價格索引（精準）：canon(model) -> price
    price_map = {}
    for p in prices:
        m = canon_model(p.get("model", ""))
        if not m:
            continue
        # price 允許數字或 '時價'
        if "price" in p:
            # 若重複出現同型號，保守取較小數字；若是字串（時價），優先保留數字
            curr = price_map.get(m)
            val = p["price"]
            if isinstance(val, (int, float)):
                if isinstance(curr, (int, float)):
                    price_map[m] = min(curr, val)
                else:
                    price_map[m] = val
            else:
                if curr is None:
                    price_map[m] = val

    price_keys = list(price_map.keys())

    total = len(attrs)
    exact_upd = fuzzy_upd = 0
    still_empty = 0

    print(f"🔗 開始合併：屬性 {len(attrs)} 筆；價格鍵 {len(price_map)} 個\n")

    # 先做精準比對
    for i, item in enumerate(attrs, start=1):
        cm = canon_model(item.get("model", ""))
        if cm and cm in price_map:
            item["price"] = price_map[cm]
            item["price_from"] = "exact"
            exact_upd += 1
        # 進度
        if i % max(1, total // 20) == 0 or i == total:
            print(f"  • 精準對齊進度 {progress_bar(i, total)}", flush=True)

    # 再做模糊比對（可關閉）
    if not args.no_fuzzy:
        no_price_indices = [idx for idx, it in enumerate(attrs) if "price" not in it or it["price"] in (0, "", None)]
        n_total = len(no_price_indices)
        print(f"\n🌀 進入模糊比對：待補 {n_total} 筆；cutoff={args.cutoff}\n")
        for j, idx in enumerate(no_price_indices, start=1):
            it = attrs[idx]
            cm = canon_model(it.get("model", ""))
            if not cm:
                continue
            match = get_close_matches(cm, price_keys, n=1, cutoff=args.cutoff)
            if match:
                it["price"] = price_map[match[0]]
                it["price_from"] = "fuzzy"
                fuzzy_upd += 1
            if j % max(1, n_total // 20) == 0 or j == n_total:
                print(f"  • 模糊對齊進度 {progress_bar(j, n_total)}", flush=True)

    # 統計仍無價格
    for it in attrs:
        if "price" not in it or it["price"] in (0, "", None):
            still_empty += 1

    # 輸出
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(attrs, f, ensure_ascii=False, indent=2)

    print("\n✅ 合併完成")
    print(f"   - 精準更新：{exact_upd}")
    print(f"   - 模糊更新：{fuzzy_upd}")
    print(f"   - 仍無價格：{still_empty}")
    print(f"💾 已寫出：{args.out}")

if __name__ == "__main__":
    main()
