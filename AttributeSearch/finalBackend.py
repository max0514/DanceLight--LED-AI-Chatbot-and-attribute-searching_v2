# -*- coding: utf-8 -*-
import os
import json
from typing import Any, Dict, List

# =========================
# 路徑設定
# =========================
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(THIS_DIR)
DATA_FILE = os.path.join(ROOT_DIR, "data", "merged_products_with_series.json")

# 全域變數
PRODUCTS: List[dict] = []
LOAD_STATUS: str = "（尚未載入）"

# =========================
# 工具：數字安全轉換
# =========================
def _to_float(v: Any) -> float:
    try:
        return float(v)
    except:
        return 0.0

# =========================
# 讀取資料
# =========================
def load_products(data_file: str = DATA_FILE) -> Dict[str, Any]:
    """
    讀取 JSON 後寫入全域 PRODUCTS。
    """
    global PRODUCTS, LOAD_STATUS

    # 若預設路徑找不到，嘗試在當前目錄找
    if not os.path.exists(data_file):
        fallback_path = os.path.join(THIS_DIR, "merged_products_with_series.json")
        if os.path.exists(fallback_path):
            data_file = fallback_path

    if not os.path.exists(data_file):
        LOAD_STATUS = f"找不到資料檔：{data_file}"
        PRODUCTS = []
        return {"ok": False, "message": LOAD_STATUS}

    try:
        with open(data_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, list):
            return {"ok": False, "message": "檔案格式錯誤：JSON 最外層應為陣列(list)"}

        normalized = []
        for p in data:
            if isinstance(p, dict):
                normalized.append(p)

        PRODUCTS = normalized
        LOAD_STATUS = f"已載入 {len(PRODUCTS)} 筆資料"
        return {"ok": True, "message": LOAD_STATUS}

    except Exception as e:
        LOAD_STATUS = f"載入失敗：{str(e)}"
        PRODUCTS = []
        return {"ok": False, "message": LOAD_STATUS}

# =========================
# 關鍵字比對邏輯
# =========================
def _match_series_keyword(p: dict, series_keyword: str) -> bool:
    q = (series_keyword or "").strip().lower()
    if not q:
        return True

    tokens = [t for t in q.split() if t]
    s = str(p.get("series", "")).lower()
    m = str(p.get("model", "")).lower()

    # 任一 token 命中就算
    return any(t in s or t in m for t in tokens)

# =========================
# 核心篩選功能
# =========================
def filter_products(
    series_keyword: str = "",
    watt_lo: float = 0, watt_hi: float = 200,
    cct_lo: float = 2000, cct_hi: float = 7000,
    beam_lo: float = 0, beam_hi: float = 120,
    lumen_lo: float = 0, lumen_hi: float = 15000,
    price_lo: float = 0, price_hi: float = 200000,
    topk: int = 50
) -> Dict[str, Any]:
    
    # 嘗試載入資料
    if not PRODUCTS:
        load_products()
        if not PRODUCTS:
            return {"ok": False, "message": "尚未載入產品資料或資料檔遺失", "items": []}

    try:
        # 1. 關鍵字過濾
        base = [p for p in PRODUCTS if _match_series_keyword(p, series_keyword)]
        
        # 2. 屬性過濾
        result = []
        for p in base:
            w  = _to_float(p.get("watt", 0))
            c  = _to_float(p.get("cct", 0))
            b  = _to_float(p.get("beam", 0))
            l  = _to_float(p.get("lumen", 0))
            pr = _to_float(p.get("price", 0))

            if not (watt_lo  <= w  <= watt_hi):   continue
            if not (cct_lo   <= c  <= cct_hi):    continue
            if not (beam_lo  <= b  <= beam_hi):   continue
            if not (lumen_lo <= l  <= lumen_hi):  continue
            if not (price_lo <= pr <= price_hi):  continue

            # ----------------------------------------------------
            # FIX: Ensure price is displayed as a number (int)
            # instead of reading the dirty string "exact" from JSON
            # ----------------------------------------------------
            display_price = int(pr) if pr.is_integer() else pr

            result.append({
                "series": p.get("series", ""),
                "model": p.get("model", ""),
                "watt": w,
                "cct": c,
                "beam": b,
                "lumen": l,
                "price": pr,
                # We overwrite 'price_from' with our clean numeric price
                "price_from": display_price, 
                "voltage": p.get("voltage", ""),
                "ip": p.get("ip", "")
            })

        # 3. 數量截斷
        if not result:
            msg = f"找不到符合條件的產品"
            return {"ok": True, "message": msg, "items": []}

        result = result[:int(topk)]
        return {"ok": True, "message": "success", "items": result}

    except Exception as e:
        return {"ok": False, "message": f"篩選過程發生錯誤: {e}", "items": []}