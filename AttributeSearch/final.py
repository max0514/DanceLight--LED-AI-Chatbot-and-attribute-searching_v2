#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
final.py
Gradio 版 — 讀取 merged_products_with_series.json，
輸入「系列關鍵字」＋屬性篩選，列出對應型號
"""

import os
import json
import gradio as gr

# ======== 讀取 JSON ========
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(BASE_DIR, "merged_products_with_series.json")  # 如有不同檔名在這裡改

def load_products():
    if not os.path.exists(DATA_FILE):
        return [], f"❌ 找不到 {DATA_FILE}，請先確認檔案存在。"
    try:
        with open(DATA_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            return [], "❌ 檔案格式錯誤：最外層應為陣列(list)。"
        return data, f"✅ 已載入 {len(data)} 筆資料。"
    except Exception as e:
        return [], f"❌ 載入失敗：{e}"

products, load_msg = load_products()

# ======== 篩選邏輯（系列 + 屬性） ========
def filter_products(
    series_keyword,
    watt_lo, watt_hi,
    cct_lo, cct_hi,
    beam_lo, beam_hi,
    lumen_lo, lumen_hi,
    price_lo, price_hi,
    topk
):
    if not products:
        return "⚠️ 尚未載入產品資料。"

    base = products

    # 1) 系列關鍵字（模糊比對）
    if series_keyword and series_keyword.strip():
        q = series_keyword.strip().lower()  # 模糊查詢 + 全部小寫比對
    
    base = [
        p for p in products
        if q in str(p.get("series", "")).lower()
        or q in str(p.get("model", "")).lower()
    ]

    if not base:
        return f"❌ 找不到與「{series_keyword}」相關的系列 / 型號。"


    # 2) 數值屬性篩選
    def num(v):
        try:
            return float(v)
        except:
            return 0.0

    result = []
    for p in base:
        w  = num(p.get("watt", 0))
        c  = num(p.get("cct", 0))
        b  = num(p.get("beam", 0))
        l  = num(p.get("lumen", 0))
        pr = num(p.get("price", 0))

        if not (watt_lo  <= w  <= watt_hi):   continue
        if not (cct_lo   <= c  <= cct_hi):    continue
        if not (beam_lo  <= b  <= beam_hi):   continue
        if not (lumen_lo <= l  <= lumen_hi):  continue
        if not (price_lo <= pr <= price_hi):  continue

        result.append(p)

    if not result:
        if series_keyword and series_keyword.strip():
            return f"❌ 系列關鍵字「{series_keyword}」下沒有符合屬性條件的產品。"
        else:
            return "❌ 沒有任何產品符合屬性條件。"

    # 3) 輸出格式
    lines = [f"### 篩選結果：共 {len(result)} 筆（顯示前 {int(topk)} 筆）\n"]
    for it in result[:int(topk)]:
        lines.append(
            f"- **系列：{it.get('series','未標示系列')}**｜"
            f"型號：`{it.get('model','未命名')}` | "
            f"功率：{it.get('watt','?')}W | "
            f"色溫：{it.get('cct','?')}K | "
            f"光束角：{it.get('beam','?')}° | "
            f"光通量：{it.get('lumen','?')}lm | "
            f"價格：{it.get('price','?')} 元"
        )
    return "\n".join(lines)

# ======== Gradio 介面 ========
with gr.Blocks(title="燈具系列篩選系統") as demo:
    gr.Markdown("# 💡 燈具系列 → 型號篩選系統")
    gr.Markdown(load_msg)

    gr.Markdown("## 🧾 先輸入系列關鍵字，再用屬性篩選型號")
    series_input = gr.Textbox(
        label="系列關鍵字（可留空）",
        placeholder="例如：排燈、軌道、平板、崁燈…（模糊搜尋，打「排燈」就會抓到所有含排燈的系列與型號）"
    )

    with gr.Row():
        watt_lo = gr.Slider(0, 200, 0, step=1, label="功率最小 (W)")
        watt_hi = gr.Slider(0, 200, 200, step=1, label="功率最大 (W)")
    with gr.Row():
        cct_lo = gr.Slider(2000, 7000, 2700, step=50, label="色溫最小 (K)")
        cct_hi = gr.Slider(2000, 7000, 6500, step=50, label="色溫最大 (K)")
    with gr.Row():
        beam_lo = gr.Slider(0, 120, 0, step=1, label="光束角最小 (°)")
        beam_hi = gr.Slider(0, 120, 120, step=1, label="光束角最大 (°)")
    with gr.Row():
        lumen_lo = gr.Slider(0, 15000, 0, step=10, label="光通量最小 (lm)")
        lumen_hi = gr.Slider(0, 15000, 15000, step=10, label="光通量最大 (lm)")
    with gr.Row():
        price_lo = gr.Slider(0, 200000, 0, step=100, label="價格最小")
        price_hi = gr.Slider(0, 200000, 200000, step=100, label="價格最大")
    topk = gr.Slider(1, 50, 20, step=1, label="最多顯示筆數")

    btn_filter = gr.Button("開始篩選", variant="primary")
    filter_output = gr.Markdown()

    btn_filter.click(
        filter_products,
        inputs=[
            series_input,
            watt_lo, watt_hi,
            cct_lo, cct_hi,
            beam_lo, beam_hi,
            lumen_lo, lumen_hi,
            price_lo, price_hi,
            topk
        ],
        outputs=[filter_output]
    )

if __name__ == "__main__":
    demo.launch()
