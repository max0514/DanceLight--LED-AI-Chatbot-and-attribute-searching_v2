# attribute.py
# Lighting Spec Finder v4.2 — GPT-4o 自動抽取 + JSON 快取（載入/儲存）+ 屬性篩選

import os, io, re, json, time, base64
import gradio as gr
import fitz                          # PyMuPDF：讀 PDF
from PIL import Image                # 圖片處理
from dotenv import load_dotenv
from openai import OpenAI

# ===== 基本設定 =====
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    print("⚠️ 找不到 OPENAI_API_KEY，請在 .env 中設定。")
client = OpenAI(api_key=OPENAI_API_KEY)

# 全域資料
products = []                          # 解析或載入後的所有產品
DEFAULT_JSON = "merged_products.json"  # 解析完成自動輸出的檔名


# =========================
# 共用工具
# =========================
def _find_json(s: str):
    """從模型輸出裡，盡力抓出 JSON 陣列或物件再 loads。抓不到就回 None。"""
    if not s:
        return None
    m = re.search(r"\[\s*\{.*\}\s*\]", s, flags=re.S)  # 先找陣列
    if not m:
        m = re.search(r"\{\s*\".*\}\s*", s, flags=re.S) # 再找物件
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except:
        return None

def _jpeg_data_url_from_page(page: fitz.Page, max_w=1280, quality=80) -> str:
    """把 PDF 頁面轉成 JPEG Data URL，提供給 VLM"""
    pix = page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5))
    img = Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")
    if img.width > max_w:
        h = int(img.height * (max_w / img.width))
        img = img.resize((max_w, h), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality, optimize=True)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"

def _to_number(x):
    """字串數字 → float（去逗號、抓第一個數字片段）"""
    try:
        s = str(x).replace(",", "")
        m = re.search(r"[-+]?\d+(\.\d+)?", s)
        return float(m.group(0)) if m else 0.0
    except Exception:
        return 0.0


# =========================
# GPT-4o 一般規格抽取（一頁）
# =========================
def _gpt_json_from_text(text: str, page_no: int, retries=2):
    """
    只用文字訊息請 gpt-4o 產生 JSON。
    用 response_format 強制回 JSON；仍備援用 _find_json 解析。
    """
    system = (
        "你是燈具規格抽取助手。"
        "只輸出 JSON，不要任何解釋。"
        "如果沒有產品，請輸出空陣列 []。"
    )
    user = (
        f"請從以下文字中抽取產品規格，輸出 JSON 陣列：\n"
        f"[{{\"model\":\"...\",\"watt\":數字,\"cct\":數字,\"beam\":數字,"
        f"\"lumen\":數字,\"cri\":數字或字串,\"ip\":\"...\",\"voltage\":\"...\",\"price\":數字或字串}}]\n\n"
        f"第 {page_no} 頁內容：\n{text[:8000]}"  # 避免超長
    )
    for _ in range(retries+1):
        try:
            resp = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role":"system","content":system},{"role":"user","content":user}],
                temperature=0.2,
                response_format={"type": "json_object"},  # 盡量讓它只回 JSON
                max_tokens=1200,
            )
            out = resp.choices[0].message.content or ""
            js = _find_json(out)
            if js is None:
                try:
                    js = json.loads(out)  # 有些情況直接是合法 JSON
                except:
                    js = None
            if js is None:
                continue
            if isinstance(js, dict):
                if "items" in js and isinstance(js["items"], list):
                    return js["items"]
                else:
                    return [js]
            if isinstance(js, list):
                return js
        except Exception:
            time.sleep(1.2)
    return None

def _gpt_json_from_image(page: fitz.Page, page_no: int, retries=2):
    """
    用圖片（VLM）請 gpt-4o 產生 JSON。作為無文字或純表格頁的備援。
    """
    system = (
        "你是燈具規格抽取助手。"
        "只輸出 JSON，不要任何解釋。"
        "如果沒有產品，請輸出空陣列 []。"
    )
    data_url = _jpeg_data_url_from_page(page)
    user_content = [
        {"type": "text", "text": (
            "從圖片中讀取燈具規格，輸出 JSON 陣列："
            "[{\"model\":\"...\",\"watt\":數字,\"cct\":數字,\"beam\":數字,"
            "\"lumen\":數字,\"cri\":數字或字串,\"ip\":\"...\",\"voltage\":\"...\",\"price\":數字或字串}]"
        )},
        {"type": "image_url", "image_url": {"url": data_url}},
    ]
    for _ in range(retries+1):
        try:
            resp = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role":"system","content":system},
                    {"role":"user","content":user_content}
                ],
                temperature=0.2,
                max_tokens=1200
            )
            out = resp.choices[0].message.content or ""
            js = _find_json(out)
            if js is None:
                try:
                    js = json.loads(out)
                except:
                    js = None
            if js is None:
                continue
            if isinstance(js, dict):
                return [js]
            if isinstance(js, list):
                return js
        except Exception:
            time.sleep(1.2)
    return None


# =========================
# PDF 全文抽取（→ products；同時輸出 JSON）
# =========================
def parse_pdf_with_gpt4o(pdf_input):
    """
    - 逐頁：文字→JSON；失敗→圖片→JSON
    - 每頁印出成功/失敗
    - 結束後把 products 存成 merged_products.json
    """
    global products
    products = []

    # gr.File 會傳入一個物件，取 name；也支援直接傳字串路徑
    pdf_path = pdf_input if isinstance(pdf_input, str) else pdf_input.name

    doc = fitz.open(pdf_path)
    total = len(doc)
    ok, fail = 0, 0

    for i, page in enumerate(doc, start=1):
        text = (page.get_text("text") or "").strip()

        items = None
        if text:  # 先試文字
            items = _gpt_json_from_text(text, i, retries=2)

        if not items:  # 文字失敗 → 用圖
            items = _gpt_json_from_image(page, i, retries=2)

        if items:
            # 正規化欄位型態，避免後續篩選時出錯
            normed = []
            for it in items:
                if not isinstance(it, dict):
                    continue
                d = dict(it)
                for k in ["watt","cct","beam","lumen","price","cri"]:
                    if k in d:
                        d[k] = _to_number(d[k])
                normed.append(d)

            products.extend(normed)
            ok += 1
            print(f"✅ 第 {i}/{total} 頁解析成功：新增 {len(normed)} 筆（累計 {len(products)}）")
        else:
            fail += 1
            print(f"⚠️ 第 {i}/{total} 頁解析失敗")

    # 存 JSON 快取
    try:
        with open(DEFAULT_JSON, "w", encoding="utf-8") as f:
            json.dump(products, f, ensure_ascii=False, indent=2)
        save_msg = f"✅ 已輸出 {DEFAULT_JSON}（{len(products)} 筆）"
    except Exception as e:
        save_msg = f"❌ 輸出 JSON 失敗：{e}"

    return f"完成：成功 {ok} 頁 / 失敗 {fail} 頁；共解析 {len(products)} 筆。\n{save_msg}"


# =========================
# 查詢 / 篩選
# =========================
def ui_search(query: str):
    if not products:
        return "⚠️ 尚未載入任何產品資料。請先解析 PDF 或載入 JSON。"
    if not query or not query.strip():
        return "⚠️ 請輸入型號或關鍵字。"

    query_type = classify_query_with_llm(query)
    lines = []

    if query_type == "series":
        matched = find_by_series_name(query)
        if not matched:
            return f"❌ 找不到與系列「{query}」相關的產品。"
        lines.append(f"### 📚 系列查詢結果：{len(matched)} 筆\n")
        for it in matched[:20]:
            lines.append(
                f"- **{it.get('model','未命名')}** | "
                f"{it.get('watt','?')}W | {it.get('cct','?')}K | "
                f"光束角 {it.get('beam','?')}° | 光通量 {it.get('lumen','?')}lm | "
                f"價格 {it.get('price','?')} 元"
            )
        return "\n".join(lines)

    else:  # 預設當成型號查詢
        q = query.strip().lower()
        matched = []
        for p in products:
            model = str(p.get("model", "")).lower()
            if q in model:
                matched.append(p)

        if not matched:
            return f"❌ 找不到型號「{query}」。"

        lines.append(f"### 🔎 型號查詢結果：{len(matched)} 筆\n")
        for it in matched[:20]:
            lines.append(
                f"- **{it.get('model','未命名')}** | "
                f"{it.get('watt','?')}W | {it.get('cct','?')}K | "
                f"光束角 {it.get('beam','?')}° | 光通量 {it.get('lumen','?')}lm | "
                f"價格 {it.get('price','?')} 元"
            )
        return "\n".join(lines)


def ui_filter(
    series_name,
    watt_lo, watt_hi,
    cct_lo, cct_hi,
    beam_lo, beam_hi,
    lumen_lo, lumen_hi,
    price_lo, price_hi,
    topk
):
    if not products:
        return "⚠️ 尚未載入任何產品資料。請先解析 PDF 或載入 JSON。"

    # 🔍 Step 1. 先依系列過濾（若有輸入）
    if series_name and series_name.strip():
        q = series_name.strip().lower()
        filtered = [p for p in products if q in str(p.get("model", "")).lower()]
        if not filtered:
            return f"❌ 找不到與系列「{series_name}」相關的產品。"
    else:
        filtered = products[:]  # 沒有系列輸入就用全部

    # 🔢 Step 2. 再依屬性篩選
    def num(x): 
        try: return float(x)
        except: return 0

    result = []
    for p in filtered:
        w  = num(p.get("watt", 0))
        c  = num(p.get("cct", 0))
        b  = num(p.get("beam", 0))
        l  = num(p.get("lumen", 0))
        pr = num(p.get("price", 0))

        if not (watt_lo <= w  <= watt_hi):  continue
        if not (cct_lo  <= c  <= cct_hi):   continue
        if not (beam_lo <= b  <= beam_hi):  continue
        if not (lumen_lo<= l  <= lumen_hi): continue
        if not (price_lo<= pr <= price_hi): continue
        result.append(p)

    if not result:
        return f"❌ 系列「{series_name or '全部'}」中沒有符合篩選條件的產品。"

    # 🧾 Step 3. 格式化輸出
    lines = [f"### 篩選結果：系列 {series_name or '（全部）'} 共 {len(result)} 筆（顯示前 {int(topk)} 筆）\n"]
    for it in result[:int(topk)]:
        lines.append(
            f"- **{it.get('model','未命名')}** | "
            f"{it.get('watt','?')}W | {it.get('cct','?')}K | "
            f"光束角 {it.get('beam','?')}° | 光通量 {it.get('lumen','?')}lm | "
            f"價格 {it.get('price','?')} 元"
        )
    return "\n".join(lines)

# ==========================================
# 🔍 智慧系列/型號辨識與篩選輔助模組
# ==========================================
def classify_query_with_llm(user_query: str) -> str:
    """
    使用 GPT 判斷使用者輸入屬於「系列」還是「型號」。
    回傳 'series' 或 'model'
    """
    if not user_query:
        return "unknown"
    try:
        resp = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role":"system","content":"你是一個燈具資料分類助手，只回答 'series' 或 'model'。"},
                {"role":"user","content":f"判斷以下輸入屬於燈具『系列名』還是『型號名』：{user_query}"}
            ],
            temperature=0
        )
        ans = resp.choices[0].message.content.strip().lower()
        if "series" in ans:
            return "series"
        if "model" in ans:
            return "model"
    except Exception as e:
        print(f"LLM 判斷失敗：{e}")
    return "unknown"


def find_by_series_name(series_query: str):
    """
    從 JSON 的文字欄位中找出屬於同系列的產品。
    例如使用者輸入 'T5 節標' 或 'T5BA1'，或者結尾包含"系列"二字 → 找出所有包含這關鍵詞的 model。
    """
    q = series_query.strip().lower()
    matched = []
    for p in products:
        if q in str(p.get("model", "")).lower():
            matched.append(p)
    return matched


# =========================
# JSON 快取：載入
# =========================
def load_products_from_json(path: str = DEFAULT_JSON):
    """從 JSON 載入 products（覆蓋全域）"""
    global products
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            products = data
            return f"✅ 已載入 {path}：{len(products)} 筆"
        return "❌ JSON 格式不是陣列。"
    except Exception as e:
        return f"❌ 載入 JSON 失敗：{e}"

def load_products_from_uploaded_json(file_obj):
    """Gradio 上傳 JSON 載入"""
    if not file_obj:
        return "⚠️ 請先上傳 JSON 檔。"
    return load_products_from_json(file_obj.name)


# =========================
# Gradio UI
# =========================
with gr.Blocks(title="Lighting Spec Finder v4.2 — GPT-4o + JSON 快取") as demo:
    gr.Markdown("# 💡 Lighting Spec Finder v4.2 — 解析後可存 JSON，重啟直接載入使用")

    # A. 先載入 JSON（重啟後建議用）
    gr.Markdown("## A. 載入現有 JSON（重啟後免重跑）")
    with gr.Row():
        btn_load_default = gr.Button("📂 載入 merged_products.json")
        json_upload = gr.File(label="或上傳自訂 JSON（陣列格式）", file_types=[".json"])
        btn_load_uploaded = gr.Button("📤 載入上傳 JSON")
    status_load = gr.Markdown("（尚未載入）")
    btn_load_default.click(lambda: load_products_from_json(DEFAULT_JSON), outputs=[status_load])
    btn_load_uploaded.click(load_products_from_uploaded_json, inputs=[json_upload], outputs=[status_load])

    # B. 重新解析 PDF（會自動存 JSON）
    gr.Markdown("## B. 重新解析 PDF（GPT-4o 全抽取，完成後自動輸出 JSON）")
    with gr.Row():
        pdf_input = gr.File(label="上傳 catalog PDF", file_types=[".pdf"], scale=3)
        btn_parse = gr.Button("🚀 開始解析（全文）", scale=1)
    status_parse = gr.Markdown("（未開始）")
    btn_parse.click(parse_pdf_with_gpt4o, inputs=pdf_input, outputs=status_parse)

    # C. 查詢 / 篩選
    gr.Markdown("## C. 查詢 / 篩選")
    with gr.Row():
        query_input = gr.Textbox(label="輸入型號或關鍵字（例如：D-FXTR7N 或 軌道燈）", placeholder="請先載入 JSON 或解析 PDF", scale=4)
        btn_search = gr.Button("查詢", variant="primary", scale=1)
    search_result = gr.Markdown("（尚未查詢）")
    btn_search.click(ui_search, inputs=[query_input], outputs=[search_result])
    
    series_input = gr.Textbox(label="系列名稱（可選）", placeholder="例如：T5、D-T5BA1、OD 系列等，可留空")

    gr.Markdown("### 屬性篩選（雙頭滑桿）")
    with gr.Row():
        watt_lo = gr.Slider(0,200,0,step=1,label="功率最小 W")
        watt_hi = gr.Slider(0,200,200,step=1,label="功率最大 W")
    with gr.Row():
        cct_lo = gr.Slider(2000,7000,2700,step=50,label="色溫最小 K")
        cct_hi = gr.Slider(2000,7000,6500,step=50,label="色溫最大 K")
    with gr.Row():
        beam_lo = gr.Slider(0,120,0,step=1,label="光束角最小 °")
        beam_hi = gr.Slider(0,120,120,step=1,label="光束角最大 °")
    with gr.Row():
        lumen_lo = gr.Slider(0,10000,0,step=10,label="光通量最小 lm")
        lumen_hi = gr.Slider(0,10000,10000,step=10,label="光通量最大 lm")
    with gr.Row():
        price_lo = gr.Slider(0,100000,0,step=100,label="價格最小")
        price_hi = gr.Slider(0,100000,100000,step=100,label="價格最大")
    with gr.Row():
        topk = gr.Slider(1,20,10,step=1,label="最多顯示筆數")

    btn_filter = gr.Button("開始篩選", variant="primary")
    filter_result = gr.Markdown()
    btn_filter.click(
        ui_filter,
        inputs=[series_input,watt_lo, watt_hi, cct_lo, cct_hi, beam_lo, beam_hi, lumen_lo, lumen_hi, price_lo, price_hi, topk],
        outputs=[filter_result]
    )

if __name__ == "__main__":
    demo.launch()
