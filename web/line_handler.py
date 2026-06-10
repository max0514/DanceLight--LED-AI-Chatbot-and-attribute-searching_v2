"""LINE Messaging API integration for the dancelight RAG.

Mounts at POST /callback on the Flask app. On each text-message event,
calls rag.engine.search() and replies with a Flex carousel of 5 product
cards (page thumbnail + specs + LLM reason).

Required env vars (loaded by .env via web/app.py):
    LINE_CHANNEL_SECRET            — used for X-Line-Signature HMAC
    LINE_CHANNEL_ACCESS_TOKEN      — long-lived token from LINE Developers Console
    LINE_PUBLIC_HOST (optional)    — public HTTPS host for image URLs in the
                                     Flex bubbles (defaults to Tailscale Funnel)
    LINE_DEFAULT_EDITION (optional) — '21st' or '22nd', defaults to engine default

Commands the user can prefix their query with:
    /21 <query>     — search 21st edition only
    /22 <query>     — search 22nd edition only
    /help           — short usage blurb
"""
from __future__ import annotations

import os
from typing import Iterable

from flask import Flask, abort, request

from linebot.v3 import WebhookHandler
from linebot.v3.exceptions import InvalidSignatureError
from linebot.v3.messaging import (
    ApiClient,
    Configuration,
    FlexBubble,
    FlexBox,
    FlexButton,
    FlexCarousel,
    FlexComponent,
    FlexImage,
    FlexMessage,
    FlexText,
    MessagingApi,
    ReplyMessageRequest,
    TextMessage,
    URIAction,
)
from linebot.v3.webhooks import MessageEvent, TextMessageContent

from rag import engine as rag_engine


# ---------- configuration -------------------------------------------------

DEFAULT_PUBLIC_HOST = "https://pytorch-backup.tail3bc6a4.ts.net"

CHANNEL_SECRET = os.environ.get("LINE_CHANNEL_SECRET", "").strip()
CHANNEL_ACCESS_TOKEN = os.environ.get("LINE_CHANNEL_ACCESS_TOKEN", "").strip()
PUBLIC_HOST = os.environ.get("LINE_PUBLIC_HOST", DEFAULT_PUBLIC_HOST).rstrip("/")
DEFAULT_EDITION = os.environ.get("LINE_DEFAULT_EDITION", rag_engine.DEFAULT_EDITION)

_handler: WebhookHandler | None = None
_config: Configuration | None = None
if CHANNEL_SECRET and CHANNEL_ACCESS_TOKEN:
    _handler = WebhookHandler(CHANNEL_SECRET)
    _config = Configuration(access_token=CHANNEL_ACCESS_TOKEN)


# ---------- Flex builder --------------------------------------------------

_SPEC_LABELS = [
    ("models",      "型號"),
    ("wattages",    "瓦數 (W)"),
    ("color_temps", "色溫 (K)"),
    ("lumens",      "光通量 (lm)"),
    ("ip_rating",   "IP"),
]


def _spec_row(label: str, value: str) -> FlexBox | None:
    if not value:
        return None
    return FlexBox(
        layout="baseline", spacing="sm",
        contents=[
            FlexText(text=label, flex=2, size="sm", color="#94a3b8"),
            FlexText(text=str(value)[:60], flex=5, size="sm",
                     color="#0f172a", wrap=True),
        ],
    )


def _bubble_for(doc: dict, edition: str, idx: int) -> FlexBubble:
    page = int(doc.get("page", 0) or 0)
    img_url = f"{PUBLIC_HOST}/api/page_image/{edition}/{page}.png?dpi=120"
    rank = doc.get("rank_label") or (f"備選 {idx}" if idx else "★ 推薦")
    name = (doc.get("name") or "").strip() or f"型錄 p.{page}"
    reason = (doc.get("reason") or "").strip()

    body_rows: list[FlexComponent] = [
        FlexText(text=rank, size="xs", color="#2563eb", weight="bold"),
        FlexText(text=name, size="md", weight="bold", wrap=True),
        FlexText(text=f"p.{page} · {doc.get('category') or '產品'}",
                 size="xs", color="#64748b"),
    ]
    if reason:
        body_rows.append(
            FlexText(text=reason, size="xs", color="#475569", wrap=True,
                     margin="sm")
        )
    body_rows.append(
        FlexBox(layout="vertical", spacing="xs", margin="md",
                contents=[row for row in (
                    _spec_row(label, doc.get(key, "")) for key, label in _SPEC_LABELS
                ) if row is not None])
    )

    return FlexBubble(
        hero=FlexImage(url=img_url, size="full", aspect_ratio="3:4",
                       aspect_mode="cover",
                       action=URIAction(uri=img_url, label="放大")),
        body=FlexBox(layout="vertical", spacing="sm", contents=body_rows),
        footer=FlexBox(
            layout="vertical", spacing="sm",
            contents=[
                FlexButton(
                    style="link", height="sm",
                    action=URIAction(uri=img_url, label="開啟型錄頁面"),
                ),
            ],
        ),
    )


def _carousel(query: str, docs: list[dict], edition: str) -> FlexMessage:
    bubbles = [_bubble_for(d, edition, i) for i, d in enumerate(docs[:10])]
    return FlexMessage(
        alt_text=f"舞光 {edition} · {query[:30]}",
        contents=FlexCarousel(contents=bubbles),
    )


def _no_results_message(query: str) -> TextMessage:
    return TextMessage(
        text=f"查無符合「{query[:40]}」的型錄產品。\n試試更具體的描述，例如：「15W 4000K IP65 崁燈」",
    )


def _help_message() -> TextMessage:
    return TextMessage(
        text=(
            "舞光 LED 型錄查詢機器人\n\n"
            "直接輸入需求即可，例如：\n"
            "  15W 6500K 崁燈\n"
            "  IP65 防水 50W 投射燈\n"
            "  T-BAR 平板燈 35W\n\n"
            "切換型錄版本：\n"
            "  /21 <需求>  → 2025 21 版\n"
            "  /22 <需求>  → 2026 22 版 (預設)\n"
        ),
    )


def _parse_query(raw: str) -> tuple[str, str | None, bool]:
    """Return (query, edition_override, is_help)."""
    text = (raw or "").strip()
    if not text:
        return "", None, False
    lo = text.lower()
    if lo in {"/help", "help", "?", "？"}:
        return "", None, True
    for tag, ed in (("/21", "21st"), ("/22", "22nd")):
        if lo.startswith(tag):
            rest = text[len(tag):].strip()
            return rest, ed, False
    return text, None, False


# ---------- Flask blueprint -----------------------------------------------

def register(app: Flask) -> bool:
    """Attach /callback to the given Flask app. Returns True if LINE is wired.

    Safe to call when credentials are missing — it just registers a stub that
    returns 503 so the rest of the app boots cleanly.
    """
    if _handler is None or _config is None:
        @app.post("/callback")
        def _line_disabled():
            return ("LINE webhook not configured "
                    "(missing LINE_CHANNEL_SECRET / LINE_CHANNEL_ACCESS_TOKEN)"), 503
        print("[line] disabled — set LINE_CHANNEL_SECRET + LINE_CHANNEL_ACCESS_TOKEN")
        return False

    @app.post("/callback")
    def line_callback():
        signature = request.headers.get("X-Line-Signature", "")
        body = request.get_data(as_text=True)
        try:
            _handler.handle(body, signature)
        except InvalidSignatureError:
            abort(403)
        return "OK", 200

    @_handler.add(MessageEvent, message=TextMessageContent)
    def _on_message(event):
        raw_text = event.message.text or ""
        query, edition_override, is_help = _parse_query(raw_text)
        if is_help or not query:
            messages = [_help_message()]
        else:
            edition = edition_override or DEFAULT_EDITION
            try:
                docs = rag_engine.search(query, top_k=5, edition=edition)
            except Exception as e:
                print(f"[line] search failed: {type(e).__name__}: {e}")
                messages = [TextMessage(
                    text=f"檢索失敗：{type(e).__name__}。請稍後再試。"
                )]
            else:
                messages = [_carousel(query, docs, edition)] if docs else [
                    _no_results_message(query)]

        with ApiClient(_config) as api_client:
            MessagingApi(api_client).reply_message_with_http_info(
                ReplyMessageRequest(
                    reply_token=event.reply_token,
                    messages=messages,
                )
            )

    print(f"[line] /callback registered (public host: {PUBLIC_HOST}, "
          f"default edition: {DEFAULT_EDITION})")
    return True
