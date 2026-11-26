"""Kharcha Paani — log expenses in plain words, by voice, or from a receipt photo.
Run locally:  streamlit run app.py
"""
import datetime as dt
import html
import io
import json
import re
import uuid
from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image

# ---------------------------------------------------------------- setup
st.set_page_config(page_title="Kharcha Paani", page_icon="📒", layout="centered")

CATS = ["Food", "Travel", "Misc"]
COLORS = {"Food": "#d9822b", "Travel": "#2a6f97", "Misc": "#8d5a97"}
IST = dt.timezone(dt.timedelta(hours=5, minutes=30))
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)


def today() -> dt.date:
    return dt.datetime.now(IST).date()


def secret(name, default=None):
    try:
        return st.secrets.get(name, default)
    except Exception:  # no secrets.toml at all
        return default


API_KEY = secret("GEMINI_API_KEY")
MODEL = secret("GEMINI_MODEL", "gemini-flash-latest")


@st.cache_resource
def gemini_client(key):
    from google import genai
    return genai.Client(api_key=key)


client = gemini_client(API_KEY) if API_KEY else None


# ---------------------------------------------------------------- storage
def khata_file(name: str) -> Path:
    safe = re.sub(r"[^a-z0-9_-]", "", name.lower()) or "default"
    return DATA_DIR / f"{safe}.json"


def load(name):
    try:
        return json.loads(khata_file(name).read_text())
    except Exception:
        return {"entries": [], "budget": {"total": 0, "Food": 0, "Travel": 0, "Misc": 0}}


def save(name, db):
    khata_file(name).write_text(json.dumps(db, ensure_ascii=False, indent=1))


# ---------------------------------------------------------------- offline parser (no AI needed)
KW = {
    "Food": r"chai|tea|coffee|swiggy|zomato|samosa|lunch|dinner|breakfast|nashta|khana|food|pizza|biryani|dosa|maggi|thali|snack|juice|sabzi|doodh|milk|grocer|blinkit|zepto|kirana|fruit|paneer|momos|vada|pav|cafe|restaurant|dhaba|mithai|chips|ice ?cream|bigbasket",
    "Travel": r"auto|uber|ola|rapido|metro|bus|cab|taxi|train|irctc|petrol|diesel|fuel|parking|toll|flight|rickshaw|fastag|bike",
}


def guess(text):
    for cat, pat in KW.items():
        if re.search(pat, text, re.I):
            return cat
    return "Misc"


def quick_parse(text):
    date = today()
    if re.search(r"\b(kal|yesterday)\b", text, re.I):
        date = today() - dt.timedelta(days=1)
    items = []
    for part in re.split(r",|\n|;|\+|\baur\b|\band\b", text, flags=re.I):
        part = re.sub(r"\b(kal|aaj|yesterday|today|ka|ke|ki|mein|me|for|rs\.?|inr|rupees?|rupaye)\b|₹", " ", part, flags=re.I)
        m = re.search(r"\d+(?:\.\d+)?", part)
        if not m:
            continue
        name = re.sub(r"\s+", " ", part.replace(m.group(), " ")).strip() or "Kharcha"
        items.append({"item": name.capitalize(), "amount": float(m.group()), "category": guess(name), "date": str(date)})
    return items


# ---------------------------------------------------------------- AI (Gemini)
def rules():
    return f"""Categories: Food (chai, snacks, meals, groceries, Swiggy/Zomato/Blinkit/Zepto, restaurants), Travel (auto, cab, Uber/Ola/Rapido, metro, bus, train, petrol, parking, toll, flights), Misc (everything else: shopping, bills, recharge, medicine, movies, rent).
The input may be English, Hindi or Hinglish (e.g. "kal metro 45 aur samosa 30", "do chai 40"). "kal" = yesterday, "parso" = day before yesterday. Today is {today()}.
The amount is the rupee total for that line.
Reply with ONLY JSON: {{"items":[{{"item":"short readable name","amount":number,"category":"Food|Travel|Misc","date":"YYYY-MM-DD"}}]}}"""


def clean(items):
    out = []
    for x in items or []:
        try:
            amt = abs(float(x.get("amount") or 0))
        except (TypeError, ValueError):
            continue
        if amt <= 0:
            continue
        cat = x.get("category") if x.get("category") in CATS else guess(str(x.get("item", "")))
        date = str(x.get("date") or "")
        date = date if re.fullmatch(r"\d{4}-\d\d-\d\d", date) else str(today())
        out.append({"item": str(x.get("item") or "Kharcha")[:80], "amount": amt, "category": cat, "date": date})
    return out


FALLBACKS = [m.strip() for m in str(secret("GEMINI_FALLBACKS", "gemini-flash-lite-latest")).split(",") if m.strip()]


def ask_gemini(parts):
    """Call Gemini; on busy/overloaded errors retry, then try lighter fallback models."""
    import time
    from google.genai import errors, types
    last = None
    for model in [MODEL] + [m for m in FALLBACKS if m != MODEL]:
        for attempt in range(3):
            try:
                r = client.models.generate_content(
                    model=model,
                    contents=parts,
                    config=types.GenerateContentConfig(response_mime_type="application/json", temperature=0),
                )
                m = re.search(r"\{[\s\S]*\}", r.text or "")
                data = json.loads(m.group()) if m else {}
                return clean(data.get("items", data if isinstance(data, list) else []))
            except errors.APIError as e:
                last = e
                if e.code in (429, 500, 503, 504):  # busy: wait and retry
                    time.sleep(1.5 * (attempt + 1))
                    continue
                break  # e.g. 404 model not found: go to the next model
    raise last


def parse_text(text):
    if not client:
        return quick_parse(text), "Sorted with the offline parser (no Gemini key)."
    try:
        items = ask_gemini([f'Split this expense note into separate items.\n{rules()}\n\nNote: """{text}"""'])
        return (items, "Sorted by Gemini.") if items else (quick_parse(text), "Gemini found nothing, used the offline parser.")
    except Exception as e:
        return quick_parse(text), f"Gemini error, used the offline parser: {e}"


def parse_voice(audio_bytes, mime="audio/wav"):
    from google.genai import types
    items = ask_gemini([
        types.Part.from_bytes(data=audio_bytes, mime_type=mime),
        "This is a voice note listing expenses, spoken in English, Hindi or Hinglish. "
        "Transcribe it, then split it into items.\n" + rules(),
    ])
    return items


def parse_receipt(image_bytes):
    from google.genai import types
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img.thumbnail((1600, 1600))
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=85)
    return ask_gemini([
        types.Part.from_bytes(data=buf.getvalue(), mime_type="image/jpeg"),
        "This image is a shop bill, restaurant receipt, or UPI payment screenshot (GPay, PhonePe, Paytm). "
        "For a UPI screenshot return ONE item: the payee name and the amount paid, dated from the screenshot if visible. "
        "For a bill return ONE item: the merchant and the final total paid (after tax/discount).\n" + rules(),
    ])


# ---------------------------------------------------------------- helpers
def rs(n):
    n = round(n)
    s = str(abs(n))
    if len(s) > 3:  # Indian grouping: 1,23,456
        s = re.sub(r"(\d)(?=(\d\d)+$)", r"\1,", s[:-3]) + "," + s[-3:]
    return ("-" if n < 0 else "") + "₹" + s


def month_stats(db, ym):
    rows = [e for e in db["entries"] if e["date"].startswith(ym)]
    by = {c: sum(e["amount"] for e in rows if e["category"] == c) for c in CATS}
    return rows, by, sum(by.values())


def warnings(by, total, b):
    out = []
    if b.get("total"):
        p = total / b["total"]
        if p > 1:
            out.append(("error", f"Budget cross ho gaya! {rs(total - b['total'])} over your {rs(b['total'])} limit."))
        elif p >= 0.8:
            out.append(("warning", f"Dhyan do: {round(p * 100)}% of this month's budget is used."))
    for c in CATS:
        if b.get(c):
            p = by[c] / b[c]
            if p > 1:
                out.append(("error", f"{c} is {rs(by[c] - b[c])} over its {rs(b[c])} limit."))
            elif p >= 0.8:
                out.append(("warning", f"{c}: {round(p * 100)}% of its {rs(b[c])} limit used."))
    return out


def set_draft(items, source, note):
    st.session_state.draft = pd.DataFrame(items, columns=["item", "amount", "category", "date"])
    st.session_state.source = source
    st.session_state.note = note


# ---------------------------------------------------------------- sidebar: khata + budget
with st.sidebar:
    st.header("📒 Your khata")
    name = st.text_input("Khata name", value=st.session_state.get("khata", "meri-khata"),
                         help="Each name keeps its own entries and budget.")
    st.session_state.khata = name
    db = load(name)

    st.subheader("💰 Monthly budget")
    b = db["budget"]
    with st.form("budget"):
        total = st.number_input("Total ₹", min_value=0, step=500, value=int(b.get("total") or 0))
        cols = st.columns(3)
        lim = {c: cols[i].number_input(f"{c} ₹", min_value=0, step=100, value=int(b.get(c) or 0)) for i, c in enumerate(CATS)}
        if st.form_submit_button("Save budget", width="stretch"):
            db["budget"] = {"total": total or sum(lim.values()), **lim}
            save(name, db)
            st.success("Budget saved")
    st.caption("Leave a category at 0 to skip its limit. Applies to every month.")

    st.subheader("⬇️ Backup")
    st.download_button("Download khata (JSON)", json.dumps(db, ensure_ascii=False, indent=1),
                       file_name=f"{name}.json", mime="application/json", width="stretch")
    up = st.file_uploader("Restore from backup", type="json")
    if up and st.button("Restore", width="stretch"):
        save(name, json.loads(up.read()))
        st.rerun()

    st.caption("AI: " + (f"Gemini `{MODEL}` ✅" if client else "off, add GEMINI_API_KEY to secrets"))

# ---------------------------------------------------------------- header + month summary
st.title("📒 Kharcha Paani")
st.caption("Bol do ya likh do, hisaab ho jaayega.")

months = sorted({e["date"][:7] for e in db["entries"]} | {str(today())[:7]}, reverse=True)
ym = st.selectbox("Month", months, format_func=lambda m: dt.date.fromisoformat(m + "-01").strftime("%B %Y"))
rows, by, total = month_stats(db, ym)
b = db["budget"]

c1, c2 = st.columns(2)
c1.metric("Spent this month", rs(total), f"{len(rows)} entries", delta_color="off")
if b.get("total"):
    left = b["total"] - total
    if ym == str(today())[:7]:
        nxt = dt.date(today().year + today().month // 12, today().month % 12 + 1, 1)
        days = (nxt - today()).days
    else:
        days = 0
    per_day = f"≈ {rs(left / days)}/day for {days} days" if left > 0 and days else None
    c2.metric("Budget left" if left >= 0 else "Over budget", rs(abs(left)), per_day, delta_color="off")
    st.progress(min(total / b["total"], 1.0))
else:
    c2.metric("Budget", "Not set", "Set it in the sidebar", delta_color="off")

st.markdown(" &nbsp; ".join(
    f"<span style='color:{COLORS[c]}'>●</span> **{c}** {rs(by[c])}" + (f" / {rs(b[c])}" if b.get(c) else "")
    for c in CATS), unsafe_allow_html=True)
for kind, text in warnings(by, total, b):
    getattr(st, kind)(text)

st.divider()

# ---------------------------------------------------------------- add expenses
st.subheader("Aaj ka kharcha")
t_type, t_voice, t_scan = st.tabs(["✍️ Type", "🎙 Bolo", "📷 Scan bill"])

with t_type:
    examples = ["chai 20, auto 60, Swiggy 250", "kal metro 45 aur samosa 30", "dawai 180, Zomato 320, Uber 210"]
    ex = st.pills("Try an example", examples, key="ex")
    text = st.text_area("Expenses in plain words", value=ex or "", placeholder="chai 20, auto 60, Swiggy 250")
    if st.button("Sort it out", type="primary") and text.strip():
        with st.spinner("Hisaab laga rahe hain…"):
            items, note = parse_text(text)
        set_draft(items, "text", note)

with t_voice:
    if not client:
        st.info("Voice needs a Gemini API key.")
    else:
        audio = st.audio_input("Speak your expenses, e.g. “chai bees, auto saath”")
        if audio and st.button("Use this recording", type="primary"):
            with st.spinner("Sun rahe hain…"):
                try:
                    set_draft(parse_voice(audio.getvalue(), audio.type or "audio/wav"), "voice", "Heard and sorted by Gemini.")
                except Exception as e:
                    st.error(f"Couldn't process the audio: {e}")

with t_scan:
    if not client:
        st.info("Receipt scanning needs a Gemini API key.")
    else:
        src = st.radio("Source", ["Upload image", "Use camera"], horizontal=True, label_visibility="collapsed")
        pic = st.file_uploader("Bill photo or UPI screenshot", type=["png", "jpg", "jpeg", "webp"]) \
            if src == "Upload image" else st.camera_input("Snap the bill")
        if pic:
            st.image(pic, width=220)
            if st.button("Read receipt", type="primary"):
                with st.spinner("Reading the receipt…"):
                    try:
                        items = parse_receipt(pic.getvalue())
                        set_draft(items or [{"item": "Receipt", "amount": 0, "category": "Misc", "date": str(today())}],
                                  "scan", "Read from your image. Check the amount." if items else "Couldn't find an amount, fill it in.")
                    except Exception as e:
                        st.error(f"Couldn't read the image: {e}")

# ---------------------------------------------------------------- review + save
if "draft" in st.session_state:
    st.caption(st.session_state.note + " Edit anything below, then save.")
    edited = st.data_editor(
        st.session_state.draft, num_rows="dynamic", width="stretch", key="editor",
        column_config={
            "item": st.column_config.TextColumn("Item", required=True),
            "amount": st.column_config.NumberColumn("Amount ₹", min_value=0, format="₹%d"),
            "category": st.column_config.SelectboxColumn("Category", options=CATS, required=True),
            "date": st.column_config.TextColumn("Date (YYYY-MM-DD)"),
        },
    )
    s1, s2 = st.columns([1, 1])
    if s1.button("✅ Save to khata", type="primary", width="stretch"):
        items = clean(edited.to_dict("records"))
        if items:
            for x in items:
                db["entries"].append({**x, "id": uuid.uuid4().hex[:8], "src": st.session_state.source})
            save(name, db)
            del st.session_state["draft"]
            st.toast(f"Saved {len(items)} entries · {rs(sum(i['amount'] for i in items))}", icon="✅")
            st.rerun()
        else:
            st.warning("Add an amount first.")
    if s2.button("Discard", width="stretch"):
        del st.session_state["draft"]
        st.rerun()

st.divider()

# ---------------------------------------------------------------- ledger
st.subheader("Khata")
q = st.text_input("Search", placeholder="Search entries", label_visibility="collapsed")
show = sorted([e for e in rows if q.lower() in e["item"].lower()], key=lambda e: e["date"], reverse=True)
if not show:
    st.info("Khata khaali hai. Add your first kharcha above." if not rows else "No entries match.")
for day in sorted({e["date"] for e in show}, reverse=True):
    d = dt.date.fromisoformat(day)
    label = "Aaj" if d == today() else "Kal" if d == today() - dt.timedelta(days=1) else d.strftime("%a, %d %b")
    day_rows = [e for e in show if e["date"] == day]
    st.markdown(f"**{label}** · {rs(sum(e['amount'] for e in day_rows))}")
    for e in day_rows:
        a, bcol, c, dcol = st.columns([5, 2, 2, 1])
        a.markdown(f"<span style='color:{COLORS[e['category']]}'>●</span> {html.escape(e['item'])}"
                   + (" <small>(receipt)</small>" if e.get("src") == "scan" else ""), unsafe_allow_html=True)
        new_cat = bcol.selectbox("Category", CATS, index=CATS.index(e["category"]), key="c" + e["id"],
                                 label_visibility="collapsed")
        if new_cat != e["category"]:
            e["category"] = new_cat
            save(name, db)
            st.rerun()
        c.markdown(f"**{rs(e['amount'])}**")
        if dcol.button("✕", key="d" + e["id"], help="Delete"):
            db["entries"] = [x for x in db["entries"] if x["id"] != e["id"]]
            save(name, db)
            st.rerun()
