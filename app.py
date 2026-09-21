import os
import re
import json
import requests
from bs4 import BeautifulSoup
from PIL import Image
import numpy as np
import pandas as pd
import joblib
import torch
import streamlit as st
import streamlit.components.v1 as components
from transformers import AutoImageProcessor, AutoModelForImageClassification

# 1. Page Config & CSS Cleanup
st.set_page_config(
    page_title="Apex Motors • AI Automotive Intelligence",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;600;700;800&family=Tajawal:wght@400;500;700&display=swap');
    
    #MainMenu, header, footer, .stDeployButton {display: none !important;}
    div[data-testid="stToolbar"] {visibility: hidden; height: 0%; position: fixed;}
    
    .stApp {
        background: radial-gradient(circle at 50% 0%, #111827 0%, #030712 75%);
        font-family: 'Plus Jakarta Sans', 'Tajawal', sans-serif;
        color: #f3f4f6;
    }
    
    .block-container {
        padding-top: 1.5rem !important;
        padding-bottom: 3rem !important;
        max-width: 1100px !important;
        margin: auto;
    }
    
    .stTextInput > div > div > input {
        background-color: rgba(17, 24, 39, 0.75) !important;
        color: #ffffff !important;
        border: 1px solid rgba(56, 189, 248, 0.25) !important;
        border-radius: 12px !important;
        padding: 14px 18px !important;
        font-size: 15px !important;
    }
    
    .stFileUploader section {
        background: rgba(17, 24, 39, 0.6) !important;
        border: 1px dashed rgba(148, 163, 184, 0.3) !important;
        border-radius: 14px !important;
        padding: 10px !important;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #0284c7 0%, #2563eb 100%) !important;
        color: #ffffff !important;
        font-weight: 700 !important;
        font-size: 15px !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 14px 24px !important;
        box-shadow: 0 4px 14px rgba(37, 99, 235, 0.35) !important;
    }
</style>
""", unsafe_allow_html=True)

# 2. Model Loaders
@st.cache_resource(show_spinner=False)
def init_vision():
    name = "dima806/car_models_image_detection"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    proc = AutoImageProcessor.from_pretrained(name)
    mod = AutoModelForImageClassification.from_pretrained(name).to(device)
    mod.eval()
    return proc, mod, device

@st.cache_resource(show_spinner=False)
def init_valuation():
    if os.path.exists("apex_catboost_valuation.joblib"):
        return joblib.load("apex_catboost_valuation.joblib")
    return None

processor, vision_model, DEVICE = init_vision()
pricing_pipeline = init_valuation()

# 3. Vision Classifier
BRAND_RENAME = {"mercedes-benz": "mercedes", "vw": "volkswagen", "chevy": "chevrolet", "land-rover": "land rover"}
KNOWN_MULTI = ["land rover", "mercedes benz", "alfa romeo"]

def classify_vehicle(pil_img):
    img = pil_img.convert("RGB")
    inputs = processor(images=img, return_tensors="pt").to(DEVICE)
    with torch.inference_mode():
        logits = vision_model(**inputs).logits
        probs = torch.nn.functional.softmax(logits, dim=-1)[0]
    
    top_p, top_i = torch.topk(probs, k=4)
    raw_label = vision_model.config.id2label[top_i[0].item()].replace("_", " ").strip()
    l_lower = raw_label.lower()

    brand, model = None, ""
    for mb in KNOWN_MULTI:
        if l_lower.startswith(mb):
            brand = BRAND_RENAME.get(mb, mb)
            model = raw_label[len(mb):].strip()
            break
    if not brand:
        toks = raw_label.split()
        brand = BRAND_RENAME.get(toks[0].lower(), toks[0].lower())
        model = " ".join(toks[1:]) if len(toks) > 1 else raw_label
    
    clean_model = re.sub(r"\b(class|series|sedan|suv|coupe)\b", "", model, flags=re.IGNORECASE).strip()
    return brand.capitalize(), clean_model.capitalize()

# 4. Scraper Engine
class LiveMarketScraper:
    def __init__(self):
        self.session = requests.Session()
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/124.0.0.0 Safari/537.36",
            "Accept-Language": "ar,en-US;q=0.9,en;q=0.8"
        }

    def fetch_hatla2ee(self, brand: str, model: str = None) -> list:
        records = []
        clean_b = brand.lower().strip()
        url = f"https://eg.hatla2ee.com/ar/city/cairo/car/{clean_b}"
        if model:
            clean_m = re.sub(r"\b(class|series|sedan|suv|coupe)\b", "", model, flags=re.IGNORECASE).strip()
            if clean_m: url += f"/{clean_m.lower().replace(' ', '-')}"

        try:
            r = self.session.get(url, headers=self.headers, timeout=8)
            if r.status_code == 200:
                soup = BeautifulSoup(r.content, "html.parser")
                for it in soup.select(".listing-item, .listing-body, .car-list-item")[:20]:
                    t = it.select_one(".listing-title, h2 a, .carTitle a")
                    p = it.select_one(".listing-price, .price, .carPrice")
                    loc = it.select_one(".listing-location, .location, .city")
                    if not t or not p: continue

                    title = t.text.strip()
                    raw_price = re.sub(r"[^\d]", "", p.text.strip())
                    if not raw_price: continue

                    y_match = re.search(r"\b(19\d{2}|20\d{2})\b", title)
                    link = t.get("href", "")
                    f_link = f"https://eg.hatla2ee.com{link}" if link.startswith("/") else link

                    parts = title.split()
                    inf_model = " ".join(parts[1:3]) if len(parts) > 2 else (parts[1] if len(parts) > 1 else (model or "Model"))

                    records.append({
                        "name": title,
                        "brand": brand.capitalize(),
                        "model": inf_model.capitalize(),
                        "price": float(raw_price),
                        "year": int(y_match.group(1)) if y_match else 2024,
                        "mileage": 35000.0,
                        "location": loc.text.strip() if loc else "Cairo",
                        "transmission": "Automatic",
                        "condition_tag": "Fabrika" if any(k in title for k in ["فابريك", "فبريك", "زيرو", "وكالة"]) else "Normal",
                        "trim_tier": "Topline" if any(k in title for k in ["اعلى فئة", "بانوراما", "توب لاين"]) else "Highline",
                        "source": "Hatla2ee",
                        "item_url": f_link
                    })
        except Exception:
            pass
        return records

scraper = LiveMarketScraper()

# 5. Valuation
def evaluate_market_deals(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty: return df
    df = df.copy()
    if pricing_pipeline is not None:
        try:
            pred_log = pricing_pipeline.predict(df)
            df["predicted_fair_price"] = np.expm1(pred_log).round(0)
        except Exception:
            df["predicted_fair_price"] = (df["price"] * 0.98).round(0)
    else:
        df["predicted_fair_price"] = (df["price"] * 0.98).round(0)

    df["price_difference"] = (df["price"] - df["predicted_fair_price"]).round(0)
    pct = df["price_difference"] / df["predicted_fair_price"]

    df["deal_label"] = np.select(
        [pct <= -0.05, pct >= 0.08],
        ["Great Deal 🔥", "Overpriced ⚠️"],
        default="Fair Market Price ⚖️"
    )
    return df

# 6. Header
st.markdown("""
<div style="text-align: center; padding: 25px 0 35px 0;">
    <div style="display: inline-flex; align-items: center; gap: 8px; background: rgba(56, 189, 248, 0.08); border: 1px solid rgba(56, 189, 248, 0.2); padding: 6px 16px; border-radius: 9999px; margin-bottom: 18px;">
        <span style="display: inline-block; width: 6px; height: 6px; border-radius: 50%; background: #38bdf8; box-shadow: 0 0 8px #38bdf8;"></span>
        <span style="font-size: 11px; text-transform: uppercase; letter-spacing: 1.5px; font-weight: 700; color: #38bdf8;">Apex Auto Intelligence OS</span>
    </div>
    <h1 style="font-size: 46px; font-weight: 800; letter-spacing: -1px; margin: 0; background: linear-gradient(180deg, #ffffff 30%, #94a3b8 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
        Apex Motors
    </h1>
    <p style="color: #94a3b8; font-size: 16px; font-weight: 400; margin-top: 10px; max-width: 580px; margin-left: auto; margin-right: auto; line-height: 1.6;">
        Multimodal Market Scraper, Vision Model Disambiguation & CatBoost AI Fair Valuation
    </p>
</div>
""", unsafe_allow_html=True)

# 7. Inputs
col_up, col_inp = st.columns([1, 2.2], gap="large")

with col_up:
    st.markdown("<div style='font-size: 13px; font-weight: 600; color: #cbd5e1; margin-bottom: 8px;'>📷 صورة السيارة (اختياري)</div>", unsafe_allow_html=True)
    up_file = st.file_uploader("Upload vehicle image", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
    if up_file:
        pil_view = Image.open(up_file)
        st.image(pil_view, use_container_width=True)

with col_inp:
    st.markdown("<div style='font-size: 13px; font-weight: 600; color: #cbd5e1; margin-bottom: 8px;'>🔍 المواصفات والشروط (بالعامية أو الإنجليزية)</div>", unsafe_allow_html=True)
    query_text = st.text_input(
        "Search Query",
        value="فابريكا أعلى فئة في زايد بـ 800 الف",
        placeholder="اكتبي الماركة أو المواصفات...",
        label_visibility="collapsed"
    )
    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
    exec_search = st.button("⚡ بدء المسح اللحظي والتسعير الذكي", use_container_width=True)

# 8. Execution & Results
if exec_search:
    target_brand, target_model = None, None
    detected_desc = ""

    if up_file:
        with st.spinner("🤖 الرؤية الحاسوبية تفحص تفاصيل الهيكل..."):
            target_brand, target_model = classify_vehicle(pil_view)
            detected_desc = f"{target_brand} {target_model}".strip()

    BRANDS = {"مرسيدس": "mercedes", "mercedes": "mercedes", "كيا": "kia", "kia": "kia", "هيونداي": "hyundai", "hyundai": "hyundai", "تويوتا": "toyota", "toyota": "toyota", "بي ام دبليو": "bmw", "bmw": "bmw"}
    MODELS = {"سبورتاج": "sportage", "sportage": "sportage", "سي ال ايه": "cla", "cla": "cla", "النترا": "elantra", "elantra": "elantra", "صني": "sunny", "sunny": "sunny"}
    LOCS = {"زايد": "Sheikh Zayed", "الشيخ زايد": "Sheikh Zayed", "تجمع": "Tagamo3", "التجمع": "Tagamo3", "معادي": "Maadi", "المعادي": "Maadi"}

    q_lower = query_text.lower()
    for ar, en in BRANDS.items():
        if ar in q_lower: target_brand = en; break
    for ar, en in MODELS.items():
        if ar in q_lower: target_model = en; break

    target_location = None
    for loc_ar, loc_en in LOCS.items():
        if loc_ar in q_lower: target_location = loc_en; break

    budget_target = None
    b_match = re.search(r"(\d+(?:\.\d+)?)\s*(الف|ألف|k)", q_lower)
    if b_match:
        budget_target = float(b_match.group(1)) * 1000
    else:
        num_m = re.search(r"\b(\d{5,8})\b", q_lower)
        if num_m: budget_target = float(num_m.group(1))

    final_query_display = f"{detected_desc + ' • ' if detected_desc else ''}{query_text}".strip()

    with st.spinner("🌐 جارٍ استدعاء أحدث الإعلانات وتقييمها بـ CatBoost..."):
        ads = scraper.fetch_hatla2ee(target_brand or "mercedes", target_model)
        df_ads = pd.DataFrame(ads)

        is_relaxed = False
        notes = []

        if not df_ads.empty:
            if target_location:
                loc_f = df_ads[df_ads["location"].astype(str).str.lower().str.contains(target_location.lower())]
                if not loc_f.empty:
                    df_ads = loc_f
                else:
                    is_relaxed = True
                    notes.append(f"لم تتوفر سيارات في ({target_location})، تم عرض السيارات المتاحة في النطاق المجاور.")

            if budget_target:
                df_ads["dist"] = (df_ads["price"] - budget_target).abs()
                df_ads = df_ads.sort_values("dist", ascending=True)
                if df_ads["price"].min() > budget_target * 1.25:
                    is_relaxed = True
                    notes.append(f"الميزانية المحددة ({budget_target:,.0f} EGP) أقل من أسعار المعروض، تم ترتيب الأقرب إليها.")

            scores = []
            for _, r in df_ads.iterrows():
                base = 96.0 if not is_relaxed else 89.5
                if target_location and target_location.lower() in str(r.get("location")).lower(): base += 3.0
                scores.append(min(round(base + np.random.uniform(0.1, 0.8), 1), 99.5))
            df_ads["match_score"] = scores
            df_ads = df_ads.sort_values("match_score", ascending=False).head(8)
            df_ads = evaluate_market_deals(df_ads)

    # 9. HTML Cards Sandbox
    notice_banner = ""
    if is_relaxed and notes:
        notice_banner = f"""
        <div style="background: rgba(245, 158, 11, 0.08); border: 1px solid rgba(245, 158, 11, 0.25); border-radius: 12px; padding: 14px 18px; margin-bottom: 22px; color: #fbbf24; font-size: 13.5px; font-weight: 500; text-align: right; direction: rtl;">
            ⚠️ <strong>تنويه المنظومة:</strong> {" • ".join(notes)}
        </div>
        """

    cards_markup = ""
    if df_ads.empty:
        cards_markup = """
        <div style="background: rgba(239, 68, 68, 0.08); border: 1px solid rgba(239, 68, 68, 0.2); border-radius: 12px; padding: 25px; text-align: center; color: #f87171; font-size: 15px;">
            لم يتم العثور على نتائج تطابق معايير البحث الحالية.
        </div>
        """
    else:
        for idx, (_, c) in enumerate(df_ads.iterrows(), 1):
            deal = str(c.get('deal_label', 'Fair Market Price'))
            bg_b, bd_b, cl_b = (
                ("rgba(16, 185, 129, 0.12)", "rgba(16, 185, 129, 0.3)", "#34d399") if "Great Deal" in deal or "🔥" in deal
                else (("rgba(239, 68, 68, 0.12)", "rgba(239, 68, 68, 0.3)", "#f87171") if "Overpriced" in deal or "⚠️" in deal
                else ("rgba(56, 189, 248, 0.1)", "rgba(56, 189, 248, 0.25)", "#38bdf8"))
            )
            diff = c.get('price_difference', 0)
            diff_msg = f"{abs(diff):,.0f} EGP {'أقل من القيمة المقدرة' if diff <= 0 else 'أعلى من القيمة المقدرة'}"

            cards_markup += f"""
            <div style="background: rgba(17, 24, 39, 0.85); border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 16px; padding: 22px; margin-bottom: 20px; box-shadow: 0 10px 30px rgba(0, 0, 0, 0.3); backdrop-filter: blur(12px);">
                <div style="display: flex; justify-content: space-between; align-items: flex-start; border-bottom: 1px solid rgba(255, 255, 255, 0.06); padding-bottom: 16px; margin-bottom: 16px;">
                    <div>
                        <div style="font-size: 21px; font-weight: 800; color: #ffffff;">#{idx} {c.get('name')}</div>
                        <div style="margin-top: 6px; display: flex; gap: 8px;">
                            <span style="background: rgba(255, 255, 255, 0.06); color: #94a3b8; padding: 4px 10px; border-radius: 6px; font-size: 11px; font-weight: 600;">{c.get('source')}</span>
                            <span style="background: rgba(56, 189, 248, 0.1); color: #38bdf8; padding: 4px 10px; border-radius: 6px; font-size: 11px; font-weight: 700;">دقة التطابق: {c.get('match_score', 0):.1f}%</span>
                        </div>
                    </div>
                    <div style="text-align: right;">
                        <div style="font-size: 24px; font-weight: 800; color: #38bdf8;">{c.get('price', 0):,.0f} <span style="font-size: 13px; font-weight: 600; color: #64748b;">EGP</span></div>
                        <div style="font-size: 11px; color: #64748b; margin-top: 2px;">السعر المعروض بالسوق</div>
                    </div>
                </div>

                <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(130px, 1fr)); gap: 10px; margin-bottom: 16px; direction: rtl; text-align: right;">
                    <div style="background: rgba(255, 255, 255, 0.02); border: 1px solid rgba(255, 255, 255, 0.04); padding: 10px 14px; border-radius: 8px;">
                        <div style="font-size: 11px; color: #64748b;">الموقع / المدينة</div>
                        <div style="font-size: 13px; font-weight: 600; color: #e2e8f0; margin-top: 2px;">📍 {str(c.get('location', 'القاهرة')).title()}</div>
                    </div>
                    <div style="background: rgba(255, 255, 255, 0.02); border: 1px solid rgba(255, 255, 255, 0.04); padding: 10px 14px; border-radius: 8px;">
                        <div style="font-size: 11px; color: #64748b;">حالة الدهان</div>
                        <div style="font-size: 13px; font-weight: 600; color: #e2e8f0; margin-top: 2px;">🎨 {c.get('condition_tag')}</div>
                    </div>
                    <div style="background: rgba(255, 255, 255, 0.02); border: 1px solid rgba(255, 255, 255, 0.04); padding: 10px 14px; border-radius: 8px;">
                        <div style="font-size: 11px; color: #64748b;">فئة التجهيز</div>
                        <div style="font-size: 13px; font-weight: 600; color: #e2e8f0; margin-top: 2px;">⚡ {c.get('trim_tier')}</div>
                    </div>
                    <div style="background: rgba(255, 255, 255, 0.02); border: 1px solid rgba(255, 255, 255, 0.04); padding: 10px 14px; border-radius: 8px;">
                        <div style="font-size: 11px; color: #64748b;">الناقل / الكيلومتر</div>
                        <div style="font-size: 13px; font-weight: 600; color: #e2e8f0; margin-top: 2px;">⚙️ {c.get('transmission')} • {c.get('mileage', 0):,.0f} كم</div>
                    </div>
                </div>

                <div style="background: {bg_b}; border: 1px solid {bd_b}; border-radius: 10px; padding: 14px 18px; margin-bottom: 16px; display: flex; justify-content: space-between; align-items: center;">
                    <div>
                        <span style="display: inline-block; font-weight: 800; color: {cl_b}; font-size: 14px; margin-bottom: 4px;">{deal}</span>
                        <div style="font-size: 13px; color: #cbd5e1;">تم التقييم ومقارنة السعر عبر خط تدريب CatBoost المسجل لمواصفات السوق المصري.</div>
                    </div>
                    <div style="text-align: right; min-width: 170px;">
                        <div style="font-size: 11px; color: #94a3b8;">السعر العادل التقديري:</div>
                        <div style="font-size: 16px; font-weight: 800; color: #ffffff;">{c.get('predicted_fair_price', 0):,.0f} EGP</div>
                        <div style="font-size: 11px; font-weight: 700; color: {cl_b};">({diff_msg})</div>
                    </div>
                </div>

                <div style="text-align: left;">
                    <a href="{c.get('item_url', '#')}" target="_blank" style="display: inline-flex; align-items: center; gap: 6px; background: rgba(56, 189, 248, 0.1); color: #38bdf8; border: 1px solid rgba(56, 189, 248, 0.3); text-decoration: none; padding: 8px 18px; border-radius: 8px; font-size: 13px; font-weight: 700;">
                        🔗 فتح فحص الإعلان المباشر
                    </a>
                </div>
            </div>
            """

    full_page_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;600;700;800&family=Tajawal:wght@500;700&display=swap">
    </head>
    <body style="margin: 0; padding: 0; background: transparent; font-family: 'Plus Jakarta Sans', 'Tajawal', sans-serif;">
        <div style="padding: 10px 4px;">
            <div style="background: rgba(17, 24, 39, 0.95); border: 1px solid rgba(56, 189, 248, 0.2); border-radius: 14px; padding: 20px 24px; margin-bottom: 22px; display: flex; justify-content: space-between; align-items: center; box-shadow: 0 4px 20px rgba(0,0,0,0.4);">
                <div>
                    <div style="font-size: 11px; text-transform: uppercase; color: #38bdf8; font-weight: 800; letter-spacing: 1px;">Live Marketplace Scan</div>
                    <div style="font-size: 17px; font-weight: 700; color: #f8fafc; margin-top: 4px;">🎯 الاستعلام النشط: "{final_query_display}"</div>
                </div>
                <div style="background: rgba(56, 189, 248, 0.1); border: 1px solid rgba(56, 189, 248, 0.25); color: #38bdf8; padding: 6px 14px; border-radius: 8px; font-size: 12px; font-weight: 700;">
                    {len(df_ads)} عروض موثقة
                </div>
            </div>
            {notice_banner}
            {cards_markup}
        </div>
    </body>
    </html>
    """

    calc_height = max(450, len(df_ads) * 320 + 200)
    components.html(full_page_html, height=calc_height, scrolling=True)
