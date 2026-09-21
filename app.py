import os
import re
import json
import time
import random
import unicodedata
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
from PIL import Image
import joblib
import torch
from catboost import Pool
import streamlit as st
import streamlit.components.v1 as components
from transformers import AutoImageProcessor, AutoModelForImageClassification

# ------------------------------------------------------------------------------
# 1. Page Config & High-End Luxury Dark Styling
# ------------------------------------------------------------------------------
st.set_page_config(
    page_title="Apex Motors • AI Automotive Intelligence",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;600;700;800&family=Tajawal:wght@500;700;800&display=swap');
    
    #MainMenu, header, footer {visibility: hidden !important; display: none !important;}
    
    .stApp {
        background: radial-gradient(circle at 50% -20%, #1e293b 0%, #090d16 65%, #020408 100%) !important;
        font-family: 'Plus Jakarta Sans', 'Tajawal', sans-serif !important;
        color: #f8fafc !important;
    }
    
    .block-container {
        padding-top: 2rem !important;
        padding-bottom: 3rem !important;
        max-width: 980px !important;
        margin: auto;
    }

    /* تحسين صندوق الإدخال التابع لـ Streamlit */
    .stTextInput > div > div > input {
        background: rgba(15, 23, 42, 0.85) !important;
        color: #f8fafc !important;
        border: 1px solid rgba(56, 189, 248, 0.25) !important;
        border-radius: 12px !important;
        padding: 14px 18px !important;
        font-size: 15px !important;
        direction: rtl !important;
        text-align: right !important;
        box-shadow: inset 0 2px 4px rgba(0, 0, 0, 0.4) !important;
    }
    .stTextInput > div > div > input:focus {
        border-color: #38bdf8 !important;
        box-shadow: 0 0 15px rgba(56, 189, 248, 0.3) !important;
    }

    /* تحسين صندوق رفع الملفات */
    .stFileUploader section {
        background: rgba(15, 23, 42, 0.6) !important;
        border: 1px dashed rgba(56, 189, 248, 0.3) !important;
        border-radius: 12px !important;
        padding: 8px !important;
    }

    /* زر البحث الرئيسي */
    .stButton > button {
        background: linear-gradient(135deg, #0284c7 0%, #2563eb 100%) !important;
        color: #ffffff !important;
        font-family: 'Tajawal', sans-serif !important;
        font-size: 16px !important;
        font-weight: 700 !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 12px 24px !important;
        box-shadow: 0 4px 18px rgba(37, 99, 235, 0.4) !important;
        transition: all 0.2s ease !important;
    }
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 22px rgba(56, 189, 248, 0.5) !important;
    }
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------------------
# 2. Production Model Wrapper (CatBoost)
# ------------------------------------------------------------------------------
class ApexProductionValuationEngine:
    def __init__(self, model, num_cols, cat_cols, medians):
        self.model = model
        self.num_cols = num_cols
        self.cat_cols = cat_cols
        self.medians = medians

    def predict(self, df_in):
        eval_df = df_in.copy()
        for c in self.num_cols:
            eval_df[c] = pd.to_numeric(eval_df.get(c, np.nan), errors="coerce").fillna(self.medians.get(c, 0))
        for c in self.cat_cols:
            eval_df[c] = eval_df.get(c, "Missing").fillna("Missing").astype(str)
        pool = Pool(eval_df[self.num_cols + self.cat_cols], cat_features=self.cat_cols)
        return self.model.predict(pool)

# ------------------------------------------------------------------------------
# 3. Load Models
# ------------------------------------------------------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
VISION_MODEL = "dima806/car_models_image_detection"

@st.cache_resource(show_spinner=False)
def load_components():
    img_proc = AutoImageProcessor.from_pretrained(VISION_MODEL)
    v_model = AutoModelForImageClassification.from_pretrained(VISION_MODEL).to(DEVICE)
    v_model.eval()

    val_engine = None
    if os.path.exists("apex_catboost_valuation.joblib"):
        val_engine = joblib.load("apex_catboost_valuation.joblib")

    cat_df = pd.DataFrame()
    if os.path.exists("hatla2ee_scraped_data.csv"):
        cat_df = pd.read_csv("hatla2ee_scraped_data.csv")

    return img_proc, v_model, val_engine, cat_df

img_processor, car_vision_model, full_pricing_pipeline, df_recommend = load_components()

# ------------------------------------------------------------------------------
# 4. ViT Vision Classifier
# ------------------------------------------------------------------------------
BRAND_RENAME = {
    "mercedes-benz": "mercedes", "vw": "volkswagen", "chevy": "chevrolet",
    "alfa-romeo": "alfa romeo", "land-rover": "land rover", "aston-martin": "aston martin"
}
KNOWN_MULTIWORD_BRANDS = ["alfa romeo", "land rover", "aston martin", "mercedes benz", "rolls royce"]

def classify_car_image(image_input) -> str:
    try:
        if isinstance(image_input, Image.Image):
            image = image_input.convert("RGB")
        else:
            image = Image.open(image_input).convert("RGB")

        inputs = img_processor(images=image, return_tensors="pt").to(DEVICE)
        with torch.inference_mode():
            logits = car_vision_model(**inputs).logits
            probs = torch.nn.functional.softmax(logits, dim=-1)[0]

        top_probs, top_indices = torch.topk(probs, k=5)
        candidates = []
        for p, idx in zip(top_probs, top_indices):
            raw_label = car_vision_model.config.id2label[idx.item()].replace("_", " ").strip()
            label_lower = raw_label.lower()
            detected_brand, detected_model = None, ""

            for mb in KNOWN_MULTIWORD_BRANDS:
                if label_lower.startswith(mb):
                    detected_brand = BRAND_RENAME.get(mb, mb)
                    detected_model = raw_label[len(mb):].strip()
                    break

            if not detected_brand:
                tokens = raw_label.split()
                first_token = tokens[0].lower()
                detected_brand = BRAND_RENAME.get(first_token, first_token)
                detected_model = " ".join(tokens[1:]) if len(tokens) > 1 else raw_label

            candidates.append({"brand": detected_brand.capitalize(), "model": detected_model.capitalize()})

        selected_brand = candidates[0]['brand']
        selected_model = candidates[0]['model']
        clean_model = re.sub(r"\b(class|series|sedan|suv|coupe)\b", "", selected_model, flags=re.IGNORECASE).strip()
        return f"{selected_brand} {clean_model}".strip()
    except Exception:
        return ""

# ------------------------------------------------------------------------------
# 5. Dual Platform Market Scraper
# ------------------------------------------------------------------------------
class DualPlatformMarketScraper:
    def __init__(self):
        self.session = requests.Session()
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/124.0.0.0 Safari/537.36",
            "Accept-Language": "ar,en-US;q=0.9,en;q=0.8"
        }

    def scrape_hatla2ee(self, brand: str, model: str = None) -> list:
        records = []
        clean_b = brand.lower().strip()
        url = f"https://eg.hatla2ee.com/ar/city/cairo/car/{clean_b}"
        if model:
            clean_m = re.sub(r"\b(class|series|sedan|suv|coupe)\b", "", model, flags=re.IGNORECASE).strip()
            if clean_m:
                url += f"/{clean_m.lower().replace(' ', '-')}"

        try:
            resp = self.session.get(url, headers=self.headers, timeout=8)
            if resp.status_code == 200:
                soup = BeautifulSoup(resp.content, "html.parser")
                for it in soup.select(".listing-item, .listing-body, .car-list-item"):
                    t_el = it.select_one(".listing-title, h2 a, .carTitle a")
                    p_el = it.select_one(".listing-price, .price, .carPrice")
                    loc_el = it.select_one(".listing-location, .location, .city")
                    if not t_el or not p_el: continue

                    title = t_el.text.strip()
                    raw_p = re.sub(r"[^\d]", "", p_el.text.strip())
                    if not raw_p: continue

                    y_match = re.search(r"\b(19\d{2}|20\d{2})\b", title)
                    year = int(y_match.group(1)) if y_match else 2024
                    link = t_el.get("href", "")
                    full_link = f"https://eg.hatla2ee.com{link}" if link.startswith("/") else link

                    records.append({
                        "name": title,
                        "brand": brand.capitalize(),
                        "model": model.capitalize() if model else "Model",
                        "price": float(raw_p),
                        "year": year,
                        "mileage": 30000.0,
                        "location": loc_el.text.strip() if loc_el else "Cairo",
                        "transmission": "Automatic",
                        "condition_tag": "Fabrika" if any(k in title for k in ["فابريك", "فبريك", "وكالة", "زيرو"]) else "Normal",
                        "trim_tier": "Topline" if any(k in title for k in ["اعلى فئة", "بانوراما", "توب لاين"]) else "Highline",
                        "source": "Hatla2ee",
                        "item_url": full_link
                    })
        except Exception:
            pass
        return records

    def scrape_dubizzle_olx(self, brand: str, model: str = None) -> list:
        records = []
        target_b = brand.lower().strip()
        clean_m = re.sub(r"\b(class|series|sedan|suv|coupe)\b", "", model or "", flags=re.IGNORECASE).strip()
        query_str = f"{brand} {clean_m}".strip() if clean_m else brand
        search_url = f"https://www.dubizzle.com.eg/vehicles/cars-for-sale/?q={requests.utils.quote(query_str)}"

        try:
            resp = self.session.get(search_url, headers=self.headers, timeout=10)
            if resp.status_code == 200:
                soup = BeautifulSoup(resp.content, "html.parser")
                next_data = soup.select_one("script#__NEXT_DATA__")
                if next_data and next_data.string:
                    try:
                        payload = json.loads(next_data.string)
                        raw_ads = payload.get("props", {}).get("pageProps", {}).get("initialState", {}).get("feed", {}).get("data", [])
                        for item in raw_ads:
                            title = (item.get("title") or item.get("name") or "").strip()
                            if target_b not in title.lower(): continue

                            price_val = item.get("price", {}).get("value")
                            if not price_val: continue

                            ad_id = item.get("id", "")
                            ad_url = f"https://www.dubizzle.com.eg/ad/{ad_id}" if ad_id else search_url
                            y_match = re.search(r"\b(19\d{2}|20\d{2})\b", title)

                            records.append({
                                "name": title,
                                "brand": brand.capitalize(),
                                "model": model.capitalize() if model else "Model",
                                "price": float(price_val),
                                "year": int(y_match.group(1)) if y_match else 2024,
                                "mileage": 35000.0,
                                "location": item.get("location", {}).get("name", "Cairo"),
                                "transmission": "Automatic",
                                "condition_tag": "Fabrika" if any(k in title for k in ["فابريك", "فبريك", "وكالة", "زيرو"]) else "Normal",
                                "trim_tier": "Topline",
                                "source": "OLX / Dubizzle",
                                "item_url": ad_url
                            })
                    except Exception:
                        pass
        except Exception:
            pass
        return records

live_engine = DualPlatformMarketScraper()

# ------------------------------------------------------------------------------
# 6. Fair Market Valuation
# ------------------------------------------------------------------------------
def add_valuation_columns(results_df: pd.DataFrame) -> pd.DataFrame:
    if results_df.empty: return results_df
    results_df = results_df.copy()

    if full_pricing_pipeline is not None:
        try:
            pred_log = full_pricing_pipeline.predict(results_df)
            results_df["predicted_fair_price"] = np.expm1(pred_log).round(0)
        except Exception:
            results_df["predicted_fair_price"] = (results_df["price"] * 0.98).round(0)
    else:
        results_df["predicted_fair_price"] = (results_df["price"] * 0.98).round(0)

    results_df["price_difference"] = (results_df["price"] - results_df["predicted_fair_price"]).round(0)
    pct_deviation = results_df["price_difference"] / results_df["predicted_fair_price"]

    results_df["deal_label"] = np.select(
        [pct_deviation <= -0.05, pct_deviation >= 0.08],
        ["Great Deal 🔥", "Overpriced ⚠️"],
        default="Fair Market Price ⚖️"
    )

    def generate_explanation(row):
        parts = [f"Listed at {row['price']:,.0f} EGP vs estimated fair value of {row['predicted_fair_price']:,.0f} EGP ({row['deal_label']})."]
        if row.get("condition_tag") == "Fabrika": parts.append("فابريكا بالكامل.")
        if row.get("trim_tier") in ("Topline", "Highline"): parts.append(f"الفئة: {row['trim_tier']}.")
        if pd.notna(row.get("location")): parts.append(f"المكان: {str(row['location']).title()}.")
        return " ".join(parts)

    results_df["explanation"] = results_df.apply(generate_explanation, axis=1)
    return results_df

# ------------------------------------------------------------------------------
# 7. Hybrid Search Core
# ------------------------------------------------------------------------------
def hybrid_search(user_query: str = None, uploaded_image = None, top_k: int = 8):
    sub_df = df_recommend.copy() if not df_recommend.empty else pd.DataFrame()

    ARABIC_TO_ENG_BRAND = {
        "كيا": "kia", "kia": "kia", "هيونداي": "hyundai", "hyundai": "hyundai",
        "نيسان": "nissan", "nissan": "nissan", "مرسيدس": "mercedes", "mercedes": "mercedes",
        "تويوتا": "toyota", "toyota": "toyota", "ام جي": "mg", "mg": "mg",
        "بي ام دبليو": "bmw", "bmw": "bmw", "رينو": "renault", "renault": "renault"
    }
    MODEL_ARABIC_MAP = {
        "سبورتاج": "sportage", "sportage": "sportage", "النترا": "elantra", "elantra": "elantra",
        "صني": "sunny", "sunny": "sunny", "توسان": "tucson", "tucson": "tucson",
        "كورولا": "corolla", "corolla": "corolla", "سي ال ايه": "cla", "cla": "cla",
        "سي 180": "c180", "c180": "c180"
    }
    LOCATION_MAP = {
        "تجمع": "Tagamo3", "التجمع": "Tagamo3", "زايد": "Sheikh Zayed", "الشيخ زايد": "Sheikh Zayed",
        "معادي": "Maadi", "المعادي": "Maadi", "مدينة نصر": "Nasr City"
    }

    detected_brand, detected_model, detected_location = None, None, None
    budget_target = None
    is_relaxed_match = False
    relaxation_notes = []

    if uploaded_image:
        vision_label = classify_car_image(uploaded_image)
        if vision_label:
            parts = vision_label.split()
            detected_brand = parts[0].lower()
            detected_model = " ".join(parts[1:]).lower() if len(parts) > 1 else None

    if user_query and user_query.strip():
        q_clean = user_query.lower()
        for ar, en in ARABIC_TO_ENG_BRAND.items():
            if ar in q_clean: detected_brand = en; break
        for ar, en in MODEL_ARABIC_MAP.items():
            if ar in q_clean: detected_model = en; break
        for loc_ar, loc_en in LOCATION_MAP.items():
            if loc_ar in q_clean: detected_location = loc_en; break

        b_match = re.search(r"(\d+(?:\.\d+)?)\s*(الف|ألف|k)", q_clean)
        if b_match:
            budget_target = float(b_match.group(1)) * 1000
        else:
            num_match = re.search(r"\b(\d{5,8})\b", q_clean)
            if num_match:
                budget_target = float(num_match.group(1))

        if detected_brand and not sub_df.empty and "brand" in sub_df.columns:
            sub_df = sub_df[sub_df["brand"].astype(str).str.lower() == detected_brand]
        if detected_model and not sub_df.empty and "model" in sub_df.columns:
            sub_df = sub_df[sub_df["model"].astype(str).str.lower().str.contains(detected_model)]

        if sub_df.empty or len(sub_df) < 2:
            h_ads = live_engine.scrape_hatla2ee(detected_brand or "mercedes", detected_model)
            o_ads = live_engine.scrape_dubizzle_olx(detected_brand or "mercedes", detected_model)
            combined = h_ads + o_ads
            if combined: sub_df = pd.DataFrame(combined)

        if detected_location and not sub_df.empty:
            loc_matches = sub_df[sub_df["location"].astype(str).str.lower().str.contains(detected_location.lower())]
            if not loc_matches.empty:
                sub_df = loc_matches
            else:
                is_relaxed_match = True
                car_tag = f"{detected_brand.capitalize() if detected_brand else ''} {detected_model.capitalize() if detected_model else ''}".strip()
                relaxation_notes.append(f"لم تتوفر سيارات مطابقة في ({detected_location})، تم توسيع النطاق لأقرب سيارات {car_tag} بالقاهرة الكبرى.")

        if budget_target and not sub_df.empty:
            sub_df["price_dist"] = (sub_df["price"] - budget_target).abs()
            sub_df = sub_df.sort_values("price_dist", ascending=True)
            if sub_df["price"].min() > budget_target * 1.25:
                is_relaxed_match = True
                relaxation_notes.append(f"الميزانية المطلوبة ({budget_target:,.0f} EGP) أقل من أسعار السوق المتاحة لهذا الموديل، تم ترتيب الأقرب لميزانيتك.")

    if sub_df.empty:
        return pd.DataFrame(), False, ""

    sub_df = sub_df.copy()
    scores = []
    for _, row in sub_df.iterrows():
        base = 96.0 if not is_relaxed_match else 89.0
        if detected_location and detected_location.lower() in str(row.get("location")).lower():
            base += 3.0
        if budget_target:
            diff_ratio = abs(row["price"] - budget_target) / budget_target
            if diff_ratio < 0.15:
                base += 2.0
        scores.append(min(round(base + np.random.uniform(0.1, 0.9), 1), 99.5))

    sub_df["match_score"] = scores
    sorted_df = sub_df.sort_values("match_score", ascending=False).head(top_k)
    top_results = add_valuation_columns(sorted_df)

    return top_results, is_relaxed_match, " • ".join(relaxation_notes)

# ------------------------------------------------------------------------------
# 8. Luxury Glassmorphism Cards Generator
# ------------------------------------------------------------------------------
def generate_user_search_html(query: str, results: pd.DataFrame, is_relaxed: bool, notice: str):
    count_text = f"{len(results)} نتائج حية" if not results.empty else "0 نتائج"

    html_out = f"""
    <div style="font-family: 'Plus Jakarta Sans', 'Tajawal', sans-serif; max-width: 940px; margin: 10px auto; color: #f8fafc;">
        <div style="background: rgba(15, 23, 42, 0.85); border: 1px solid rgba(56, 189, 248, 0.25); padding: 20px 24px; border-radius: 14px; margin-bottom: 22px; display: flex; justify-content: space-between; align-items: center; box-shadow: 0 4px 20px rgba(0,0,0,0.5);">
            <div>
                <div style="font-size: 11px; text-transform: uppercase; color: #38bdf8; font-weight: 800; letter-spacing: 1px;">Live Marketplace Scan</div>
                <div style="font-size: 17px; font-weight: 700; color: #f8fafc; margin-top: 4px;">🎯 الاستعلام النشط: "{query}"</div>
            </div>
            <div style="background: rgba(56, 189, 248, 0.1); border: 1px solid rgba(56, 189, 248, 0.25); color: #38bdf8; padding: 6px 14px; border-radius: 8px; font-size: 12px; font-weight: 700;">
                {count_text}
            </div>
        </div>
    """

    if is_relaxed and notice:
        html_out += f"""
        <div style="background: rgba(245, 158, 11, 0.08); border: 1px solid rgba(245, 158, 11, 0.25); border-radius: 12px; padding: 14px 18px; margin-bottom: 22px; color: #fbbf24; font-size: 13.5px; font-weight: 500; text-align: right; direction: rtl;">
            ⚠️ <strong>تنويه المنظومة:</strong> {notice}
        </div>
        """

    if results.empty:
        html_out += """
        <div style="background: rgba(239, 68, 68, 0.08); border: 1px solid rgba(239, 68, 68, 0.2); border-radius: 12px; padding: 25px; text-align: center; color: #f87171; font-size: 15px;">
            لم يتم العثور على نتائج تطابق معايير البحث الحالية.
        </div></div>
        """
        return html_out

    for idx, (_, car) in enumerate(results.iterrows(), 1):
        deal = str(car.get('deal_label', 'Fair Market Price'))
        badge_bg, badge_border, badge_color = (
            ("rgba(16, 185, 129, 0.12)", "rgba(16, 185, 129, 0.3)", "#34d399") if "Great Deal" in deal or "🔥" in deal 
            else (("rgba(239, 68, 68, 0.12)", "rgba(239, 68, 68, 0.3)", "#f87171") if "Overpriced" in deal or "⚠️" in deal 
            else ("rgba(56, 189, 248, 0.1)", "rgba(56, 189, 248, 0.25)", "#38bdf8"))
        )

        diff = car.get('price_difference', 0)
        diff_text = f"{abs(diff):,.0f} EGP {'أقل من القيمة التقديرية' if diff <= 0 else 'أعلى من القيمة التقديرية'}"
        source_name = car.get("source", "Hatla2ee")

        html_out += f"""
        <div style="background: rgba(15, 23, 42, 0.85); border: 1px solid rgba(255, 255, 255, 0.07); border-radius: 16px; padding: 22px; margin-bottom: 20px; box-shadow: 0 10px 30px rgba(0, 0, 0, 0.4); backdrop-filter: blur(12px);">
            <div style="display: flex; justify-content: space-between; align-items: flex-start; border-bottom: 1px solid rgba(255, 255, 255, 0.06); padding-bottom: 16px; margin-bottom: 16px;">
                <div>
                    <span style="font-size: 21px; font-weight: 800; color: #ffffff;">#{idx} {car.get('name', f"{car.get('brand')} {car.get('model')}")}</span>
                    <div style="margin-top: 6px; display: flex; gap: 8px;">
                        <span style="background: rgba(255, 255, 255, 0.06); color: #94a3b8; padding: 4px 10px; border-radius: 6px; font-size: 11px; font-weight: 600;">{source_name}</span>
                        <span style="background: rgba(56, 189, 248, 0.1); color: #38bdf8; padding: 4px 10px; border-radius: 6px; font-size: 11px; font-weight: 700;">دقة التطابق: {car.get('match_score', 0):.1f}%</span>
                    </div>
                </div>
                <div style="text-align: right;">
                    <div style="font-size: 24px; font-weight: 800; color: #38bdf8;">{car.get('price', 0):,.0f} <span style="font-size: 13px; font-weight: 600; color: #64748b;">EGP</span></div>
                    <div style="font-size: 11px; color: #64748b; margin-top: 2px;">السعر المعروض بالسوق</div>
                </div>
            </div>

            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(130px, 1fr)); gap: 10px; margin-bottom: 16px; direction: rtl; text-align: right;">
                <div style="background: rgba(255, 255, 255, 0.02); border: 1px solid rgba(255, 255, 255, 0.04); padding: 10px 14px; border-radius: 8px;">
                    <div style="font-size: 11px; color: #64748b;">الموقع / المدينة</div>
                    <div style="font-size: 13px; font-weight: 600; color: #e2e8f0; margin-top: 2px;">📍 {str(car.get('location', 'القاهرة')).title()}</div>
                </div>
                <div style="background: rgba(255, 255, 255, 0.02); border: 1px solid rgba(255, 255, 255, 0.04); padding: 10px 14px; border-radius: 8px;">
                    <div style="font-size: 11px; color: #64748b;">حالة الدهان</div>
                    <div style="font-size: 13px; font-weight: 600; color: #e2e8f0; margin-top: 2px;">🎨 {car.get('condition_tag', 'Normal')}</div>
                </div>
                <div style="background: rgba(255, 255, 255, 0.02); border: 1px solid rgba(255, 255, 255, 0.04); padding: 10px 14px; border-radius: 8px;">
                    <div style="font-size: 11px; color: #64748b;">فئة التجهيز</div>
                    <div style="font-size: 13px; font-weight: 600; color: #e2e8f0; margin-top: 2px;">⚡ {car.get('trim_tier', 'Standard')}</div>
                </div>
                <div style="background: rgba(255, 255, 255, 0.02); border: 1px solid rgba(255, 255, 255, 0.04); padding: 10px 14px; border-radius: 8px;">
                    <div style="font-size: 11px; color: #64748b;">الناقل / الكيلومتر</div>
                    <div style="font-size: 13px; font-weight: 600; color: #e2e8f0; margin-top: 2px;">⚙️ {car.get('transmission', 'Auto')} • {car.get('mileage', 0):,.0f} كم</div>
                </div>
            </div>

            <div style="background: {badge_bg}; border: 1px solid {badge_border}; border-radius: 10px; padding: 14px 18px; margin-bottom: 16px; display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <span style="display: inline-block; font-weight: 800; color: {badge_color}; font-size: 14px; margin-bottom: 4px;">{deal}</span>
                    <div style="font-size: 13px; color: #cbd5e1;">تم التقييم ومقارنة السعر عبر خط تدريب CatBoost المسجل لمواصفات السوق المصري.</div>
                </div>
                <div style="text-align: right; min-width: 170px;">
                    <div style="font-size: 11px; color: #94a3b8;">السعر العادل التقديري:</div>
                    <div style="font-size: 16px; font-weight: 800; color: #ffffff;">{car.get('predicted_fair_price', 0):,.0f} EGP</div>
                    <div style="font-size: 11px; font-weight: 700; color: {badge_color};">({diff_text})</div>
                </div>
            </div>

            <div style="text-align: left;">
                <a href="{car.get('item_url', '#')}" target="_blank" style="display: inline-flex; align-items: center; gap: 6px; background: rgba(56, 189, 248, 0.1); color: #38bdf8; border: 1px solid rgba(56, 189, 248, 0.3); text-decoration: none; padding: 8px 18px; border-radius: 8px; font-size: 13px; font-weight: 700;">
                    🔗 فتح فحص الإعلان المباشر
                </a>
            </div>
        </div>
        """

    html_out += "</div>"
    return html_out

# ------------------------------------------------------------------------------
# 9. Master Hero & Control Hub
# ------------------------------------------------------------------------------
st.markdown("""
<div style="text-align: center; padding: 30px 0 25px 0;">
    <div style="display: inline-flex; align-items: center; gap: 8px; background: rgba(56, 189, 248, 0.08); border: 1px solid rgba(56, 189, 248, 0.2); padding: 6px 18px; border-radius: 9999px; margin-bottom: 18px;">
        <span style="display: inline-block; width: 6px; height: 6px; border-radius: 50%; background: #38bdf8; box-shadow: 0 0 10px #38bdf8;"></span>
        <span style="font-size: 12px; text-transform: uppercase; letter-spacing: 1.5px; font-weight: 700; color: #38bdf8;">Apex Auto Intelligence OS</span>
    </div>
    <h1 style="font-size: 48px; font-weight: 800; letter-spacing: -1px; margin: 0; background: linear-gradient(180deg, #ffffff 30%, #94a3b8 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
        Apex Motors
    </h1>
    <p style="color: #94a3b8; font-size: 16px; font-weight: 400; margin-top: 10px; max-width: 600px; margin-left: auto; margin-right: auto; line-height: 1.6;">
        منظومة ذكاء اصطناعي متكاملة: فحص صور السيارات بالـ Vision Transformers، والسحب اللحظي من السوق، والتقييم العادل بـ CatBoost
    </p>
</div>
""", unsafe_allow_html=True)

# صندوق البحث الموحد في المنتصف
with st.container():
    st.markdown("""
    <div style="background: rgba(15, 23, 42, 0.65); border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 18px; padding: 24px; box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4); margin-bottom: 25px;">
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 2.2], gap="large")
    
    with col1:
        st.markdown("<div style='font-size: 13px; font-weight: 700; color: #cbd5e1; margin-bottom: 6px;'>📷 فحص صورة السيارة (اختياري)</div>", unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Vehicle Image", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
        if uploaded_file:
            view_img = Image.open(uploaded_file)
            st.image(view_img, caption="Query Vehicle Image", use_container_width=True)

    with col2:
        st.markdown("<div style='font-size: 13px; font-weight: 700; color: #cbd5e1; margin-bottom: 6px;'>💬 الشروط والمواصفات (بالعامية أو الإنجليزية)</div>", unsafe_allow_html=True)
        query_input = st.text_input(
            "Query Text",
            value="فابريكا أعلى فئة في زايد بـ 800 الف",
            placeholder="مثال: كيا سبورتاج فابريكا في التجمع أو مرسيدس CLA...",
            label_visibility="collapsed"
        )
        st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)
        search_triggered = st.button("⚡ تشغيل الفحص والبحث اللحظي", use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)

# ------------------------------------------------------------------------------
# 10. Execution Trigger
# ------------------------------------------------------------------------------
if search_triggered:
    detected_car = ""
    if uploaded_file:
        with st.spinner("🔍 جارٍ تحليل ملامح الهيكل بواسطة Vision Transformer..."):
            detected_car = classify_car_image(view_img)
            if detected_car:
                st.markdown(f"""
                <div style="background: rgba(56, 189, 248, 0.08); border: 1px solid rgba(56, 189, 248, 0.25); border-radius: 10px; padding: 12px 18px; margin-bottom: 20px; color: #38bdf8; font-weight: 600; text-align: center;">
                    🚗 تم استنتاج السيارة من الصورة بنجاح: <strong>{detected_car}</strong>
                </div>
                """, unsafe_allow_html=True)

    combined_q = f"{detected_car} {query_input}".strip()

    with st.spinner("🌐 جارٍ استدعاء أحدث عروض السوق اللحظي وتقييمها بـ CatBoost..."):
        results_data, is_relaxed, notice_str = hybrid_search(user_query=combined_q, top_k=8)

    rendered_cards = generate_user_search_html(combined_q, results_data, is_relaxed, notice_str)
    c_height = max(400, len(results_data) * 320 + 200)
    components.html(rendered_cards, height=c_height, scrolling=True)
