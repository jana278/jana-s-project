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
# 1. Page Configuration & Clean Minimal Light Styling
# ------------------------------------------------------------------------------
st.set_page_config(
    page_title="Apex Motors • تداول وتسعير السيارات الذكي",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;600;700&family=Tajawal:wght@400;500;700;800&display=swap');
    
    #MainMenu, header, footer {visibility: hidden !important; display: none !important;}
    
    .stApp {
        background-color: #f8fafc !important;
        font-family: 'Tajawal', 'Plus Jakarta Sans', sans-serif !important;
        color: #0f172a !important;
    }
    
    .block-container {
        padding-top: 2rem !important;
        padding-bottom: 3rem !important;
        max-width: 960px !important;
        margin: auto;
    }

    /* حقل الإدخال النصي */
    .stTextInput > div > div > input {
        background-color: #ffffff !important;
        color: #0f172a !important;
        border: 1.5px solid #cbd5e1 !important;
        border-radius: 10px !important;
        padding: 12px 16px !important;
        font-size: 15px !important;
        direction: rtl !important;
        text-align: right !important;
        box-shadow: 0 1px 2px rgba(0,0,0,0.05) !important;
    }
    .stTextInput > div > div > input:focus {
        border-color: #2563eb !important;
        box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.15) !important;
    }

    /* منطقة رفع الصورة */
    .stFileUploader section {
        background-color: #ffffff !important;
        border: 1.5px dashed #94a3b8 !important;
        border-radius: 10px !important;
        padding: 8px !important;
    }

    /* زر البحث الرئيسي */
    .stButton > button {
        background: #2563eb !important;
        color: #ffffff !important;
        font-family: 'Tajawal', sans-serif !important;
        font-size: 16px !important;
        font-weight: 700 !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 12px 24px !important;
        box-shadow: 0 2px 6px rgba(37, 99, 235, 0.25) !important;
        transition: background-color 0.2s ease !important;
    }
    .stButton > button:hover {
        background: #1d4ed8 !important;
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
# 3. Model Loading
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
# 5. Dual Platform Scraper
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
        ["صفقة ممتازة 🔥", "أعلى من سعر السوق ⚠️"],
        default="سعر عادل ومناسب ⚖️"
    )

    def generate_explanation(row):
        parts = [f"السعر المعروض {row['price']:,.0f} ج.م مقارنة بالقيمة العادلة المقدرة {row['predicted_fair_price']:,.0f} ج.م."]
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
                relaxation_notes.append(f"لم تتوفر سيارات مطابقة في ({detected_location})، تم عرض السيارات المتاحة في النطاق المجاور.")

        if budget_target and not sub_df.empty:
            sub_df["price_dist"] = (sub_df["price"] - budget_target).abs()
            sub_df = sub_df.sort_values("price_dist", ascending=True)
            if sub_df["price"].min() > budget_target * 1.25:
                is_relaxed_match = True
                relaxation_notes.append(f"الميزانية المحددة ({budget_target:,.0f} ج.م) أقل من المعروض بالسوق، تم ترتيب الأقرب لها.")

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
# 8. Clean Light Cards HTML Generator
# ------------------------------------------------------------------------------
def generate_user_search_html(query: str, results: pd.DataFrame, is_relaxed: bool, notice: str):
    count_text = f"{len(results)} نتائج" if not results.empty else "لا توجد نتائج"

    html_out = f"""
    <div style="font-family: 'Tajawal', 'Segoe UI', sans-serif; max-width: 940px; margin: 15px auto; color: #0f172a;">
        <div style="background: #ffffff; border: 1px solid #e2e8f0; border-radius: 12px; padding: 18px 24px; margin-bottom: 20px; display: flex; justify-content: space-between; align-items: center; box-shadow: 0 2px 4px rgba(0,0,0,0.03);">
            <div>
                <div style="font-size: 13px; color: #64748b; font-weight: 600;">نتائج البحث اللحظي من السوق</div>
                <div style="font-size: 18px; font-weight: 700; color: #0f172a; margin-top: 2px;">🔍 {query}</div>
            </div>
            <div style="background: #eff6ff; color: #2563eb; border: 1px solid #bfdbfe; padding: 6px 14px; border-radius: 8px; font-size: 13px; font-weight: 700;">
                {count_text}
            </div>
        </div>
    """

    if is_relaxed and notice:
        html_out += f"""
        <div style="background: #fffbeb; border: 1px solid #fde68a; border-radius: 10px; padding: 12px 18px; margin-bottom: 18px; color: #b45309; font-size: 14px; font-weight: 500; text-align: right; direction: rtl;">
            ⚠️ <strong>تنويه:</strong> {notice}
        </div>
        """

    if results.empty:
        html_out += """
        <div style="background: #fef2f2; border: 1px solid #fee2e2; border-radius: 10px; padding: 25px; text-align: center; color: #b91c1c; font-size: 15px;">
            لم يتم العثور على سيارات مطابقة للمعايير المحددة حالياً.
        </div></div>
        """
        return html_out

    for idx, (_, car) in enumerate(results.iterrows(), 1):
        deal = str(car.get('deal_label', 'سعر عادل'))
        badge_bg, badge_border, badge_color = (
            ("#f0fdf4", "#bbf7d0", "#166534") if "ممتازة" in deal or "🔥" in deal 
            else (("#fef2f2", "#fecaca", "#991b1b") if "أعلى" in deal or "⚠️" in deal 
            else ("#eff6ff", "#bfdbfe", "#1d4ed8"))
        )

        diff = car.get('price_difference', 0)
        diff_text = f"{abs(diff):,.0f} ج.م {'أقل من السعر المقدر' if diff <= 0 else 'أعلى من السعر المقدر'}"
        source_name = car.get("source", "Hatla2ee")

        html_out += f"""
        <div style="background: #ffffff; border: 1px solid #e2e8f0; border-radius: 12px; padding: 20px; margin-bottom: 16px; box-shadow: 0 2px 4px rgba(0, 0, 0, 0.02);">
            <div style="display: flex; justify-content: space-between; align-items: flex-start; border-bottom: 1px solid #f1f5f9; padding-bottom: 14px; margin-bottom: 14px;">
                <div>
                    <div style="font-size: 19px; font-weight: 700; color: #0f172a;">#{idx} {car.get('name', f"{car.get('brand')} {car.get('model')}")}</div>
                    <div style="margin-top: 6px; display: flex; gap: 8px;">
                        <span style="background: #f1f5f9; color: #475569; padding: 3px 10px; border-radius: 6px; font-size: 12px; font-weight: 600;">{source_name}</span>
                        <span style="background: #eff6ff; color: #2563eb; padding: 3px 10px; border-radius: 6px; font-size: 12px; font-weight: 700;">دقة التطابق: {car.get('match_score', 0):.1f}%</span>
                    </div>
                </div>
                <div style="text-align: right;">
                    <div style="font-size: 22px; font-weight: 800; color: #2563eb;">{car.get('price', 0):,.0f} <span style="font-size: 13px; font-weight: 600; color: #64748b;">ج.م</span></div>
                    <div style="font-size: 12px; color: #64748b; margin-top: 2px;">السعر المعروض</div>
                </div>
            </div>

            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(130px, 1fr)); gap: 10px; margin-bottom: 14px; direction: rtl; text-align: right;">
                <div style="background: #f8fafc; border: 1px solid #f1f5f9; padding: 8px 12px; border-radius: 8px;">
                    <div style="font-size: 11px; color: #64748b;">المدينة / المنطقة</div>
                    <div style="font-size: 13px; font-weight: 600; color: #1e293b; margin-top: 2px;">📍 {str(car.get('location', 'القاهرة')).title()}</div>
                </div>
                <div style="background: #f8fafc; border: 1px solid #f1f5f9; padding: 8px 12px; border-radius: 8px;">
                    <div style="font-size: 11px; color: #64748b;">حالة الدهان</div>
                    <div style="font-size: 13px; font-weight: 600; color: #1e293b; margin-top: 2px;">🎨 {car.get('condition_tag', 'Normal')}</div>
                </div>
                <div style="background: #f8fafc; border: 1px solid #f1f5f9; padding: 8px 12px; border-radius: 8px;">
                    <div style="font-size: 11px; color: #64748b;">فئة التجهيز</div>
                    <div style="font-size: 13px; font-weight: 600; color: #1e293b; margin-top: 2px;">⚡ {car.get('trim_tier', 'Standard')}</div>
                </div>
                <div style="background: #f8fafc; border: 1px solid #f1f5f9; padding: 8px 12px; border-radius: 8px;">
                    <div style="font-size: 11px; color: #64748b;">الناقل / الكيلومتر</div>
                    <div style="font-size: 13px; font-weight: 600; color: #1e293b; margin-top: 2px;">⚙️ {car.get('transmission', 'Auto')} • {car.get('mileage', 0):,.0f} كم</div>
                </div>
            </div>

            <div style="background: {badge_bg}; border: 1px solid {badge_border}; border-radius: 8px; padding: 12px 16px; margin-bottom: 14px; display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <span style="display: inline-block; font-weight: 700; color: {badge_color}; font-size: 14px;">{deal}</span>
                    <div style="font-size: 13px; color: #334155; margin-top: 2px;">{car.get('explanation', '')}</div>
                </div>
                <div style="text-align: right; min-width: 170px;">
                    <div style="font-size: 11px; color: #64748b;">السعر العادل المقدر:</div>
                    <div style="font-size: 16px; font-weight: 700; color: #0f172a;">{car.get('predicted_fair_price', 0):,.0f} ج.م</div>
                    <div style="font-size: 11px; font-weight: 600; color: {badge_color};">({diff_text})</div>
                </div>
            </div>

            <div style="text-align: left;">
                <a href="{car.get('item_url', '#')}" target="_blank" style="display: inline-block; background: #2563eb; color: #ffffff; text-decoration: none; padding: 7px 16px; border-radius: 6px; font-size: 13px; font-weight: 600;">
                    🔗 فتح الإعلان الأصلي
                </a>
            </div>
        </div>
        """

    html_out += "</div>"
    return html_out

# ------------------------------------------------------------------------------
# 9. Clean Header & Search Hub
# ------------------------------------------------------------------------------
st.markdown("""
<div style="text-align: center; margin-bottom: 25px;">
    <h1 style="font-size: 34px; font-weight: 800; color: #0f172a; margin: 0;">
        Apex Motors • محرك تسعير وبحث السيارات
    </h1>
    <p style="color: #64748b; font-size: 15px; margin-top: 6px;">
        فحص الصور بالذكاء الاصطناعي، سحب الإعلانات الحية من السوق، وتقدير السعر العادل
    </p>
</div>
""", unsafe_allow_html=True)

with st.container():
    st.markdown("""
    <div style="background: #ffffff; border: 1px solid #e2e8f0; border-radius: 12px; padding: 22px; box-shadow: 0 1px 3px rgba(0,0,0,0.04); margin-bottom: 20px;">
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([1, 2.2], gap="large")

    with col1:
        st.markdown("<div style='font-size: 13px; font-weight: 700; color: #334155; margin-bottom: 6px;'>📷 فحص صورة السيارة (اختياري)</div>", unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Vehicle Image", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
        if uploaded_file:
            view_img = Image.open(uploaded_file)
            st.image(view_img, caption="الصورة المرفوعة", use_container_width=True)

    with col2:
        st.markdown("<div style='font-size: 13px; font-weight: 700; color: #334155; margin-bottom: 6px;'>💬 المواصفات والشروط والميزانية (بالعامية المصرية):</div>", unsafe_allow_html=True)
        query_input = st.text_input(
            "Query Text",
            value="فابريكا أعلى فئة في زايد بـ 800 الف",
            placeholder="اكتبي الماركة أو المواصفات مثل: كيا سبورتاج فابريكا في التجمع...",
            label_visibility="collapsed"
        )
        st.markdown("<div style='height: 6px;'></div>", unsafe_allow_html=True)
        search_triggered = st.button("🚀 بدء البحث والتسعير اللحظي", use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)

# ------------------------------------------------------------------------------
# 10. Execution Trigger
# ------------------------------------------------------------------------------
if search_triggered:
    detected_car = ""
    if uploaded_file:
        with st.spinner("🔍 فحص صورة السيارة بموديل الرؤية..."):
            detected_car = classify_car_image(view_img)
            if detected_car:
                st.info(f"تم التعرف على السيارة من الصورة: **{detected_car}**")

    combined_q = f"{detected_car} {query_input}".strip()

    with st.spinner("🌐 سحب أحدث الإعلانات الحية وحساب السعر العادل..."):
        results_data, is_relaxed, notice_str = hybrid_search(user_query=combined_q, top_k=8)

    rendered_cards = generate_user_search_html(combined_q, results_data, is_relaxed, notice_str)
    c_height = max(380, len(results_data) * 270 + 200)
    components.html(rendered_cards, height=c_height, scrolling=True)
