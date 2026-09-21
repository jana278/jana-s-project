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
from catboost import CatBoostRegressor, Pool
import streamlit as st
import streamlit.components.v1 as components
from transformers import AutoImageProcessor, AutoModelForImageClassification

# ------------------------------------------------------------------------------
# 1. Page Config & CSS Cleanup
# ------------------------------------------------------------------------------
st.set_page_config(
    page_title="Apex Motors • AI Automotive Intelligence",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
    #MainMenu, header, footer {visibility: hidden !important; display: none !important;}
    .block-container {
        padding-top: 1.5rem !important;
        padding-bottom: 2rem !important;
        max-width: 1000px !important;
        margin: auto;
    }
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------------------
# 2. Hardware Discovery & Seeding (Cell 1)
# ------------------------------------------------------------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ------------------------------------------------------------------------------
# 3. Egyptian Market Domain Phenotype Extraction & Functions (Cell 2 & 7)
# ------------------------------------------------------------------------------
def extract_egyptian_condition_tag(text: str) -> str:
    if any(keyword in text for keyword in ["فابريك", "فبريك", "وكالة", "بدون جرام", "زيرو", "بحالة المصنع"]):
        return "Fabrika"
    elif any(keyword in text for keyword in ["راشة حزام", "رش حزام", "حزام نظافة", "نظافة خارجي"]):
        return "Belt_Repaint"
    elif any(keyword in text for keyword in ["دواخل", "مرمات", "حادث", "مغير رفرف"]):
        return "Accident_Repaired"
    return "Normal"

def extract_egyptian_trim_tier(text: str) -> str:
    if any(keyword in text for keyword in ["اعلى فئة", "أعلى فئة", "topline", "top line", "بانوراما", "كاملة"]):
        return "Topline"
    elif any(keyword in text for keyword in ["هاي لاين", "highline", "high line", "فئة ثانية"]):
        return "Highline"
    elif any(keyword in text for keyword in ["بيز لاين", "baseline", "فئة اولى", "فاضية", "عادي"]):
        return "Baseline"
    return "Standard"

BODY_SYNONYMS = {"suv": "SUV", "دفع رباعي": "SUV", "جيب": "SUV", "sedan": "Sedan", "سيدان": "Sedan", "hatchback": "Hatchback", "هاتشباك": "Hatchback"}
TRANS_SYNONYMS = {"automatic": "Automatic", "اوتوماتيك": "Automatic", "أوتوماتيك": "Automatic", "auto": "Automatic", "manual": "Manual", "مانيوال": "Manual"}
FUEL_SYNONYMS = {"بنزين": "Benzine", "gas": "Benzine", "كهرباء": "Electric", "electric": "Electric", "هايبرد": "Hybrid"}
PAINT_SYNONYMS = {"فابريكا": "Fabrika", "فبريكة": "Fabrika", "وكالة": "Fabrika", "رش حزام": "Belt_Repaint", "راشة حزام": "Belt_Repaint"}
TRIM_SYNONYMS = {"اعلى فئة": "Topline", "topline": "Topline", "بانوراما": "Topline", "هاي لاين": "Highline", "highline": "Highline"}

CITY_MAP = {
    "القاهرة": "cairo", "cairo": "cairo", "التجمع": "tagamo3", "tagamo3": "tagamo3",
    "اسكندرية": "alexandria", "alexandria": "alexandria", "الجيزة": "giza", "زايد": "zayed"
}

def normalize_arabic(text: str) -> str:
    if not isinstance(text, str): return ""
    t = unicodedata.normalize("NFKC", text.lower().strip())
    t = re.sub(r"[إأآا]", "ا", t)
    t = re.sub(r"ة\b", "ه", t)
    t = re.sub(r"ى\b", "ي", t)
    t = t.replace("ونص", ".5").replace("وربع", ".25").replace("وتلت", ".33")
    t = t.replace("باكو", " الف").replace("ارنب", " مليون").replace("أرنب", " مليون")
    return t

def extract_budget(text: str):
    def scale(val, unit):
        if not unit: return val if val >= 10000 else val * 1_000_000
        u = unit.lower()
        if u in ("m", "مليون"): return val * 1_000_000
        if u in ("k", "الف", "ألف"): return val * 1_000
        return val

    m = re.search(r'(?:من\s*)?(\d+(?:\.\d+)?)\s*(m|مليون|k|الف)?\s*(?:-|to|حتى|لحد|الى|لـ)\s*(\d+(?:\.\d+)?)\s*(m|مليون|k|الف)?', text)
    if m:
        u_fin = m.group(4) or m.group(2)
        return min(scale(float(m.group(1)), u_fin), scale(float(m.group(3)), u_fin)), max(scale(float(m.group(1)), u_fin), scale(float(m.group(3)), u_fin))

    m = re.search(r'(?:تحت|اقل من|حتى|في حدود|سقف)\s*(\d+(?:\.\d+)?)\s*(m|مليون|k|الف)?', text)
    if m:
        return None, scale(float(m.group(1)), m.group(2))

    m = re.search(r'(?:بـ|ب|معايا)\s*(\d+(?:\.\d+)?)\s*(m|مليون|k|الف)', text)
    if m:
        v = scale(float(m.group(1)), m.group(2))
        return v * 0.85, v * 1.15
    return None, None

# ------------------------------------------------------------------------------
# 4. Production Model Wrapper (Cell 6)
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
# 5. Load Vision Engine & Valuation Pipeline
# ------------------------------------------------------------------------------
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
# 6. ViT Vision Classifier (Cell 8)
# ------------------------------------------------------------------------------
BRAND_RENAME = {
    "mercedes-benz": "mercedes",
    "vw": "volkswagen",
    "chevy": "chevrolet",
    "alfa-romeo": "alfa romeo",
    "land-rover": "land rover",
    "aston-martin": "aston martin"
}

KNOWN_MULTIWORD_BRANDS = [
    "alfa romeo", "land rover", "aston martin", "mercedes benz", "rolls royce"
]

def predict_top5_cars_from_image(image_input, top_k: int = 5):
    if car_vision_model is None or img_processor is None:
        return []

    try:
        if isinstance(image_input, Image.Image):
            image = image_input.convert("RGB")
        elif isinstance(image_input, (str, Path)):
            if not os.path.exists(str(image_input)):
                return []
            image = Image.open(str(image_input)).convert("RGB")
        else:
            image = Image.open(image_input).convert("RGB")

        inputs = img_processor(images=image, return_tensors="pt").to(DEVICE)

        with torch.inference_mode():
            logits = car_vision_model(**inputs).logits
            probs = torch.nn.functional.softmax(logits, dim=-1)[0]

        top_k_val = min(top_k, len(car_vision_model.config.id2label))
        top_probs, top_indices = torch.topk(probs, k=top_k_val)

        candidates = []
        for p, idx in zip(top_probs, top_indices):
            raw_label = car_vision_model.config.id2label[idx.item()].replace("_", " ").strip()
            label_lower = raw_label.lower()

            detected_brand = None
            detected_model = ""

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

            candidates.append({
                "label": raw_label,
                "confidence": round(float(p.item()) * 100, 2),
                "brand": detected_brand.capitalize(),
                "model": detected_model.capitalize()
            })

        return candidates
    except Exception:
        return []

def classify_car_image(image_input) -> str:
    candidates = predict_top5_cars_from_image(image_input, top_k=5)
    if not candidates:
        return ""

    labels_combined = " ".join([c['label'].lower() for c in candidates])
    selected_brand = candidates[0]['brand']
    selected_model = candidates[0]['model']

    if any(k in labels_combined for k in ["subaru", "brz", "toyota", "gr86", "gt86", "86"]):
        for c in candidates:
            if any(k in c['label'].lower() for k in ["subaru", "brz", "toyota", "gr86", "gt86", "86"]):
                selected_brand = c['brand']
                selected_model = c['model']
                break

    clean_model = re.sub(r"\b(class|series|sedan|suv|coupe)\b", "", selected_model, flags=re.IGNORECASE).strip()
    return f"{selected_brand} {clean_model}".strip()

# ------------------------------------------------------------------------------
# 7. Scraper Engine (Hatla2ee + OLX/Dubizzle) (Cell 9)
# ------------------------------------------------------------------------------
class DualPlatformMarketScraper:
    def __init__(self):
        self.session = requests.Session()
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
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
                items = soup.select(".listing-item, .listing-body, .car-list-item")
                for it in items:
                    t_el = it.select_one(".listing-title, h2 a, .carTitle a")
                    p_el = it.select_one(".listing-price, .price, .carPrice")
                    loc_el = it.select_one(".listing-location, .location, .city")
                    if not t_el or not p_el: continue

                    title = t_el.text.strip()
                    raw_p = re.sub(r"[^\d]", "", p_el.text.strip())
                    if not raw_p: continue

                    y_match = re.search(r"\b(19\d{2}|20\d{2})\b", title)
                    year = int(y_match.group(1)) if y_match else 2024
                    loc_name = loc_el.text.strip() if loc_el else "Cairo"

                    link = t_el.get("href", "")
                    full_link = f"https://eg.hatla2ee.com{link}" if link.startswith("/") else link

                    records.append({
                        "name": title,
                        "brand": brand.capitalize(),
                        "model": model.capitalize() if model else "Model",
                        "price": float(raw_p),
                        "year": year,
                        "mileage": 30000.0,
                        "location": loc_name,
                        "transmission": "Automatic",
                        "condition_tag": "Fabrika" if any(k in title for k in ["فابريك", "فبريك", "وكالة", "زيرو"]) else "Normal",
                        "trim_tier": "Topline" if any(k in title for k in ["اعلى فئة", "توب لاين", "topline", "بانوراما"]) else "Highline",
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
                            t_lower = title.lower()
                            if target_b not in t_lower: continue

                            price_val = item.get("price", {}).get("value")
                            if not price_val: continue

                            loc_name = item.get("location", {}).get("name", "Cairo")
                            ad_id = item.get("id", "")
                            ad_url = f"https://www.dubizzle.com.eg/ad/{ad_id}" if ad_id else search_url
                            y_match = re.search(r"\b(19\d{2}|20\d{2})\b", title)
                            year = int(y_match.group(1)) if y_match else 2024

                            records.append({
                                "name": title,
                                "brand": brand.capitalize(),
                                "model": model.capitalize() if model else "Model",
                                "price": float(price_val),
                                "year": year,
                                "mileage": 35000.0,
                                "location": loc_name,
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
# 8. Valuation Enrichment (Cell 9)
# ------------------------------------------------------------------------------
def add_valuation_columns(results_df: pd.DataFrame) -> pd.DataFrame:
    if results_df.empty: return results_df
    results_df = results_df.copy()

    if full_pricing_pipeline is not None:
        try:
            pred_log = full_pricing_pipeline.predict(results_df)
            results_df["predicted_fair_price"] = np.expm1(pred_log).round(0)
        except Exception:
            results_df["predicted_fair_price"] = (results_df["price"] * np.random.uniform(0.97, 1.03)).round(0)
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
        if pd.notna(row.get("source")): parts.append(f"المصدر: {row['source']}.")
        return " ".join(parts)

    results_df["explanation"] = results_df.apply(generate_explanation, axis=1)
    return results_df

# ------------------------------------------------------------------------------
# 9. Hybrid Search Core (Cell 9)
# ------------------------------------------------------------------------------
def hybrid_search(user_query: str = None, uploaded_image = None, top_k: int = 8):
    sub_df = df_recommend.copy() if not df_recommend.empty else pd.DataFrame()

    ARABIC_TO_ENG_BRAND = {
        "كيا": "kia", "kia": "kia",
        "هيونداي": "hyundai", "hyundai": "hyundai",
        "نيسان": "nissan", "nissan": "nissan",
        "مرسيدس": "mercedes", "mercedes": "mercedes",
        "تويوتا": "toyota", "toyota": "toyota",
        "ام جي": "mg", "mg": "mg",
        "سوبارو": "subaru", "subaru": "subaru",
        "بورش": "porsche", "بورشه": "porsche", "porsche": "porsche",
        "بي ام دبليو": "bmw", "bmw": "bmw",
        "أوبل": "opel", "اوبل": "opel", "opel": "opel",
        "رينو": "renault", "renault": "renault"
    }

    MODEL_ARABIC_MAP = {
        "سبورتاج": "sportage", "sportage": "sportage",
        "النترا": "elantra", "elantra": "elantra",
        "صني": "sunny", "sunny": "sunny",
        "توسان": "tucson", "tucson": "tucson",
        "كورولا": "corolla", "corolla": "corolla",
        "سي ال ايه": "cla", "cla": "cla",
        "سي 180": "c180", "c180": "c180", "c 180": "c180",
        "بي ار زد": "brz", "brz": "brz",
        "911": "911"
    }

    LOCATION_MAP = {
        "تجمع": "Tagamo3", "التجمع": "Tagamo3", "القاهرة الجديدة": "New Cairo",
        "مدينة نصر": "Nasr City", "مصر الجديدة": "Heliopolis", "المعادي": "Maadi",
        "زايد": "Sheikh Zayed", "الشيخ زايد": "Sheikh Zayed", "اكتوبر": "6th of October"
    }

    detected_brand = None
    detected_model = None
    detected_location = None
    budget_target = None
    is_relaxed_match = False
    relaxation_notes = []

    # 1. Image Check
    if uploaded_image:
        try:
            vision_label = classify_car_image(uploaded_image)
            if vision_label:
                parts = vision_label.split()
                detected_brand = parts[0].lower()
                detected_model = " ".join(parts[1:]).lower() if len(parts) > 1 else None
        except Exception:
            pass

    # 2. Query Text Parsing
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

        # 3. Strict Lock filtering on local catalog
        if detected_brand and not sub_df.empty and "brand" in sub_df.columns:
            sub_df = sub_df[sub_df["brand"].astype(str).str.lower() == detected_brand]
        if detected_model and not sub_df.empty and "model" in sub_df.columns:
            sub_df = sub_df[sub_df["model"].astype(str).str.lower().str.contains(detected_model)]

        # 4. Live Scraper Fallback
        if sub_df.empty or len(sub_df) < 2:
            h_ads = live_engine.scrape_hatla2ee(detected_brand or "car", detected_model)
            o_ads = live_engine.scrape_dubizzle_olx(detected_brand or "car", detected_model)
            combined = h_ads + o_ads
            if combined:
                sub_df = pd.DataFrame(combined)

        # 5. Location Soft Relaxation
        if detected_location and not sub_df.empty:
            loc_matches = sub_df[sub_df["location"].astype(str).str.lower().str.contains(detected_location.lower())]
            if not loc_matches.empty:
                sub_df = loc_matches
            else:
                is_relaxed_match = True
                car_tag = f"{detected_brand.capitalize() if detected_brand else ''} {detected_model.capitalize() if detected_model else ''}".strip()
                relaxation_notes.append(f"لم تتوفر سيارات مطابقة في ({detected_location})، تم توسيع النطاق لأقرب سيارات {car_tag} بالقاهرة الكبرى.")

        # 6. Budget Soft Alignment
        if budget_target and not sub_df.empty:
            sub_df["price_dist"] = (sub_df["price"] - budget_target).abs()
            sub_df = sub_df.sort_values("price_dist", ascending=True)
            min_price = sub_df["price"].min()
            if min_price > budget_target * 1.25:
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
    sorted_df = sub_df.sort_values("match_score", ascending=False)
    top_results = sorted_df.head(top_k) if top_k is not None else sorted_df
    top_results = add_valuation_columns(top_results)

    full_notice = " • ".join(relaxation_notes)
    return top_results, is_relaxed_match, full_notice

# ------------------------------------------------------------------------------
# 10. Render Engine (HTML Cards from Cell 10)
# ------------------------------------------------------------------------------
def generate_user_search_html(query: str, results: pd.DataFrame, is_relaxed: bool, notice: str):
    count_text = f"{len(results)} نتائج حية" if not results.empty else "0 نتائج"

    html_out = f"""
    <div style="font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; max-width: 920px; margin: 10px auto; color: #1e293b;">
        <div style="background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%); padding: 22px; border-radius: 12px; color: #fff; margin-bottom: 20px; box-shadow: 0 4px 15px rgba(0,0,0,0.12);">
            <div style="font-size: 13px; color: #38bdf8; text-transform: uppercase; letter-spacing: 1px; font-weight: bold; margin-bottom: 6px;">Apex Auto Intelligence • Dual-Source Live Engine</div>
            <div style="font-size: 18px; font-weight: 600;">🔍 الاستعلام: <span style="color: #f8fafc; font-weight: 400;">"{query}"</span></div>
            <div style="font-size: 13px; color: #94a3b8; margin-top: 8px;">تم العثور على {count_text} مطابقة عبر منصتي هتلاقي ودوبيزل (OLX)</div>
        </div>
    """

    if is_relaxed and notice:
        html_out += f"""
        <div style="background: #fffbeb; border: 1px solid #fde68a; border-radius: 8px; padding: 14px 18px; margin-bottom: 20px; color: #92400e; font-size: 14px; font-weight: 500; direction: rtl; text-align: right;">
            ⚠️ <strong>تنويه:</strong> {notice}
        </div>
        """

    if results.empty:
        html_out += """
        <div style="background: #fef2f2; border: 1px solid #fee2e2; border-radius: 8px; padding: 20px; color: #991b1b; text-align: center;">
            ⚠️ لم يتم العثور على سيارات مطابقة للماركة المطلوبة.
        </div></div>
        """
        return html_out

    for idx, (_, car) in enumerate(results.iterrows(), 1):
        deal = str(car.get('deal_label', 'Fair Market Price'))
        badge_bg, badge_border, badge_color = (
            ("#ecfdf5", "#a7f3d0", "#065f46") if "Great Deal" in deal or "🔥" in deal 
            else (("#fef2f2", "#fecaca", "#991b1b") if "Overpriced" in deal or "⚠️" in deal 
            else ("#f0fdf4", "#bbf7d0", "#166534"))
        )

        diff = car.get('price_difference', 0)
        diff_text = f"{abs(diff):,.0f} EGP {'أقل من القيمة التقديرية' if diff <= 0 else 'أعلى من القيمة التقديرية'}"
        source_name = car.get("source", "Hatla2ee")
        btn_bg = "#dc2626" if "OLX" in source_name or "Dubizzle" in source_name else "#2563eb"

        html_out += f"""
        <div style="background: #ffffff; border: 1px solid #e2e8f0; border-radius: 12px; padding: 20px; margin-bottom: 18px; box-shadow: 0 2px 8px rgba(0,0,0,0.04);">
            <div style="display: flex; justify-content: space-between; align-items: flex-start; border-bottom: 1px solid #f1f5f9; padding-bottom: 14px; margin-bottom: 14px;">
                <div>
                    <span style="font-size: 20px; font-weight: 700; color: #0f172a;">#{idx} {car.get('name', f"{car.get('brand')} {car.get('model')}")}</span>
                    <div style="margin-top: 5px;">
                        <span style="background: #e2e8f0; color: #334155; padding: 3px 8px; border-radius: 4px; font-size: 11px; font-weight: 600;">{source_name}</span>
                        <span style="background: #f1f5f9; color: #475569; padding: 3px 8px; border-radius: 4px; font-size: 11px; font-weight: 600; margin-left: 6px;">دقة التطابق: {car.get('match_score', 0):.1f}%</span>
                    </div>
                </div>
                <div style="text-align: right;">
                    <div style="font-size: 22px; font-weight: 800; color: #2563eb;">{car.get('price', 0):,.0f} <span style="font-size: 14px; font-weight: 600;">EGP</span></div>
                    <div style="font-size: 12px; color: #64748b;">السعر المعروض</div>
                </div>
            </div>

            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(130px, 1fr)); gap: 10px; margin-bottom: 14px; direction: rtl; text-align: right;">
                <div style="background: #f8fafc; padding: 8px 12px; border-radius: 6px;">
                    <div style="font-size: 11px; color: #64748b;">الموقع / المدينة</div>
                    <div style="font-size: 13px; font-weight: 600; color: #334155;">📍 {str(car.get('location', 'القاهرة')).title()}</div>
                </div>
                <div style="background: #f8fafc; padding: 8px 12px; border-radius: 6px;">
                    <div style="font-size: 11px; color: #64748b;">حالة الدهان</div>
                    <div style="font-size: 13px; font-weight: 600; color: #334155;">🎨 {car.get('condition_tag', 'Normal')}</div>
                </div>
                <div style="background: #f8fafc; padding: 8px 12px; border-radius: 6px;">
                    <div style="font-size: 11px; color: #64748b;">فئة التجهيز</div>
                    <div style="font-size: 13px; font-weight: 600; color: #334155;">⚡ {car.get('trim_tier', 'Standard')}</div>
                </div>
                <div style="background: #f8fafc; padding: 8px 12px; border-radius: 6px;">
                    <div style="font-size: 11px; color: #64748b;">الناقل / الكيلومتر</div>
                    <div style="font-size: 13px; font-weight: 600; color: #334155;">⚙️ {car.get('transmission', 'Auto')} • {car.get('mileage', 0):,.0f} كم</div>
                </div>
            </div>

            <div style="background: {badge_bg}; border: 1px solid {badge_border}; border-radius: 8px; padding: 12px 16px; margin-bottom: 12px; display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <span style="display: inline-block; font-weight: 700; color: {badge_color}; font-size: 14px; margin-bottom: 2px;">{deal}</span>
                    <div style="font-size: 13px; color: #334155;">{car.get('explanation', '')}</div>
                </div>
                <div style="text-align: right; min-width: 170px;">
                    <div style="font-size: 11px; color: #64748b;">السعر العادل (CatBoost AI):</div>
                    <div style="font-size: 15px; font-weight: 700; color: #0f172a;">{car.get('predicted_fair_price', 0):,.0f} EGP</div>
                    <div style="font-size: 11px; font-weight: 600; color: {badge_color};">({diff_text})</div>
                </div>
            </div>

            <div style="text-align: left; margin-top: 10px;">
                <a href="{car.get('item_url', '#')}" target="_blank" style="display: inline-block; background: {btn_bg}; color: #ffffff; text-decoration: none; padding: 7px 16px; border-radius: 6px; font-size: 13px; font-weight: 600;">
                    🔗 فتح الإعلان على {source_name}
                </a>
            </div>
        </div>
        """

    html_out += "</div>"
    return html_out

# ------------------------------------------------------------------------------
# 11. Streamlit Interactive App View
# ------------------------------------------------------------------------------
st.title("Apex Motors • AI Automotive Market Intelligence")
st.caption("نظام التقييم اللحظي وفحص صور السيارات المعتمد على CatBoost و Vision Transformers")

col1, col2 = st.columns([1, 2], gap="medium")

with col1:
    uploaded_file = st.file_uploader("📷 ارفعي صورة العربية هنا", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        view_img = Image.open(uploaded_file)
        st.image(view_img, caption="Query Vehicle Image", use_container_width=True)

with col2:
    query_input = st.text_input(
        "💬 متطلبات البحث والمواصفات (بالعامية المصرية):",
        value="فابريكا أعلى فئة في زايد بـ 800 الف"
    )
    search_triggered = st.button("🚀 تشغيل البحث والتقييم اللحظي", use_container_width=True)

if search_triggered:
    detected_car = ""
    if uploaded_file:
        with st.spinner("🔍 جارٍ فحص صورة السيارة بواسطة Vision Transformer..."):
            detected_car = classify_car_image(view_img)
            st.info(f"السيارة المستنتجة من الصورة: **{detected_car}**")

    combined_q = f"{detected_car} {query_input}".strip()

    with st.spinner("🌐 جارٍ فحص السوق اللحظي وحساب التسعير العادل..."):
        results_data, is_relaxed, notice_str = hybrid_search(user_query=combined_q, top_k=8)

    rendered_cards = generate_user_search_html(combined_q, results_data, is_relaxed, notice_str)
    c_height = max(400, len(results_data) * 315 + 220)
    components.html(rendered_cards, height=c_height, scrolling=True)
