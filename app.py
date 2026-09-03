# -*- coding: utf-8 -*-
"""
app.py
======
داشبورد Streamlit. فقط UI است؛ هیچ منطق مدلی اینجا نیست — همه چیز از طریق
درخواست HTTP به api.py (FastAPI) انجام می‌شود.

اجرا (لوکال):
    streamlit run app.py
"""

import os

import requests
import streamlit as st

API_URL = os.environ.get("API_URL", "http://localhost:8000")

st.set_page_config(page_title="تخمین قیمت آپارتمان", page_icon="🏠", layout="centered")

st.title("🏠 تخمین قیمت آپارتمان")
st.caption("این مدل روی آگهی‌های فروش آپارتمان دیوار آموزش دیده و صرفاً یک برآورد تقریبی ارائه می‌دهد.")

with st.form("predict_form"):
    col1, col2 = st.columns(2)
    with col1:
        city = st.text_input("شهر *", placeholder="مثلاً اصفهان")
        area = st.number_input("متراژ (متر مربع) *", min_value=1.0, value=80.0, step=1.0)
        rooms = st.number_input("تعداد اتاق", min_value=0, max_value=10, value=2, step=1)
        built_year = st.number_input(
            "سال ساخت (شمسی)", min_value=1300, max_value=1410, value=1399, step=1
        )
    with col2:
        district = st.text_input("محله / منطقه", placeholder="مثلاً ملک‌شهر")
        floor = st.number_input("طبقه", min_value=-1, max_value=40, value=2, step=1)
        features_text = st.text_input(
            "ویژگی‌ها (اختیاری)", placeholder="پارکینگ، انباری، آسانسور"
        )

    description = st.text_area(
        "توضیحات آگهی (اختیاری، ولی هرچه کامل‌تر باشد دقت بالاتر می‌رود)",
        placeholder="واحد نوساز با کابینت ام‌دی‌اف و کف سرامیک، دارای بالکن و دوربین مداربسته...",
        height=120,
    )

    submitted = st.form_submit_button("محاسبه قیمت", use_container_width=True)

if submitted:
    if not city or not area:
        st.error("لطفاً فیلدهای ستاره‌دار (شهر، متراژ) را پر کنید.")
    else:
        payload = {
            "features_text": features_text,
            "description": description,
            "rooms": int(rooms),
            "floor": int(floor),
            "area": float(area),
            "built_year_shamsi": int(built_year),
            "city": city,
            "district": district,
        }
        try:
            with st.spinner("در حال محاسبه..."):
                resp = requests.post(f"{API_URL}/predict", json=payload, timeout=15)
            if resp.status_code == 200:
                data = resp.json()
                price = data["predicted_price_toman"]
                price_per_meter = data["predicted_price_per_meter_toman"]

                st.success("محاسبه انجام شد")
                m1, m2 = st.columns(2)
                m1.metric("قیمت تخمینی کل", f"{price:,.0f} تومان")
                m2.metric("قیمت هر متر", f"{price_per_meter:,.0f} تومان")

                if data.get("note"):
                    st.info(data["note"])
            else:
                st.error(f"خطا از سمت API ({resp.status_code}): {resp.json().get('detail', resp.text)}")
        except requests.exceptions.ConnectionError:
            st.error(
                f"اتصال به API برقرار نشد. مطمئن شو api.py روی {API_URL} در حال اجراست "
                f"(`uvicorn api:app --reload --port 8000`)."
            )